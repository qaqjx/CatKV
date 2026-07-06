from multiprocessing.pool import ThreadPool
import os
from sys import flags
import threading
import time
from typing import Any, Dict, List, Optional

import torch


from catkv.kv_manager.compress.abstract_compress import CompressType
from catkv.kv_manager.compress.utils import CompressFactory
from catkv.kv_manager.cpu_bufferpool import MAX_BUFFER_SIZE
from catkv.kv_manager.db import DataCenter, OffloadMode
from catkv.kv_manager.disk.safe_tensor import IOMode
from catkv.kv_manager.utils import (
    _catkv_nvtx_annotate,
    get_kvcache_filename,
    get_shared_key_sv_filename,
    uuid_to_tensor,
)

class MemoryPinnedBuffer:
    """
    A class to manage pinned memory buffers.
    """
    MAX_PINNED_MEMORY = 16 * 1024 * 1024 * 1024  # 16 GB

    def __init__(self, shape):
        assert shape != None
        self.max_len = self.MAX_PINNED_MEMORY // (shape[2] * shape[3] * 2 * 2) 
        shape = (shape[0] , self.max_len , shape[2] , shape[3])

        self.shape = shape
        self.buffer = torch.empty(self.shape, dtype=torch.bfloat16).pin_memory()
        self.offset = 0

    def allocate(self, size: int):
        """
            Allocate a buffer of the given size from the pinned memory.
            if not enough memory, reset the buffer.
        """

        if self.offset + size > self.max_len:
            self.reset()
        start = self.offset
        self.offset += size
        return self.buffer[: , start:self.offset]

    def reset(self):
        self.offset = 0

class KVCacheManager:
    """
    A singleton class to manage key-value cache.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, shape = None, layer_num = None, device = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = KVCacheManager(shape = shape, layer_num = layer_num , device = device)
        return cls._instance
    
    @classmethod
    def clean_instance(cls):
        if cls._instance is not None:
            cls._instance.clean()

    @classmethod
    def flush(cls):
        if cls._instance is not None:
            if cls._instance._offload_manager is not None:
                cls._instance.offload_manager.flush()

    def __init__(
        self,
        layer_num: int,
        device="cuda",
        shape=None,
        max_buffer_size=MAX_BUFFER_SIZE,
        offload_mode=OffloadMode.DISK,
        io_mode=IOMode.S3,
        compress_type=CompressType.NONE,
        compress_config=None,
        io_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the KVCacheManager with specified parameters.

        Args:
            layer_num: Number of layers in the model.
            device: Device to use for computation.
            shape: Shape of the KV cache tensors.
            max_buffer_size: Maximum buffer size for CPU buffer pool.
            offload_mode: Offload mode (CPU or DISK).
            io_mode: IO mode (SAFETENSOR or S3).
            compress_type: Compression type.
            compress_config: Compression configuration.
            io_config: IO configuration (e.g., {"config_path": "path/to/s3.ini"} for S3).
        """
        self.device = device
        self.max_buffer_size = max_buffer_size
        self.offload_mode = offload_mode
        self.io_mode = io_mode
        self.io_config = io_config or {}
        self.db = DataCenter(
            max_buffer_size=max_buffer_size,
            offload_mode=offload_mode,
            io_mode=io_mode,
            device=device,
            io_config=self.io_config,
        )
        self.compress_type = compress_type
        self.layer_num = layer_num
        self.compress_config = compress_config or {}
        self.compressor = CompressFactory.create_compressor(compress_type, self.compress_config, layer_num=layer_num , device=self.device)

        self.pinned_buffer_allocator = MemoryPinnedBuffer(shape)
        
        # self.transfer_stream = torch.cuda.Stream(device=self.device)

        self.keys_set = set()
        self.init_keys_set()

        self._offload_manager = None
        if self.compress_type == CompressType.OURS:
            self._init_offload_manager()

    

    def init_keys_set(self):
        # Initialize the set of keys in the KV cache by read the kvcache dir
        import catkv.kv_manager.utils as kv_utils
        # ensure the directory exists
        if not os.path.exists(kv_utils.store_kvcache_dir):
            os.makedirs(kv_utils.store_kvcache_dir)

        file_names = os.listdir(kv_utils.store_kvcache_dir)

        # for file_name in file_names:
        #     if "layer_0" in file_name:
        #         self.keys_set.add(file_name)
        

    def _init_offload_manager(self):
        from catkv.kv_manager.offload_worker import OffloadManager

        compressor_kwargs = {
            "ratio": self.compress_config.get("ratio", 0.2),
            "dtype": torch.bfloat16,
        }
        db_kwargs = {
            "max_buffer_size": self.max_buffer_size,
            "offload_mode": self.offload_mode,
            "io_mode": self.io_mode,
            "device": "cpu",
            "io_config": self.io_config,
        }
        shape = self.pinned_buffer_allocator.shape
        self._offload_manager = OffloadManager(
            compressor_kwargs=compressor_kwargs,
            db_kwargs=db_kwargs,
            device=self.device,
            num_workers=4,  # Use 4 workers for parallel compression
            num_heads=shape[2],
            head_dim=shape[3],
        )
        self._offload_manager.start()

    def check_keys_exist(self, keys: list) -> list[bool]:
        """
        Check if the given keys exist in the KV cache.
        """
        if self._offload_manager is not None:
            new_keys = self._offload_manager.drain_keys_ack()
            self.keys_set.update(new_keys)

        return [key.split("/")[-1] in self.keys_set for key in keys]

    def flush_offload(self, timeout: float = 30.0) -> None:
        """Wait for all pending offload tasks to complete.

        This method blocks until all compression and storage tasks submitted
        to the offload manager have been processed.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._offload_manager is not None:
            self._offload_manager.flush(timeout=timeout)
            # Sync keys_set after flush
            new_keys = self._offload_manager.drain_keys_ack()
            self.keys_set.update(new_keys)

    def set_compress_type(self, compress_type: CompressType, compress_config=None):
        """
        Set the compression type for the KV cache.
        """
        self.compress_type = compress_type
        self.compress_config = compress_config or {}
        self.compressor = CompressFactory.create_compressor(compress_type, self.compress_config, layer_num=self.layer_num, device=self.device)
        # if self.compress_type == CompressType.OURS:
        #     self._init_offload_manager()
        print(f"KVCacheManager initialized with compress type: {self.compress_type}")

    @_catkv_nvtx_annotate("retrieve_data")
    def retrieve_data(self, key: str , kv, layer_idx: int , stream=None):
        """
        Load data from a file.
        """
        data = self.db.retrieve_data(key)
      
        with _catkv_nvtx_annotate("decompress_inner"):
            data = self.compressor.decompress(data , kv , layer_idx )
        
        return  data
    
    def store_data(self, key: str, data , score = None , layer_idx: int = None):
        """
        Save data to a file or cpu.
        """
        if self._offload_manager is not None:
            self._offload_manager.submit_store_data(key, data, score, layer_idx)
            if "layer_0" in key:
                self.keys_set.add(key.split("/")[-1])
            return True

        data = self.compressor.compress(data , score , layer_idx=layer_idx)
        flag = self.db.store_data(key, data)

        if flag and "layer_0" in key:
            self.keys_set.add(key.split("/")[-1])
            print(f"Stored key: {key}, current keys_set size: {len(self.keys_set)}")

        return  flag
    
    def store_multi_data(
        self,
        key_hash_list: List[str],
        indices = None,
        layer_idx = None,
        device = None,
        kv = None,
        group_uuid = None,
    ):
        """
        Save multi-layer data to files or cpu.
        """
        if isinstance(key_hash_list , str):
            key_hash_list = [key_hash_list]

        compress_kwargs = {"indices": indices}
        if self.compress_type == CompressType.OURS:
            if group_uuid is None:
                group_uuid = uuid_to_tensor(
                    ",".join(str(key_hash) for key_hash in key_hash_list)
                )
            compress_kwargs["uuid"] = group_uuid
        compress_data = self.compressor.compress_multi(kv, **compress_kwargs)

        def key_sv_rank(payload):
            key_sv = payload["key_sv_quantized"]
            return key_sv.shape[-2] if key_sv.dim() >= 2 else key_sv.numel()

        split_payloads = []
        shared_key_sv_candidates = {}
        for idx , key_hash in enumerate(key_hash_list):
            key = get_kvcache_filename( key_hash , layer_idx=layer_idx ,device=device)
            if self.compress_type == CompressType.OURS:
                # split u and sv
                key_sv_compress_data = {
                    "key_sv_quantized": compress_data[idx]["key_sv_quantized"],
                    "key_sv_meta": compress_data[idx]["key_sv_meta"],
                    "key_residual_sv": compress_data[idx]["key_residual_sv"]
                }

                other_compress_data = {
                    "u_quantized": compress_data[idx]["u_quantized"],
                    "u_meta": compress_data[idx]["u_meta"],
                    "value_sv_quantized": compress_data[idx]["value_sv_quantized"],
                    "value_sv_meta": compress_data[idx]["value_sv_meta"],
                    "value_residual_sv": compress_data[idx]["value_residual_sv"]
                }

                if "uuid" in compress_data[idx]:
                    key_sv_compress_data["uuid"] = compress_data[idx]["uuid"]
                    other_compress_data["uuid"] = compress_data[idx]["uuid"]

                key_sv_path = key + "_key_sv"
                if layer_idx is not None and layer_idx > 2 and "uuid" in key_sv_compress_data:
                    key_sv_path = get_shared_key_sv_filename(
                        key_sv_compress_data["uuid"],
                        layer_idx=layer_idx,
                        base_key=key,
                    )
                if key_sv_path != key + "_key_sv":
                    existing = shared_key_sv_candidates.get(key_sv_path)
                    if (
                        existing is None
                        or key_sv_rank(key_sv_compress_data) > key_sv_rank(existing)
                    ):
                        shared_key_sv_candidates[key_sv_path] = key_sv_compress_data
                split_payloads.append((key, key_sv_path, key_sv_compress_data, other_compress_data))
            else:
                flag = self.db.store_data(key, compress_data[idx])
            if "layer_0" in key:
                self.keys_set.add(key.split("/")[-1])
        stored_shared_key_sv_paths = set()
        for key, key_sv_path, key_sv_compress_data, other_compress_data in split_payloads:
            if key_sv_path == key + "_key_sv":
                flag = self.db.store_data(key_sv_path, key_sv_compress_data)
            elif key_sv_path not in stored_shared_key_sv_paths:
                flag = self.db.store_data(
                    key_sv_path,
                    shared_key_sv_candidates[key_sv_path],
                )
                stored_shared_key_sv_paths.add(key_sv_path)
            flag = self.db.store_data(key + "_other", other_compress_data)
            # print(f"Stored key: {key}_key_sv and {key}_other, current keys_set size: {len(self.keys_set)}")
        return None

    def offload_compress_data(self , key: str, data: List[torch.Tensor]):
        if "layer_0" in key:
            self.keys_set.add(key.split("/")[-1])
            print(f"Stored key: {key}, current keys_set size: {len(self.keys_set)}")    
        sequence_len = data[0].size(1)

        # allocate pinned memory
        key_pinned = self.pinned_buffer_allocator.allocate(sequence_len)
        value_pinned = self.pinned_buffer_allocator.allocate(sequence_len)

        # copy to pinned memory
        with _catkv_nvtx_annotate("copy_to_pinned_memory"):
            key_pinned.copy_(data[0] , non_blocking=True)
            value_pinned.copy_(data[1] , non_blocking=True)

        torch.cuda.current_stream(device=self.device).synchronize()

        if self._offload_manager is not None:
            self._offload_manager.submit_offload_compress(
                key, [key_pinned, value_pinned]
            )
            return True

        data = {"key": key_pinned , "value": value_pinned}
        flag = self.db.store_data(key , data , compress_flag=True)

        return flag
        
    def retrieve_by_task_id(self, task_id):
        compress_data_cpu , compress_data_gpu = self.db.retrive_by_task(task_id)
        return compress_data_gpu

    def retrieve_keys(self , keys: List[str]):
        task_id = self.db.retrieve_keys(keys)
        if isinstance(task_id, list):
            result = []
            for data in task_id:
                result.append(self.compressor.transfer(data))
            return result
        return task_id

    def decompress(self, compressed_data , kv_len):
        return self.compressor.decompress(compressed_data , kv_len)

    def clean(self):
        if self._offload_manager is not None:
            self._offload_manager.shutdown()
            self._offload_manager = None
        self.db.clean()

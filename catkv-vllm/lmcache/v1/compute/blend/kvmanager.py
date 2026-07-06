from asyncio.log import logger
from concurrent.futures import Future
import queue
import os
import threading
import time
from typing import Any, Dict, List, Optional

import torch

from lmcache.v1.compute.blend.compress.abstract import CompressType

# Profiling toggle
ENABLE_PROFILING = os.environ.get("LMCACHE_ENABLE_PROFILING", "0") == "1"

def profile_log(msg: str, *args, **kwargs):
    if ENABLE_PROFILING:
        print(f"[PROFILE] {msg}", flush=True)


def get_compress_type_from_env() -> CompressType:
    """
    Get compression type from environment variable LMCACHE_COMPRESS_TYPE.
    Supported values: NONE, OURS
    Default: OURS
    """
    env_value = os.environ.get("LMCACHE_COMPRESS_TYPE", "OURS").upper()
    return _normalize_compress_type(env_value)


def _normalize_compress_type(value) -> CompressType:
    if isinstance(value, CompressType):
        return value
    env_value = str(value).upper()
    compress_map = {
        "NONE": CompressType.NONE,
        "OURS": CompressType.OURS,
    }
    if env_value in compress_map:
        profile_log(f"[KVCacheManager] Using compress type from env: {env_value}")
        return compress_map[env_value]
    raise ValueError(
        f"Unsupported compression type: {value}. "
        "Open-source CatKV supports only NONE and OURS."
    )


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config_value(config: Optional[Dict[str, Any]], key: str, default):
    if not config:
        return default
    value = config.get(key)
    return default if value is None else value


def _config_flag(
    config: Optional[Dict[str, Any]], key: str, default: bool
) -> bool:
    value = _config_value(config, key, None)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


from lmcache.v1.compute.blend.compress.normal import Normal
from lmcache.v1.compute.blend.compress.our import Ours
from lmcache.v1.compute.blend.db import DataCenter
from lmcache.v1.compute.blend.catkv_ops_adapter import (
    CatKVOpsAdapterConfig,
    CatKVOpsBlendAdapter,
)
from lmcache.v1.compute.blend.shared_key_sv import (
    default_group_uuid_from_key,
    get_shared_key_sv_filename,
    normalize_uuid,
)



class CompressFactory:
    """
    Factory class for creating compression instances.
    """
    @staticmethod
    def create_compressor(compress_type: CompressType, compress_config: dict = None, layer_num : int = 0, device="cuda"):
        compress_config = compress_config or {}
    
        if compress_type == CompressType.NONE:
            return Normal(device=device, config=compress_config)
        elif compress_type == CompressType.OURS:
            return Ours(device=device)
        raise ValueError(
            f"Unsupported compression type: {compress_type}. "
            "Open-source CatKV supports only NONE and OURS."
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
    def get_instance(
        cls,
        shape=None,
        layer_num=None,
        device=None,
        compress_type=None,
        catkv_ops_config: Optional[Dict[str, Any]] = None,
    ):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = KVCacheManager(
                        shape=shape,
                        layer_num=layer_num,
                        device=device,
                        compress_type=compress_type,
                        catkv_ops_config=catkv_ops_config,
                    )
        return cls._instance
    
    @classmethod
    def clean_instance(cls):
        if cls._instance is not None:
            cls._instance.clean()

    def __init__(
        self,
        layer_num: int,
        device="cuda",
        shape=None,
        max_buffer_size=0 * 1024 * 1024 * 1024,
        compress_type=None,  # If None, read from env LMCACHE_COMPRESS_TYPE
        compress_config=None,
        io_config: Optional[Dict[str, Any]] = None,
        catkv_ops_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the KVCacheManager with specified parameters.

        Args:
            layer_num: Number of layers in the model.
            device: Device to use for computation.
            shape: Shape of the KV cache tensors.
            max_buffer_size: Maximum buffer size for CPU buffer pool.
            offload_mode: Offload mode (CPU or DISK).
            io_mode: IO mode (SAFETENSOR, BIN, CPP, S3, etc.).
            compress_type: Compression type.
            compress_config: Compression configuration.
            io_config: IO configuration (e.g., {"config_path": "path/to/s3.ini"} for S3).
        """
        self.device = device
        self.max_buffer_size = max_buffer_size

        self.io_config = io_config or {}
        self.catkv_ops_config = catkv_ops_config or {}
        self.db = DataCenter(
            max_buffer_size=max_buffer_size,
            device=device,
            io_config=self.io_config,
        )
        # If compress_type is None, read from environment variable
        if compress_type is None:
            compress_type = get_compress_type_from_env()
        else:
            compress_type = _normalize_compress_type(compress_type)
        self.compress_type = compress_type
        self.layer_num = layer_num
        self.compress_config = compress_config or {}
        self.compressor = CompressFactory.create_compressor(compress_type, self.compress_config, layer_num=layer_num , device=self.device)
        catkv_ops_enabled = _config_flag(
            self.catkv_ops_config,
            "enabled",
            _env_flag("LMCACHE_USE_CATKV_OPS", False),
        )
        catkv_ops_store_enabled = _config_flag(
            self.catkv_ops_config,
            "store_enabled",
            _env_flag("LMCACHE_CATKV_OPS_STORE", catkv_ops_enabled),
        )
        catkv_ops_prefetch_enabled = _config_flag(
            self.catkv_ops_config,
            "prefetch_enabled",
            _env_flag("LMCACHE_CATKV_OPS_PREFETCH", catkv_ops_enabled),
        )
        self._catkv_ops_adapter = CatKVOpsBlendAdapter(
            CatKVOpsAdapterConfig(
                enabled=catkv_ops_store_enabled or catkv_ops_prefetch_enabled,
                config_path=self.io_config.get("catkv_ops_s3_config")
                or self.io_config.get("catkv_s3_config")
                or _config_value(self.catkv_ops_config, "s3_config", None)
                or os.environ.get("LMCACHE_CATKV_OPS_S3_CONFIG"),
                store_enabled=catkv_ops_store_enabled,
                prefetch_enabled=catkv_ops_prefetch_enabled,
                ratio=_config_value(
                    self.catkv_ops_config,
                    "ratio",
                    self.compress_config.get("ratio", 0.15),
                ),
                num_workers=int(
                    _config_value(
                        self.catkv_ops_config,
                        "num_workers",
                        self.compress_config.get("num_workers", 32),
                    )
                ),
                max_queue_bytes=int(
                    _config_value(
                        self.catkv_ops_config,
                        "max_queue_bytes",
                        self.compress_config.get("max_queue_bytes", 0),
                    )
                ),
                dtype=self.compress_config.get("dtype", torch.bfloat16),
                prefetch_workers=int(
                    _config_value(
                        self.catkv_ops_config,
                        "prefetch_workers",
                        self.compress_config.get("prefetch_workers", 16),
                    )
                ),
                queue_log_path=_config_value(
                    self.catkv_ops_config,
                    "queue_log_path",
                    os.environ.get("LMCACHE_CATKV_OPS_QUEUE_LOG"),
                ),
                queue_log_stdout=_config_flag(
                    self.catkv_ops_config,
                    "queue_log_stdout",
                    _env_flag("LMCACHE_CATKV_OPS_QUEUE_LOG_STDOUT", False),
                ),
                queue_log_interval=float(
                    _config_value(
                        self.catkv_ops_config,
                        "queue_log_interval",
                        os.environ.get("LMCACHE_CATKV_OPS_QUEUE_LOG_INTERVAL", "0.5"),
                    )
                ),
            )
        )

        self.pinned_buffer_allocator = MemoryPinnedBuffer(shape)
        
        self.transfer_stream = torch.cuda.Stream(device=self.device)

        self.keys_set = set()
        self._offload_lock = threading.Lock()
        self._offload_queue: queue.Queue[Any] = queue.Queue()
        self._offload_worker: Optional[threading.Thread] = None
        self._pending_offload_futures: List[Future] = []

    def _start_offload_worker(self) -> None:
        with self._offload_lock:
            if self._offload_worker is not None and self._offload_worker.is_alive():
                return

            self._offload_queue = queue.Queue()
            self._offload_worker = threading.Thread(
                target=self._offload_worker_loop,
                name="blend-prefill-offload-worker",
                daemon=True,
            )
            self._offload_worker.start()

    def _drain_completed_offload_futures(self) -> None:
        still_pending: List[Future] = []
        for future in self._pending_offload_futures:
            if future.done():
                future.result()
                continue
            still_pending.append(future)
        self._pending_offload_futures = still_pending

    def _offload_worker_loop(self) -> None:
        while True:
            job = self._offload_queue.get()
            if job is None:
                self._offload_queue.task_done()
                break

            try:
                key = job["key"]
                data = job["data"]
                layer_idx = job["layer_idx"]
                producer_event = job["producer_event"]
                future = job["future"]

                sequence_len = data[0].size(1)
                key_pinned = self.pinned_buffer_allocator.allocate(sequence_len)
                value_pinned = self.pinned_buffer_allocator.allocate(sequence_len)

                self.transfer_stream.wait_event(producer_event)
                with torch.cuda.stream(self.transfer_stream):
                    key_pinned.copy_(data[0], non_blocking=True)
                    value_pinned.copy_(data[1], non_blocking=True)

                self.transfer_stream.synchronize()

                offloaded_data = {
                    "key": key_pinned,
                    "value": value_pinned,
                }
                flag = self.db.store_data(
                    key,
                    offloaded_data,
                    compress_flag=True,
                    async_disk_save=True,
                )

                if flag and "layer_0" in key:
                    self.keys_set.add(key.split("/")[-1])

                future.set_result(flag)
            except Exception as e:
                logger.exception("Async prefill offload worker failed")
                future.set_exception(e)
            finally:
                self._offload_queue.task_done()

    def flush_pending_offloads(self) -> None:
        for future in list(self._pending_offload_futures):
            future.result()
        self._pending_offload_futures = []

    def shutdown_offload_worker(self, wait: bool = True) -> None:
        worker = self._offload_worker
        if worker is None:
            return

        if wait:
            self.flush_pending_offloads()

        self._offload_queue.put(None)
        if wait:
            worker.join()

        self._offload_worker = None

    def _supports_stream_prefill_offload(self) -> bool:
        # Raw pinned-memory offload bypasses self.compressor.compress() and stores
        # plain key/value tensors directly. That is only correct for the NONE
        # path. OURS must go through store_data() so its encode path runs.
        return self.compress_type in {
            CompressType.NONE,
        }

    def _get_catkv_ops_adapter(self):
        return getattr(self, "_catkv_ops_adapter", None)

    def _supports_catkv_ops_pipeline(self) -> bool:
        adapter = self._get_catkv_ops_adapter()
        return adapter is not None and adapter.supports(self.compress_type)

    def _supports_catkv_ops_store(self) -> bool:
        adapter = self._get_catkv_ops_adapter()
        if adapter is None:
            return False
        if hasattr(adapter, "supports_store"):
            return adapter.supports_store(self.compress_type)
        return adapter.supports(self.compress_type)

    def _supports_catkv_ops_prefetch(self) -> bool:
        adapter = self._get_catkv_ops_adapter()
        if adapter is None:
            return False
        if hasattr(adapter, "supports_prefetch"):
            return adapter.supports_prefetch(self.compress_type)
        return adapter.supports(self.compress_type)

    def _is_empty_payload(self, payload: Any) -> bool:
        if payload is None:
            return True
        if isinstance(payload, (list, tuple, dict)):
            return len(payload) == 0
        return False

    def check_keys_exist(self, keys: list) -> list[bool]:
        """
        Check if the given keys exist in the KV cache.
        """
        # keys: list of str ["xx/xx/filename"]
        # get the filenames
        
        return [key.split("/")[-1] in self.keys_set for key in keys]

    def set_compress_type(self, compress_type: CompressType, compress_config=None):
        """
        Set the compression type for the KV cache.
        """
        self.compress_type = compress_type
        self.compress_config = compress_config or {}
        self.compressor = CompressFactory.create_compressor(compress_type, self.compress_config, layer_num=self.layer_num, device=self.device)


    def retrieve_data(self, key: str , kv, layer_idx: int , stream=None):
        """
        Load data from a file.
        """
        data = self.db.retrieve_data(key)
      
        data = self.compressor.decompress(data , kv , layer_idx )
        
        return  data
    
    def store_data(
        self,
        key: str,
        data,
        score = None,
        layer_idx: int = None,
        group_uuid: Optional[str] = None,
    ):
        """
        Save data to a file or cpu.
        """
        if self.compress_type == CompressType.OURS:
            if group_uuid is None:
                group_uuid = default_group_uuid_from_key(key)
            data = self.compressor.compress(
                data,
                score,
                layer_idx=layer_idx,
                uuid=normalize_uuid(group_uuid),
            )
        else:
            data = self.compressor.compress(data , score , layer_idx=layer_idx)
        # if layer_idx == 0:
        #     ===print(f"[KVManager.store_data] layer={layer_idx}, key={key}, data_keys={list(data.keys())}", flush=True)
        if self.compress_type == CompressType.OURS and isinstance(data, dict):
            key_sv_data = {
                "key_sv_quantized": data["key_sv_quantized"],
                "key_sv_meta": data["key_sv_meta"],
                "key_residual_sv": data["key_residual_sv"],
            }
            other_data = {
                "u_quantized": data["u_quantized"],
                "u_meta": data["u_meta"],
                "value_sv_quantized": data["value_sv_quantized"],
                "value_sv_meta": data["value_sv_meta"],
                "value_residual_sv": data["value_residual_sv"],
            }
            if "uuid" in data:
                key_sv_data["uuid"] = data["uuid"]
                other_data["uuid"] = data["uuid"]

            key_sv_path = f"{key}_key_sv"
            if layer_idx is not None and layer_idx > 2 and "uuid" in key_sv_data:
                key_sv_path = get_shared_key_sv_filename(
                    key_sv_data["uuid"],
                    layer_idx=layer_idx,
                    base_key=key,
                )
            flag = self.db.store_data(key_sv_path, key_sv_data)
            flag = self.db.store_data(f"{key}_other", other_data) and flag
        else:
            flag = self.db.store_data(key, data)
        
        if flag and "layer_0" in key:
            # only need to store the filename
            self.keys_set.add(key.split("/")[-1])

        return  flag

    def offload_compress_data(self, key: str, data: List[torch.Tensor], layer_idx: int = None):
        if self.compress_type == CompressType.OURS:
            if self._supports_catkv_ops_store():
                return self.offload_layer_data(
                    key,
                    data,
                    layer_idx=layer_idx,
                )
            return self.store_data(key, data, layer_idx=layer_idx)

        sequence_len = data[0].size(1)

        # allocate pinned memory
        key_pinned = self.pinned_buffer_allocator.allocate(sequence_len)
        value_pinned = self.pinned_buffer_allocator.allocate(sequence_len)

        # copy to pinned memory
        key_pinned.copy_(data[0] , non_blocking=True)
        value_pinned.copy_(data[1] , non_blocking=True)

        torch.cuda.current_stream(device=self.device).synchronize()
        data = {"key": key_pinned , "value": value_pinned}
        flag = self.db.store_data(
            key,
            data,
            compress_flag=True,
            async_disk_save=True,
        )

        return flag

    def offload_layer_data(
        self,
        key: str,
        data,
        layer_idx: int = None,
        group_uuid: Optional[str] = None,
    ):
        """
        Offload a layer's KV tensors to pinned CPU memory via a dedicated CUDA
        stream worker. For compressor formats that cannot consume raw key/value
        payloads on retrieval, fall back to the existing synchronous store path.
        """
        if self._supports_catkv_ops_store():
            return self._get_catkv_ops_adapter().offload(
                key,
                {"key": data[0], "value": data[1]},
                group_uuid or key,
            )

        if (
            not torch.cuda.is_available()
            or str(data[0].device) == "cpu"
            or not self._supports_stream_prefill_offload()
        ):  
            profile_log(f"Offloading {key} synchronously without stream offload")
            return self.store_data(key, data, layer_idx=layer_idx)

        self._start_offload_worker()
        self._drain_completed_offload_futures()

        future: Future = Future()
        producer_event = torch.cuda.Event()
        torch.cuda.current_stream(device=self.device).record_event(producer_event)
        profile_log(f"Scheduling offload for key={key} on layer {layer_idx}")
        self._pending_offload_futures.append(future)
        self._offload_queue.put(
            {
                "key": key,
                "data": data,
                "layer_idx": layer_idx,
                "producer_event": producer_event,
                "future": future,
            }
        )

        return future
        
    def retrieve_by_task_id(self, task_id):
        if self._supports_catkv_ops_prefetch():
            return self._get_catkv_ops_adapter().get_prefetch_result(task_id)

        t0 = time.perf_counter()
        compress_data_cpu , compress_data_gpu = self.db.retrive_by_task(task_id)
        t1 = time.perf_counter()
        profile_log(f"retrieve_by_task_id: db.retrive_by_task took {(t1-t0)*1000:.2f}ms")

        if isinstance(compress_data_gpu, list):
            result = []
            t_transfer_start = time.perf_counter()
            for i, data in enumerate(compress_data_gpu):
                if self._is_empty_payload(data):
                    result.append(data)
                    continue
                transferred = self.compressor.transfer(data)
                result.append(transferred)
            t_transfer_end = time.perf_counter()
            profile_log(f"retrieve_by_task_id: transfer {len(compress_data_gpu)} items took {(t_transfer_end-t_transfer_start)*1000:.2f}ms")
            return result
        if self._is_empty_payload(compress_data_gpu):
            return compress_data_gpu
        if compress_data_gpu is not None:
            return self.compressor.transfer(compress_data_gpu)
        return compress_data_gpu
  
    def retrieve_keys(self, keys: List[str], **kwargs):
        logger.info(f"Retrieving keys: {keys}")
        if self._supports_catkv_ops_prefetch():
            return self._get_catkv_ops_adapter().prefetch_remote(
                keys, device=self.device, **kwargs
            )

        # if keys and len(keys) > 0:
        #     print(f"[KVManager.retrieve_keys] keys={keys}", flush=True)
        t_start = time.perf_counter()
        task_id = self.db.retrieve_keys(keys)
        t_db_end = time.perf_counter()
        profile_log(f"retrieve_keys: db.retrieve_keys for {len(keys)} keys took {(t_db_end - t_start) * 1000:.2f}ms")
        if isinstance(task_id, list):
            result = []
            t_transfer_start = time.perf_counter()
            for i, data in enumerate(task_id):
                # if i == 0:
                #     print(f"[KVManager.retrieve_keys] loaded data[0] keys={list(data.keys()) if isinstance(data, dict) else type(data)}", flush=True)
                result.append(self.compressor.transfer(data))
            t_transfer_end = time.perf_counter()
            profile_log(f"retrieve_keys: compressor.transfer for {len(task_id)} entries took {(t_transfer_end - t_transfer_start) * 1000:.2f}ms")
            return result
        return task_id

    def decompress(self, compressed_data , kv_len):
        t_decompress_start = time.perf_counter()
        decompressed = self.compressor.decompress(compressed_data , kv_len)
        t_decompress_end = time.perf_counter()
        profile_log(f"decompress: compressor.decompress for kv_len={kv_len} took {(t_decompress_end - t_decompress_start) * 1000:.2f}ms")
        return decompressed

    def clean(self):
        self.shutdown_offload_worker(wait=True)
        adapter = self._get_catkv_ops_adapter()
        if adapter is not None:
            adapter.shutdown()
        self.db.clean()

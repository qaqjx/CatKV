
from enum import Enum
from pickletools import pydict
from typing import Dict, List, Optional, Any

import torch
from lmcache.v1.compute.blend.cpu_buffer import MAX_BUFFER_SIZE, CPUBufferPool
from lmcache.v1.compute.blend.s3_manager import S3DiskManager

class OffloadMode(Enum):
    CPU = "cpu"
    DISK = "disk"

class DataCenter:
    """
    Base class for data center operations.
    """
    def __init__(
        self,
        max_buffer_size=MAX_BUFFER_SIZE,
        offload_mode=OffloadMode.DISK,
        device="cuda",
        io_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.offload_mode = offload_mode
        self.io_config = io_config or {}
        self.disk_io_manager = S3DiskManager()
        self.cpu_buffer_pool = CPUBufferPool(max_size=max_buffer_size)
        self.device = device
        self.pending_save_tasks: Dict[str, int] = {}

    def _wait_pending_save(self, key: str):
        task_id = self.pending_save_tasks.pop(key, None)
        if task_id is None:
            return
        self.disk_io_manager.wait(task_id)

    def wait_pending_saves(self):
        for key in list(self.pending_save_tasks):
            self._wait_pending_save(key)

    def retrieve_data(self, key: str):
        """
        Load data from a file.
        """
        data = self.cpu_buffer_pool.get_data(key)
        if self.offload_mode == OffloadMode.CPU:
            return data
        
        if data is None:
            self._wait_pending_save(key)
            data = self.disk_io_manager.load_data(key)
            if data is not None:
                self.cpu_buffer_pool.add_data(key, data)
                return data
    
        return data
    

    def retrieve_keys(self, keys: List[str]):
        for key in keys:
            self._wait_pending_save(key)

        if self.offload_mode != OffloadMode.DISK or not hasattr(
            self.disk_io_manager, "exists"
        ):
            return self.disk_io_manager.load_datas(keys, self.device)

        existing_indices: List[int] = []
        existing_keys: List[str] = []
        for index, key in enumerate(keys):
            if self.disk_io_manager.exists(key):
                existing_indices.append(index)
                existing_keys.append(key)

        task_id = self.disk_io_manager.load_datas(existing_keys, self.device)
        return {
            "task_id": task_id,
            "existing_indices": existing_indices,
            "requested_len": len(keys),
        }

    def retrive_by_task(self, task_id):
        if isinstance(task_id, dict):
            requested_len = int(task_id.get("requested_len", 0))
            existing_indices = list(task_id.get("existing_indices", []))
            load_task_id = task_id.get("task_id", -1)

            result_cpu = [[] for _ in range(requested_len)]
            result_gpu = [[] for _ in range(requested_len)]
            if load_task_id is None or int(load_task_id) < 0:
                loaded_cpu, loaded_gpu = [], []
            else:
                loaded_cpu, loaded_gpu = self.disk_io_manager.load_task(load_task_id)

            for offset, index in enumerate(existing_indices):
                if offset < len(loaded_cpu) and loaded_cpu[offset] is not None:
                    result_cpu[index] = loaded_cpu[offset]
                if offset < len(loaded_gpu) and loaded_gpu[offset] is not None:
                    result_gpu[index] = loaded_gpu[offset]

            return result_cpu, result_gpu

        result = self.disk_io_manager.load_task(task_id)
        return result

    def store_data(self, key: str, data , compress_flag=False, async_disk_save: bool = False):
        """
        Save data to a file.
        """
        self.cpu_buffer_pool.add_data(key, data)

        if self.offload_mode == OffloadMode.DISK:
            if async_disk_save:
                self.pending_save_tasks[key] = self.disk_io_manager.save_data_async(
                    key,
                    data,
                    compress_flag=compress_flag,
                )
            else:
                self.disk_io_manager.save_data(key, data, compress_flag=compress_flag)

        return True
    
    def clean(self):
        self.wait_pending_saves()
        self.cpu_buffer_pool.clean()

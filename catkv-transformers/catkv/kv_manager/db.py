
from enum import Enum
from typing import Dict, List, Optional, Any

from catkv.kv_manager.cpu_bufferpool import MAX_BUFFER_SIZE, CPUBufferPool
from catkv.kv_manager.disk.safe_tensor import IOMode, create_io_manager
from catkv.kv_manager.utils import _catkv_nvtx_annotate

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
        offload_mode=OffloadMode.CPU,
        io_mode=IOMode.SAFETENSOR,
        device="cuda",
        io_config: Optional[Dict[str, Any]] = None,
    ):
        self.max_buffer_size = max_buffer_size
        self.offload_mode = offload_mode
        self.io_config = io_config or {}
        self.disk_io_manager = create_io_manager(io_mode, **self.io_config)
        self.cpu_buffer_pool = CPUBufferPool(max_size=max_buffer_size)
        self.device = device

    def set_io_mode(self, io_mode: IOMode, **kwargs):
        """
        Set the IO mode for disk operations.
        """
        config = {**self.io_config, **kwargs}
        self.disk_io_manager = create_io_manager(io_mode, **config)

    def retrieve_data(self, key: str):
        """
        Load data from a file.
        """
        data = self.cpu_buffer_pool.get_data(key)
        if self.offload_mode == OffloadMode.CPU:
            return data
        
        if data is None:
            with _catkv_nvtx_annotate("disk_load_data"):
                data = self.disk_io_manager.load_data(key)
                if data is not None:
                    self.cpu_buffer_pool.add_data(key, data)
                    return data
        
        return data
    

    def retrieve_keys(self, keys: List[str]):
        # get by cpu buffer
        result = [self.cpu_buffer_pool.get_data(key) for key in keys]
        
        disk_keys = [keys[i] for i in range(len(keys)) if result[i] is None]
        if disk_keys == []:
            return result

        task_id = self.disk_io_manager.load_datas(disk_keys, self.device) 
        # for i, key in enumerate(disk_keys):
        #     if disk_result_cpu[i] is not None:
        #         self.cpu_buffer_pool.add_data(key, disk_result_cpu[i].clone())
        
        return task_id
        # merge result

        for i, key in enumerate(disk_keys):
            if disk_result[i] is not None:
                self.cpu_buffer_pool.add_data(key, disk_result[i])
        result = [data if data is not None else self.cpu_buffer_pool.get_data(key) for data, key in zip(result, keys)]
        return result

    def retrive_by_task(self, task_id):
        result = self.disk_io_manager.load_task(task_id)
        return result

    def store_data(self, key: str, data , compress_flag=False):
        """
        Save data to a file.
        """
        self.cpu_buffer_pool.add_data(key, data)

        if self.offload_mode == OffloadMode.DISK:
            self.disk_io_manager.save_data(key, data, compress_flag=compress_flag)

        return True
    
    def clean(self):
        self.cpu_buffer_pool.clean()

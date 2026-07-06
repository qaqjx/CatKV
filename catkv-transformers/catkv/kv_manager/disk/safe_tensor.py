
from enum import Enum
import os
from multiprocessing.pool import ThreadPool
from typing import List

from safetensors import safe_open
from safetensors.torch import save_file


class IOMode(Enum):
    SAFETENSOR = "safetensor"
    S3 = "s3"


class DiskIOManager:
    def __init__(self):
        self.executor = ThreadPool(8)

    def load_data(self, file_name: str, device="cpu") -> dict:
        pass

    def save_data(self, file_name: str, data: dict):
        pass

    def load_datas(self, file_names: List[str], device="cpu") -> List[dict]:
        pass


class SafeTensorDiskManager(DiskIOManager):
    def __init__(self):
        super().__init__()

    def load_data(self, file_name: str, device="cpu") -> dict:
        """
        Load key-value pairs from a file.
        """
 
        with safe_open(file_name, framework="pt") as f:
            loaded_dict = {key: f.get_tensor(key).to(device) for key in f.keys()}

        return loaded_dict
    
    def save_data(self, file_name: str, data: dict, compress_flag=False):
        """
        Save key-value pairs to a file.
        """
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        save_file(data, file_name)

    def load_datas(self, file_names, device="cpu"):
        results = []
        for file_name in file_names:
            data = self.load_data(file_name, device)
            results.append(data)
        return results


def _normalize_io_mode(mode: IOMode | str) -> IOMode:
    if isinstance(mode, IOMode):
        return mode
    try:
        return IOMode[str(mode).upper()]
    except KeyError:
        try:
            return IOMode(str(mode).lower())
        except ValueError:
            raise ValueError(
                f"Unsupported IO mode: {mode}. Open-source CatKV supports only SAFETENSOR and S3."
            ) from None


def create_io_manager(mode: IOMode | str, **kwargs) -> DiskIOManager:
    mode = _normalize_io_mode(mode)
    if mode == IOMode.SAFETENSOR:
        return SafeTensorDiskManager()
    if mode == IOMode.S3:
        from catkv.kv_manager.disk.s3_disk import S3DiskManager

        return S3DiskManager(kwargs.get("config_path"))
    raise ValueError(f"Unsupported IO mode: {mode}")

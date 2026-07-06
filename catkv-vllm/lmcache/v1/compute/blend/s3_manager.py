from __future__ import annotations

import os
from typing import Dict, List, Optional, Union

import torch

try:
    from catkv_ops import S3Manager, S3Schedule
except ImportError as e:
    raise ImportError(
        "catkv_ops module not found. Please build the C++ extension with S3 support."
    ) from e


class S3DiskManager():
    """Disk IO manager backed by the S3/MinIO C++ extension."""

    MAX_TASK_THREADS = 64  # Increased from 16 to support concurrent blend requests

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize S3DiskManager.

        Args:
            config_path: Path to the S3 config INI file.
                         Example format:
                         [s3]
                         endpoint = localhost:9000
                         access_key = minioadmin
                         secret_key = minioadmin
                         bucket = kv-cache
                         region = us-east-1
                         use_ssl = false
        """
        super().__init__()
        self._config_path = config_path or os.environ.get(
            "LMCACHE_CATKV_OPS_S3_CONFIG"
        )
        if not self._config_path:
            raise ValueError(
                "S3 config path is required. Set LMCACHE_CATKV_OPS_S3_CONFIG "
                "or pass config_path."
            )
        self._backend = S3Schedule(self._config_path, self.MAX_TASK_THREADS)
        self._manager = S3Manager(self._config_path)
        print("S3DiskManager initialized with config:", self._config_path)

    def load_data(self, key: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Load tensor dictionary from S3.

        Args:
            key: S3 object key (path within bucket).
            device: Target device (currently ignored, data loaded to CPU first).

        Returns:
            Dictionary mapping tensor names to tensors.
        """
        # Use S3Manager.load() directly for synchronous single-file load
        return self._manager.load(key)

    def save_data(
        self, key: str, data: Dict[str, torch.Tensor], compress_flag: bool = False
    ):
        """
        Save tensor dictionary to S3.

        Args:
            key: S3 object key (path within bucket).
            data: Dictionary mapping tensor names to tensors.
            compress_flag: Compression flag (currently ignored).
        """
        # Use S3Manager.save() directly for synchronous save
        self._manager.save(key, data)

    def save_data_async(
        self, key: str, data: Dict[str, torch.Tensor], compress_flag: bool = False
    ) -> int:
        """
        Submit an asynchronous save task to the S3 scheduler.

        Returns:
            Task ID for checking readiness or waiting via `wait()`.
        """
        return self._backend.submit_save(key, data)

    def _get_device_index(self, device: Union[str, torch.device]) -> int:
        """Extract CUDA device index from device specification."""
        if isinstance(device, str):
            if device == "cpu":
                return -1
            elif device.startswith("cuda"):
                if ":" in device:
                    return int(device.split(":")[1])
                return 0
            return -1
        elif isinstance(device, torch.device):
            if device.type == "cpu":
                return -1
            elif device.type == "cuda":
                return device.index if device.index is not None else 0
        return -1

    def load_datas(
        self, keys: List[str], device: Union[str, torch.device] = "cpu"
    ) -> int:
        """
        Submit batch load task to S3.

        Args:
            keys: List of S3 object keys to load.
            device: Target CUDA device for GPU transfer.

        Returns:
            Task ID for retrieving results via load_task().
        """
        if not keys:
            return -1

        device_index = self._get_device_index(device)
        # print(f"Submitting batch load of {(keys)} keys to device index {device_index}")
        task_id = self._backend.submit_batch_load_to_gpu(keys, device_index)
        return task_id

    def load_task(self, task_id):
        """
        Get results of a batch load task.

        Args:
            task_id: Task ID from load_datas(), or -1 for empty.

        Returns:
            Tuple of (cpu_results, gpu_results) where each is a list of tensor dicts.
        """
        if not isinstance(task_id, int) or task_id < 0:
            return [], []
        return self._backend.get_batch_load_to_gpu_result(task_id)

    def is_ready(self, task_id: int) -> bool:
        """Check if a task has completed."""
        return self._backend.is_ready(task_id)

    def wait(self, task_id: int):
        """Block until a task completes."""
        self._backend.wait(task_id)

    def exists(self, key: str) -> bool:
        """Check if an object exists in S3."""
        return self._manager.exists(key)

    def remove(self, key: str):
        """Delete an object from S3."""
        self._manager.remove(key)

    def get_object_size(self, key: str) -> int:
        """Get the size of an object in bytes."""
        return self._manager.get_object_size(key)

    def warmup_connections(self, num_connections: int = 0):
        """
        Pre-establish TCP connections to S3 server.

        Call this before batch operations to eliminate TCP handshake latency.

        Args:
            num_connections: Number of connections to warm up (0 = pool size).
        """
        self._manager.warmup_connections(num_connections)


from collections import OrderedDict
import threading

from torch import tensor


MAX_BUFFER_SIZE = 10 * 1024 * 1024 * 1024  # 10GB

class CPUBufferPool:
    """
    A class to manage a pool of CPU buffers for efficient memory management.
    """
    def __init__(self, max_size: int = MAX_BUFFER_SIZE):
        self.max_size = max_size
        self.pool = OrderedDict()  # Use an OrderedDict to maintain insertion order
        self.current_size = 0
        self.lock = threading.Lock()

    def compute_size(self , data: dict) -> int:
        """
        Compute the total size of the buffers in the pool.
        """
        total_size = 0

        if "chunk_size" in data:
            total_size = data["chunk_size"] * 4 * 1024 * 32
            return total_size

        for _ , item in data.items():
            total_size += item.numel() * item.element_size()
        return total_size
    
    def get_data(self, key: str) -> list[tensor]:
        """
        Retrieve data from the pool by key.
        """
        if key in self.pool:
            # Move the accessed item to the end to maintain order
            self.pool.move_to_end(key)
            return self.pool[key]
        return None
    
    def add_data(self, key: str, data: dict) -> bool:
        """
        Add data to the pool under the specified key.
        If the pool exceeds max_size, remove the oldest entry.
        """
        with self.lock:
            if key in self.pool:
                self.pool.move_to_end(key)
                return True
            
            self.pool[key] = data
            self.current_size += self.compute_size(data)

            while self.current_size > self.max_size:
                old_key, old_data = self.pool.popitem(last=False)
                self.current_size -= self.compute_size(old_data)

        return True

    def clean(self):
        self.current_size = 0
        self.pool = OrderedDict()
    

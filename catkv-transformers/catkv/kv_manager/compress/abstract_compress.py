
from enum import Enum

class CompressType(Enum):
    """
    Public compression methods kept in the open-source release.
    """
    NONE = "None"
    OURS = "ours"


class AbstractCompress:

    def __init__(self, device="cuda"):
        """
        Initialize the compression algorithm with a device.
        """
        self.device = device
        
    """
    Abstract base class for compression algorithms.
    """
    def compress(self, data):
        """
        Compress the given data.
        """
        pass
    
    def decompress(self, compressed_data , kv_len):
        """
        Decompress the given compressed data.
        """
        pass

    def transfer(self, compressed_data):
        pass

    def compress_multi(self, layer_kv: list, indices = None):
        key , value = layer_kv
        data_list = []
        for idx in indices:
           data_list.append(self.compress((key[: , idx[0] : idx[1] ] , value[: , idx[0] : idx[1]])))

        return data_list



from time import time
import torch
from catkv.kv_manager.compress.abstract_compress import AbstractCompress


class Normal(AbstractCompress):
    """
    SVD compression algorithm.
    """
    def __init__(self, device="cuda", config: dict = None):
        super().__init__(device=device)
        config = config or {}

    def compress(self, data, score=None , layer_idx = None) -> dict:
        """
        Not compress the given data .
        """

        key , value = data
        # Get the time 
        data_dict = {
            "key": key.squeeze(0).contiguous().cpu().pin_memory(),
            "value": value.squeeze(0).contiguous().cpu().pin_memory(),
        }

        return data_dict
    
    def transfer(self, compressed_data):
        key = compressed_data["key"].to(self.device, non_blocking=True)
        value = compressed_data["value"].to(self.device, non_blocking=True)
        return key, value

    def decompress(self, compressed_data , kv_len):
        """
        Decompress the given compressed data using SVD.
        """
        return compressed_data["key"] , compressed_data["value"]
    
    def compress_multi(self, layer_kv: list, indices = None):
        key , value = layer_kv
        data_list = []
        for idx in indices:
           data_list.append(self.compress((key[: , idx[0] : idx[1] ] , value[: , idx[0] : idx[1]])))

        return data_list
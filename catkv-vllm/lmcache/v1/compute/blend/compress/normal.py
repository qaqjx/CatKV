

from lmcache.v1.compute.blend.compress.abstract import AbstractCompress

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
        data_dict = {
            "key": key.contiguous().cpu().pin_memory(),
            "value": value.contiguous().cpu().pin_memory(),
        }

        return data_dict
    
    def transfer(self, compressed_data):
        for key in compressed_data:
            compressed_data[key] = compressed_data[key].to(self.device, non_blocking=True)
    
        return compressed_data

    def decompress(self, compressed_data , kv_len):
        """
        Decompress the given compressed data using SVD.
        """
        return compressed_data["key"] , compressed_data["value"]
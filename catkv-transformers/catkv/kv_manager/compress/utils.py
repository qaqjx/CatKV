
from catkv.kv_manager.compress.abstract_compress import CompressType
from catkv.kv_manager.compress.ours import Ours
from catkv.kv_manager.compress.normal import Normal


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
            return Ours(device=device , layer_num = layer_num, config=compress_config)
        else:
            raise ValueError(
                f"Unsupported compression type: {compress_type}. "
                "Open-source CatKV supports only NONE and OURS."
            )

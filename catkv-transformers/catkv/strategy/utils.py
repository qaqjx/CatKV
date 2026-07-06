from catkv.strategy.Cacheblend import CacheBlend
from catkv.strategy.Epic import EPIC
from catkv.strategy.abstract_blend import Blender, ProcessType
from catkv.strategy.Kvshare import KVShare


class BlenderFactory:
  def get_blender(self, layer_idx: int, blend_meta: dict) -> Blender:
    if blend_meta["select_strategy"] == ProcessType.CACHEBLEND:
      return CacheBlend(layer_idx, blend_meta)
    elif blend_meta["select_strategy"] == ProcessType.EPIC:
      return EPIC(layer_idx, blend_meta)
    elif blend_meta["select_strategy"] == ProcessType.KVSHARE:
      return KVShare(layer_idx, blend_meta)
    else:
      raise ValueError(f"Invalid blend type: {blend_meta['select_strategy']}")

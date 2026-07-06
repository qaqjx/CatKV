
from enum import Enum
from torch import tensor


class ProcessType(Enum):
    DEFAULT = "default"
    CACHEBLEND = "cacheblend"
    EPIC = "epic"
    KVSHARE = "kvshare"



class Blender:

  def __init__(self, layer_idx: int, blend_meta: dict):
    self.layer_idx = layer_idx
    self.select_config = blend_meta["select_config"]
    self.blend_meta = blend_meta
    self.device = blend_meta["device"]

  def blend_forward(
    self, 
    query: tensor, 
    key: tensor,
    value: tensor,
    retrieve_kv = None
  ) -> tensor:
      """
      Perform the blend forward operation.
      
      Args:
          query (tensor): The query tensor.
          key (tensor): The key tensor.
          value (tensor): The value tensor.
          blend_meta (dict): Metadata for blending operation.
      
      Returns:
          tensor: The result of the blend forward operation.
      """
      # Implement the blend forward logic here
      pass

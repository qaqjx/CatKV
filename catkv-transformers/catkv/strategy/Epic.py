import torch
from catkv.strategy.abstract_blend import Blender


class EPIC(Blender):
  """
  EPIC is a class that extends the Blender class to implement the EPIC blending method.
  It inherits from the Blender class and can be used to perform blending operations with EPIC capabilities.
  """

  def __init__(self, layer_idx: int, blend_meta: dict):
    super().__init__(layer_idx, blend_meta)

  def blend_forward(self, query, key, value, retrieve_kv):
    """
    Perform the EPIC blend forward operation.
    
    Args:
        query (tensor): The query tensor.
        key (tensor): The key tensor.
        value (tensor): The value tensor.

    Returns:
        tensor: The result of the EPIC blend forward operation.
    """
    # Implement the EPIC blend forward logic here
    not_reused_mask = torch.ones(
        query.size(1), device=query.device, dtype=torch.bool
    )

    for indice in self.blend_meta["indices"]:
      start = indice[-2]
      end = indice[-1]
      not_reused_mask[start + self.select_config["recompute_num"]:end] = False

    return torch.sort(torch.where(not_reused_mask)[0]).values 
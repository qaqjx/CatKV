import numpy as np
import torch

class CacheBlend():
  """
  Cacheblend is a class that extends the blend class to implement a caching mechanism for blending operations.
  It inherits from the blend class and can be used to perform blending operations with caching capabilities.
  """

  def __init__(self, blend_meta: dict, device: str = "cuda"):
    self.blend_meta = blend_meta
    self.device = device

  def set_rope(self, rope):
    self.rope = rope

  def calc_deviation(self, x, y):
    deviation = torch.mean((x - y) ** 2, dim=[0, -2, -1])
   
    return deviation

  def blend_forward(self, query, key, value, retrieve_kv , active_indices):
    """
    Perform the blend forward operation with caching.
    
    Args:
        query (tensor): The query tensor.
        key (tensor): The key tensor.
        value (tensor): The value tensor.

    Returns:
        tensor: The result of the blend forward operation.
    """
    not_reused_mask = torch.ones(
        query.size(1), device=query.device, dtype=torch.bool
    )
    
    for indice in active_indices:
      start = indice[-2]
      end = indice[-1]
      not_reused_mask[start:end] = False

    retrieve_key, retrieve_value = retrieve_kv

    key_or_value = "value"
    
    target, ref = (value, retrieve_value) if key_or_value == "value" else (key, retrieve_key)
    deviation = self.calc_deviation(target, ref)
      # Set the value deviation to 0 for tokens that are not reused
    deviation[torch.where(not_reused_mask)[0]] = 0.0

    recomputed_token_num = int(
      np.ceil(
          (key.size(1) - (not_reused_mask).sum().item()) *  0.15
      )
    )
    recompute_idx = torch.topk(deviation, recomputed_token_num, dim=0)[1].view(-1)

    return torch.sort(torch.cat((recompute_idx, torch.where(not_reused_mask)[0]))).values

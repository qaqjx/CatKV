import numpy as np
import torch
from catkv.strategy.abstract_blend import Blender

class CacheBlend(Blender):
  """
  Cacheblend is a class that extends the blend class to implement a caching mechanism for blending operations.
  It inherits from the blend class and can be used to perform blending operations with caching capabilities.
  """

  def __init__(self, layer_idx: int, blend_meta: dict):
    super().__init__(layer_idx, blend_meta)
  
  def set_rope(self, rope):
    self.rope = rope

  def calc_deviation(self, x, y):
    if "deviation_type" not in self.select_config or self.select_config["deviation_type"] == "LMCACHE":
      deviation = torch.mean((x - y) ** 2, dim=[0, -2, -1])
    elif self.select_config["deviation_type"] == "L2_Dist":
      deviation = torch.mean(torch.linalg.norm(x - y, ord=2, dim=-1), dim=[0, 2])
    elif self.select_config["deviation_type"] == "L1_Dist":
      deviation = torch.mean(torch.abs(x - y), dim=[0, -2, -1])
    elif self.select_config["deviation_type"] == "Cosine_Dist":
      deviation = torch.mean(1.0 - torch.nn.functional.cosine_similarity(x, y, dim=-1), dim=[0, 2])
    elif self.select_config["deviation_type"] == "Dist":
      l1_deviation = torch.norm(x - y, p = 1, dim=[0, -2, -1])
      cosine_deviation = torch.mean(1.0 - torch.nn.functional.cosine_similarity(x, y, dim=-1), dim=[0, 2])
      deviation = l1_deviation * cosine_deviation
    elif self.select_config["deviation_type"] == "Ratio":
      abs_deviation = torch.abs(x - y)
      deviation = torch.mean(torch.abs(abs_deviation / x) , dim = [0 , -2 ,-1])
    else:
      raise ValueError(f"Invalid deviation type: {self.select_config['deviation_type']}")
    return deviation

  def blend_forward(self, query, key, value, retrieve_kv):
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

    offset = 0 if "ablation" not in self.select_config or self.select_config["ablation"] == "False" else 32

    for indice in self.blend_meta["indices"]:
      start = indice[-2]
      end = indice[-1]
      not_reused_mask[start + offset:end] = False

    retrieve_key, retrieve_value = retrieve_kv

    key_or_value = self.select_config.get("key_or_value", "value")
    
    positions = torch.arange(0, query.size(1), device=self.device)
    retrieve_key, key = self.rope(retrieve_key, key, positions.unsqueeze(0).expand(query.size(0), -1))
    
    target, ref = (value, retrieve_value) if key_or_value == "value" else (key, retrieve_key)
    deviation = self.calc_deviation(target, ref)
      # Set the value deviation to 0 for tokens that are not reused
    deviation[torch.where(not_reused_mask)[0]] = 0.0

    recomputed_token_num = int(
      np.ceil(
          (key.size(1) - (not_reused_mask).sum().item()) * self.select_config.get("recompute_ratio", 0.15)
      )
    )
    # recomputed_token_num  = 32 * self.blend_meta["indices"].__len__()
    recompute_idx = torch.topk(deviation, recomputed_token_num, dim=0)[1].view(-1)

    return torch.sort(torch.cat((recompute_idx, torch.where(not_reused_mask)[0]))).values

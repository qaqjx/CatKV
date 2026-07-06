import numpy as np
import torch
from catkv.strategy.abstract_blend import Blender
MAX_ATTENTION_BLOCK_SIZE = 4096  # Define a constant for maximum attention block size


class KVShare(Blender):
  """
  KVShare is a class that extends the blend class to implement a caching mechanism for blending operations.
  It inherits from the blend class and can be used to perform blending operations with caching capabilities.
  """

  def __init__(self, layer_idx: int, blend_meta: dict):
    super().__init__(layer_idx, blend_meta)

  def set_rope(self, rope):
    self.rope = rope

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
    seq_len = key.size(1)
    not_reused_mask = torch.ones(
        seq_len, device=query.device, dtype=torch.bool
    )

    if "indices" in self.blend_meta:
        for start, end in self.blend_meta["indices"]:
          not_reused_mask[start:end] = False

    recomputed_token_num = int(
        np.ceil(
            (seq_len - (not_reused_mask).sum().item()) * self.select_config["recompute_ratio"]
        )
    )
    head_group_num = query.size(-2) // key.size(-2)
            
    diff_value = torch.sum(torch.abs(value - retrieve_kv[1]), dim=[0, -2, -1])
    
    positions = torch.arange(0, seq_len, device=self.device)
    query , key = self.rope(query, key, positions.unsqueeze(0).expand(query.size(0), -1))

    cumulative_attention = torch.zeros(seq_len, device=query.device)
    for q_start in range(0 , seq_len , MAX_ATTENTION_BLOCK_SIZE):
        q_end = min(q_start + MAX_ATTENTION_BLOCK_SIZE, seq_len)
        attn_weight = torch.einsum(
            "bqhd,bkhd->bqk",
            query[:, q_start:q_end, :, :],
            torch.repeat_interleave(key , head_group_num, dim = 2)
        ).squeeze(0)

        # Apply causal mask to the attention weights
        causal_mask = torch.arange(q_start, q_end, device=query.device).unsqueeze(1) >= torch.arange(seq_len, device=query.device)
        attn_weight = attn_weight * causal_mask * (key.size(-1) ** 0.5)
        attn_weight = torch.softmax(attn_weight, dim=-1)

        cumulative_attention += torch.sum(attn_weight, dim=0)

    cumulative_attention.masked_fill_(not_reused_mask, -torch.inf)
    deviation = cumulative_attention * diff_value

    recompute_idx = torch.topk(deviation, recomputed_token_num, dim=0)[1].view(-1)

    return torch.sort(torch.cat((recompute_idx, torch.where(not_reused_mask)[0]))).values

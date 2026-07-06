import flashinfer
import torch


class AdaptiveKVCacheAttention:
    def __init__(self, phase):
        self.phase = phase

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        # q, k, v: (batch_size, num_heads, len_q, dim_head)
        q = query.contiguous().squeeze(0)
        k = key.contiguous().squeeze(0)
        v = value.contiguous().squeeze(0)

        if mask is None:
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        else:
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)

        return o

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        # q, k, v: (batch_size, num_heads, len_q, dim_head)
        q = query.contiguous().squeeze(0)
        k = key.contiguous().squeeze(0)
        v = value.contiguous().squeeze(0)
       
        q = q.squeeze(0)
        o = flashinfer.single_decode_with_kv_cache(
            q, k, v, use_tensor_cores=True
        )

        return o.unsqueeze(0)


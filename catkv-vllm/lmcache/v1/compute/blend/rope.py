import torch
import math
import flashinfer

class RotaryEmbeddingESM(torch.nn.Module):
    """
    Precomputes rotary position embeddings up to max_seq_len during initialization.
    Supports custom positions via indexing into the precomputed cache.
    """

    def __init__(self, rotary_embedding, head_dim, device='cuda'):
        
        super().__init__()
        self.rotary_embedding = rotary_embedding
        self.rotary_dim = rotary_embedding.rotary_dim
        self.device = device
        self.head_dim = rotary_embedding.head_size
        self.is_neox_style = True
        self.base = rotary_embedding.base
        self.max_position_embeddings = rotary_embedding.max_position_embeddings
        self.max_length = self.max_position_embeddings

        if hasattr(rotary_embedding, 'rope_scaling'):
            self.max_length = int(rotary_embedding.rope_scaling["factor"] * self.max_length)
            self.yarn_factor = getattr(rotary_embedding.rope_scaling, 'factor', 1.0)

        self.position_ids = torch.arange(0, self.max_length, device=self.device)
        self.inv_freq = rotary_embedding._compute_inv_freq(self.base)
        # self.attention_scaling = rotary_embedding.attention_scaling

        # self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
        self.cos_sin_cache = rotary_embedding._compute_cos_sin_cache().to(self.device)

    # def _set_cos_sin_cache(self):
    #     from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
    #     from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
    #     from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    #     from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding

    #     if isinstance(self.rotary_embedding, LlamaRotaryEmbedding):
    #         t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    #         freqs = torch.outer(t, self.inv_freq)
    #         return (freqs.cos()*self.attention_scaling).to(self.device), (freqs.sin()*self.attention_scaling).to(self.device)
    #     elif isinstance(self.rotary_embedding, MistralRotaryEmbedding):
    #         t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    #         freqs = torch.outer(t, self.inv_freq)
    #         return (freqs.cos()*self.attention_scaling).to(self.device), (freqs.sin()*self.attention_scaling).to(self.device)
    #     elif isinstance(self.rotary_embedding, Qwen2RotaryEmbedding) or isinstance(self.rotary_embedding, Qwen3RotaryEmbedding):
    #         if self.max_length > 32768:
    #             def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    #                 """Inverse dimension formula to find the dimension based on the number of rotations"""
    #                 return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    #             def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    #                 """Find dimension range bounds based on rotations"""
    #                 low = math.floor(find_correction_dim(low_rot, dim, base, max_position_embeddings))
    #                 high = math.ceil(find_correction_dim(high_rot, dim, base, max_position_embeddings))
    #                 return max(low, 0), min(high, dim - 1)

    #             def linear_ramp_factor(min, max, dim):
    #                 if min == max:
    #                     max += 0.001  # Prevent singularity

    #                 linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    #                 ramp_func = torch.clamp(linear_func, 0, 1)
    #                 return ramp_func
                
    #             attention_factor = 0.1 * math.log(self.yarn_factor) + 1.0
    #             beta_fast = 32
    #             beta_slow = 1

    #             pos_freqs = self.base ** (torch.arange(0, self.head_dim, 2).float().to(self.inv_freq.device) / self.head_dim)
    #             inv_freq_extrapolation = 1.0 / pos_freqs
    #             inv_freq_interpolation = 1.0 / (self.yarn_factor * pos_freqs)

    #             low, high = find_correction_range(beta_fast, beta_slow, self.head_dim, self.base, self.max_position_embeddings)

    #             inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, self.head_dim // 2).float().to(self.inv_freq.device)
    #             inv_freq = (
    #                 inv_freq_interpolation * (1 - inv_freq_extrapolation_factor)
    #                 + inv_freq_extrapolation * inv_freq_extrapolation_factor
    #             )

    #             self.inv_freq = inv_freq
    #             self.attention_scaling = attention_factor

    #         t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    #         freqs = torch.outer(t, self.inv_freq)
    #         return (freqs.cos()*self.attention_scaling).to(self.device), (freqs.sin()*self.attention_scaling).to(self.device)

    def forward(self, query, key, position_ids):
        bsz, seq_len, qo_heads, head_dim = query.shape
        bsz, seq_len, kv_heads, head_dim = key.shape
        query = query.view(-1, head_dim * qo_heads)
        key = key.view(-1, head_dim * kv_heads)
        # NOTE(liyi): DONOT use inplace here to avoid implicit storing post_rope_key_cache
        # flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(position_ids, query, key, self.head_dim, self.cos_sin_cache, self.is_neox_style)
        post_rope_query, post_rope_key = flashinfer.rope.apply_rope_with_cos_sin_cache(position_ids, query, key, self.head_dim, self.cos_sin_cache, self.is_neox_style)

        post_rope_query = post_rope_query.view(bsz, seq_len, qo_heads, head_dim)
        post_rope_key = post_rope_key.view(bsz, seq_len, kv_heads, head_dim)
        return post_rope_query, post_rope_key

    def forward_old(self, x, position):
        cos, sin = self.rotary_embedding(x, position)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (x * cos) + (self.rotate_half(x) * sin)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)

        return torch.cat((-x2, x1), dim=-1)

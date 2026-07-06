import hashlib
import os
from nvtx import annotate  # type: ignore
import torch
import xxhash

store_kvcache_dir = "kvcache/"

def get_kvcache_filename(hash_str : str, layer_idx = -1, device = "cuda:0"):

    filename = os.path.join(store_kvcache_dir, str(hash_str) + "-layer_" + str(layer_idx) + "-device_" + str(device) + ".bin")
    return filename


def _to_signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def uuid_to_tensor(uuid: str) -> torch.Tensor:
    h = xxhash.xxh64(str(uuid)).intdigest()
    high = _to_signed_int32((h >> 32) & 0xFFFFFFFF)
    low = _to_signed_int32(h & 0xFFFFFFFF)
    return torch.tensor([high, low], dtype=torch.int32)


def uuid_tensor_to_path_id(uuid) -> str:
    if isinstance(uuid, str):
        return uuid
    if not isinstance(uuid, torch.Tensor):
        raise TypeError(f"uuid must be a string or tensor, got {type(uuid)!r}")
    values = uuid.detach().cpu().to(torch.int32).flatten().tolist()
    return "_".join(str(int(value)) for value in values)


def get_shared_key_sv_filename(uuid, layer_idx: int, base_key: str = "") -> str:
    directory = os.path.dirname(base_key)
    filename = f"{uuid_tensor_to_path_id(uuid)}_layer_{layer_idx}_key_sv"
    return os.path.join(directory, filename) if directory else filename

_NVTX_COLORS = ["green", "blue", "purple", "rapids"]

def _get_color_for_nvtx(name):
    m = hashlib.sha256()
    m.update(name.encode())
    hash_value = int(m.hexdigest(), 16)
    idx = hash_value % len(_NVTX_COLORS)
    return _NVTX_COLORS[idx]

def _catkv_nvtx_annotate(func_or_name, domain="catkv"):
    """Decorator for applying nvtx annotations to methods in catkv.
    
    Can be used as:
    1. Decorator: @_catkv_nvtx_annotate
    2. Context manager: with _catkv_nvtx_annotate("name"):
    """
    if isinstance(func_or_name, str):
        # Used as context manager with string name
        return annotate(
            message=func_or_name,
            color=_get_color_for_nvtx(func_or_name),
            domain=domain,
        )
    else:
        # Used as decorator with function
        return annotate(
            message=func_or_name.__qualname__,
            color=_get_color_for_nvtx(func_or_name.__qualname__),
            domain=domain,
        )(func_or_name)


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def min_p(data: torch.Tensor, ratio: float = 0.1) -> torch.Tensor:
    # data shape: [num_heads, seq_len, seq_len]
    reduced_data = data.mean(dim=0)  # [seq_len, seq_len]
    threshold = torch.max(reduced_data, dim=-1, keepdim=True).values * ratio
    reduced_data = torch.where(reduced_data < threshold, torch.tensor(0.0, device=data.device), reduced_data)
    prefix_sum = torch.cumsum(reduced_data, dim=-1)
    return prefix_sum

def top_p(data: torch.Tensor, indices: list, ratio: float = 0.8):
    # data shape: [seq_len, seq_len]
    L, _ = data.shape
    device = data.device
    results = []

    self_attn = data.diagonal(dim1=0, dim2=1)

    for start, end in indices:
        if start == 0: continue
        if start >= end: continue
        intra_chunk_score = data[start:end, start]
        attn_score = self_attn[start:end]
        ratio_chunk =  (attn_score - intra_chunk_score) / (attn_score + 1e-8)
        below_threshold = ratio_chunk < ratio

        local_idx, = torch.where(below_threshold)
        if local_idx.numel() > 0:
            token_idx = start + local_idx
            results.append(token_idx)

    if not results:
        return torch.empty((0,), dtype=torch.long, device=device)
    
    return torch.cat(results, dim=0)

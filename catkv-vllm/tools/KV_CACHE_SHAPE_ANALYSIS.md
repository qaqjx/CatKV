# vLLM v1 KV Cache Shape Analysis

## Problem

When running `examples/blend_kv_v1/blend.py`, the adapter prints a KV cache
shape similar to:

```text
torch.Size([2, 12038, 16, 8, 128])
```

The question is where dimension `16` comes from.

## Short Answer

Dimension `16` is vLLM `block_size`, the number of tokens stored in each paged
cache block. It is not `num_kv_heads`.

## Shape Meaning

`torch.Size([2, 12038, 16, 8, 128])` means:

| Dimension | Value | Meaning |
|-----------|-------|---------|
| dim 0 | 2 | K and V |
| dim 1 | 12038 | `num_blocks` allocated on GPU |
| dim 2 | 16 | `block_size` |
| dim 3 | 8 | `num_kv_heads` for Mistral-7B GQA |
| dim 4 | 128 | attention head size |

## vLLM Layout

vLLM v1 uses the NHD layout:

```text
[2, num_blocks, block_size, num_kv_heads, head_size]
```

This is confirmed by logs such as:

```text
Connectors do not specify a kv cache layout, defaulting to NHD.
```

The shape definition is in `vllm/v1/attention/backends/flash_attn.py`:

```python
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[int, int, int, int, int]:
    return (2, num_blocks, block_size, num_kv_heads, head_size)
```

## Mistral-7B Configuration

For `Mistral-7B-Instruct-v0.2`:

```python
num_attention_heads: 32
num_key_value_heads: 8
hidden_size: 4096
head_dim: 128
num_hidden_layers: 32
```

The `block_size` value is a vLLM cache setting and is commonly 16 by default.

## Initialization Flow

vLLM allocates the raw paged buffer, reshapes it to
`[2, num_blocks, block_size, num_kv_heads, head_size]`, and binds it to
attention layers. LMCache then reads those already allocated tensors through the
vLLM adapter.

## Common Confusion

If you expect an HND layout, you may expect:

```text
[2, 12038, 8, 16, 128]
```

In the actual NHD layout, the middle dimensions are:

```text
[2, 12038, 16, 8, 128]
```

So `16` is `block_size`, and `8` is `num_kv_heads`.

## Capacity Check

Total token capacity:

```text
12038 blocks * 16 tokens/block = 192608 tokens
```

This matches the corresponding vLLM KV cache capacity log.

## References

| Topic | File |
|-------|------|
| Shape definition | `vllm/v1/attention/backends/flash_attn.py` |
| Cache allocation | `vllm/v1/worker/gpu/attn_utils.py` |
| Cache reshape | `vllm/v1/worker/gpu/attn_utils.py` |
| Layout decision | `vllm/v1/attention/backends/utils.py` |
| LMCache KV access | `lmcache/integration/vllm/vllm_v1_adapter.py` |

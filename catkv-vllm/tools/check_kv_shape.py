#!/usr/bin/env python3
"""Inspect the vLLM KV cache shape directly."""
import os
import sys

# Set CUDA device.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Minimal configuration.
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_ENABLE_BLENDING"] = "False"  # Disable blending for a simpler test.
os.environ["LMCACHE_USE_LAYERWISE"] = "False"  # Disable layerwise mode for a simpler test.
os.environ["LMCACHE_LOCAL_CPU"] = "True"

from transformers import AutoConfig


print("=" * 80)
print("1. Mistral-7B-Instruct-v0.2 model configuration")
print("=" * 80)

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
config = AutoConfig.from_pretrained(model_name)

print(f"num_attention_heads:     {config.num_attention_heads}")
print(f"num_key_value_heads:     {config.num_key_value_heads}")
print(f"hidden_size:             {config.hidden_size}")
print(f"head_dim:                {config.hidden_size // config.num_attention_heads}")
print(f"num_hidden_layers:       {config.num_hidden_layers}")

print("\n" + "=" * 80)
print("2. LMCache configuration inferred from the vLLM v1 adapter")
print("=" * 80)

# Simulate the vLLM-side calculation.
num_kv_head = config.num_key_value_heads  # Expected: 8.
head_size = config.hidden_size // config.num_attention_heads  # Expected: 128.
chunk_size = 256
num_layer = config.num_hidden_layers  # Expected: 32.

print(f"LMCache kv_shape: ({num_layer}, 2, {chunk_size}, {num_kv_head}, {head_size})")
print(f"  - num_layer:   {num_layer}")
print("  - K and V:     2")
print(f"  - chunk_size:  {chunk_size}")
print(f"  - num_kv_head: {num_kv_head}")
print(f"  - head_size:   {head_size}")

print("\n" + "=" * 80)
print("3. vLLM v1 paged KV cache format hypothesis")
print("=" * 80)

print("""
vLLM v1 paged KV cache may use one of these layouts:

Format A (NHD - Num heads, Head dim):
  [2, num_blocks, num_kv_heads, block_size, head_size]

Format B (HND - Head dim, Num heads):
  [2, num_blocks, head_size, num_kv_heads, block_size]

If the observed shape is [2, 12038, 16, 8, 128]:
  - 2:      K and V
  - 12038:  total GPU paged-cache blocks
  - 16:     possibly num_kv_heads, or twice the expected block_size
  - 8:      possibly block_size or num_kv_heads
  - 128:    head_size

Possible explanations:
1. block_size = 16 and num_kv_heads = 8 with an HND-like layout.
2. num_kv_heads = 16 because of a parallelism or config difference.
3. The tensor has been reshaped or merged by another path.
""")

print("\n" + "=" * 80)
print("4. Conclusion")
print("=" * 80)

print("""
Dimension 16 is most likely from:
1. vLLM block_size, usually 16 by default.
2. A KV cache layout transform.

To confirm, check the full blend.py output for this log line:
  INFO ... [kv_cache_utils.py] Connectors do not specify a kv cache layout, defaulting to NHD.

That line indicates vLLM uses the NHD layout.

With NHD, the shape should be:
  [2, num_blocks, num_kv_heads, block_size, head_size]
  [2, 12038,      8,             16,         128]

If the observed shape is [2, 12038, 16, 8, 128], then either:
- a dimension order is wrong somewhere, or
- block_size and num_kv_heads have been swapped.
""")

#!/usr/bin/env python3
"""Debug where the KV cache shape comes from."""
# SPDX-License-Identifier: Apache-2.0
import os
from dataclasses import asdict

# Set environment variables.
os.environ["LMCACHE_CHUNK_SIZE"] = "256"
os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
os.environ["LMCACHE_BLEND_SPECIAL_STR"] = "# #"
os.environ["LMCACHE_USE_LAYERWISE"] = "True"
os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
os.environ["LMCACHE_LOCAL_CPU"] = "True"
os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"

# Use cuda:1.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs


model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("=" * 80)
print("1. Inspect model configuration")
print("=" * 80)
config = AutoConfig.from_pretrained(model_name)
print(f"Model: {model_name}")
print(f"  num_attention_heads: {config.num_attention_heads}")
print(f"  num_key_value_heads: {config.num_key_value_heads}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  head_dim: {config.hidden_size // config.num_attention_heads}")

print("\n" + "=" * 80)
print("2. Initialize vLLM and inspect KV cache shape")
print("=" * 80)

lmcache_connector = "LMCacheConnectorV1"
ktc = KVTransferConfig(
    kv_connector=lmcache_connector,
    kv_role="kv_both",
)

llm_args = EngineArgs(
    model=model_name,
    kv_transfer_config=ktc,
    max_model_len=8192,
    gpu_memory_utilization=0.5,
    enable_prefix_caching=False,
    enforce_eager=True,
)

print("Initializing LLM...")
llm = LLM(**asdict(llm_args))

# Get cache config details.
print("\nvLLM cache configuration:")
print(f"  block_size: {llm.llm_engine.cache_config.block_size}")
print(f"  num_gpu_blocks: {llm.llm_engine.cache_config.num_gpu_blocks}")

print("\n" + "=" * 80)
print("3. Run a small generation and inspect KV cache")
print("=" * 80)

tokenizer = AutoTokenizer.from_pretrained(model_name)
prompt = tokenizer.encode("Hello world" * 10)[1:]
sampling_params = SamplingParams(temperature=0, max_tokens=1)

print(f"Prompt length: {len(prompt)} tokens")
print("Generating...")

outputs = llm.generate(
    prompts={"prompt_token_ids": prompt},
    sampling_params=sampling_params
)

print("Generation completed")
print("\nKV cache shape during inference is printed by the adapter.")

print("\n" + "=" * 80)
print("Explanation:")
print("=" * 80)
print("""
The vLLM v1 paged KV cache format is:
  [2, num_blocks, num_kv_heads, block_size, head_size]

Where:
  - Dimension 1 (2): K and V
  - Dimension 2 (num_blocks): total GPU KV cache blocks
  - Dimension 3 (num_kv_heads): number of KV heads
  - Dimension 4 (block_size): tokens per block, usually 16
  - Dimension 5 (head_size): attention head size, usually 128 for this model

If the observed shape is [2, 12038, 16, 8, 128], it means:
  - 2: K and V
  - 12038: total allocated blocks
  - 16: likely block_size or a changed num_kv_heads
  - 8: likely num_kv_heads or block_size
  - 128: head_size

Note: LMCache kv_shape and vLLM paged KV cache shape are different:
  - LMCache kv_shape: (num_layer, 2, chunk_size, num_kv_head, head_size)
  - vLLM paged KV: (2, num_blocks, num_kv_head, block_size, head_size)
""")

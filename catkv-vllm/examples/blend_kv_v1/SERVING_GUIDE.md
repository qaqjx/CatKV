# LMCache CacheBlend Serving Guide

## Environment

This guide assumes:

- vLLM is installed and available in the active Python environment.
- LMCache is installed from this repository.
- The vLLM integration is configured to use `LMCacheConnectorV1`.

## Quick Start

Start serving:

```bash
cd catkv-vllm/examples/blend_kv_v1
./start_serving.sh
```

Optional environment overrides:

```bash
MODEL="meta-llama/Llama-3.1-8B-Instruct" ./start_serving.sh
PORT=8001 ./start_serving.sh
GPU_MEM=0.9 ./start_serving.sh
MAX_LEN=8192 ./start_serving.sh
```

Run the test client in another terminal:

```bash
cd catkv-vllm/examples/blend_kv_v1
python3 test_serving.py
```

Expected behavior:

- Request 1 is slower because it is the cold run.
- Requests 2 and 3 should have lower TTFT if CacheBlend reuses cached chunks.
- A speedup greater than 1.5x is a useful quick signal.

## cURL Test

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.2",
    "prompt": "Hello # # World",
    "max_tokens": 10,
    "temperature": 0
  }'
```

## Core Environment Variables

The default settings in `start_serving.sh` are:

```bash
LMCACHE_CHUNK_SIZE=256                    # Chunk size.
LMCACHE_ENABLE_BLENDING=True              # Enable blending.
LMCACHE_BLEND_SPECIAL_STR=" # # "        # Chunk separator.
LMCACHE_USE_LAYERWISE=True                # Process layer by layer.
LMCACHE_BLEND_CHECK_LAYERS=1              # Number of layers to check.
LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15       # Recompute ratio.
LMCACHE_LOCAL_CPU=True                    # Use CPU backend.
LMCACHE_MAX_LOCAL_CPU_SIZE=5              # CPU cache cap in GB.
```

To switch to the disk backend, edit `start_serving.sh`, comment out the CPU
backend settings, and enable:

```bash
export LMCACHE_LOCAL_CPU=False
export LMCACHE_LOCAL_DISK=file://./local_disk/
export LMCACHE_MAX_LOCAL_DISK_SIZE=10
```

## Integration Notes

The separator used in prompts must exactly match `LMCACHE_BLEND_SPECIAL_STR`.
Reusable contexts should be arranged as chunks joined by the separator.
CacheBlend can reuse chunks even when the request uses a different chunk order.

OpenAI SDK example:

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
sep = " # # "
prompt = f"{sys_prompt}{sep}{chunk1}{sep}{chunk2}{sep}{question}"

response = client.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    prompt=prompt,
    max_tokens=100,
    temperature=0,
)
print(response.choices[0].text)
```

## Troubleshooting

For startup failures, check GPU availability with `nvidia-smi`, verify the port
with `lsof -i :8000`, and inspect vLLM logs for LMCache initialization.

If CacheBlend does not appear to work, verify the separator configuration,
check for a "CacheBlend enabled" log line, and confirm that chunks are detected
as expected.

For OOM errors, lower `gpu_memory_utilization`, reduce `max_model_len`, or
decrease `LMCACHE_MAX_LOCAL_CPU_SIZE`.

## Files

- `start_serving.sh`: serving startup script.
- `test_serving.py`: OpenAI API test client.
- `blend.py`: offline inference example.
- `README.md`: notes about the vLLM integration.

#!/bin/bash
CUDA_VISIBLE_DEVICES=1
# CacheBlend serving startup script.

# Set LMCache environment variables.
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_ENABLE_BLENDING=True
export LMCACHE_BLEND_SPECIAL_STR=" # # "
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_BLEND_CHECK_LAYERS=1
export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15  # Improve out-of-order blending.
export LMCACHE_BLEND_MIN_TOKENS=64  # Allow smaller batches to blend.

# CPU backend configuration.
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5

# To use the disk backend, comment the two CPU lines above and uncomment below.
# export LMCACHE_LOCAL_CPU=False
# export LMCACHE_LOCAL_DISK=file://./local_disk/
# export LMCACHE_MAX_LOCAL_DISK_SIZE=10

# Model configuration.
MODEL=${MODEL:-"mistralai/Mistral-7B-Instruct-v0.3"}
PORT=${PORT:-12345}
GPU_MEM=${GPU_MEM:-0.6}
MAX_LEN=${MAX_LEN:-32000}

echo "=========================================="
echo "Starting LMCache CacheBlend Serving"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEM"
echo "Max Model Length: $MAX_LEN"
echo "Blending: Enabled"
echo "Backend: CPU (max ${LMCACHE_MAX_LOCAL_CPU_SIZE}GB)"
echo "Special String: '$LMCACHE_BLEND_SPECIAL_STR'"
echo "=========================================="

# Start vLLM serving.
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len "$MAX_LEN" \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --enforce-eager \
  -tp 1

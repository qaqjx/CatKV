# Example script to run the container vLLM OpenAI server with LMCache
#
# Prerequisite:
# - If CUDA then require NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Set the following variables:
IMAGE=<IMAGE_NAME>:<TAG>
HF_MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
RUNTIME=nvidia
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
CONTAINER_HF_CACHE=${CONTAINER_HF_CACHE:-/cache/huggingface}

docker run --runtime $RUNTIME --gpus all \
    --env "HF_TOKEN=<REPLACE_WITH_YOUR_HF_TOKEN>" \
    --env "HF_HOME=$CONTAINER_HF_CACHE" \
    --env "LMCACHE_CHUNK_SIZE=256" \
    --env "LMCACHE_LOCAL_CPU=True" \
    --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
    --volume "$HF_CACHE_DIR:$CONTAINER_HF_CACHE" \
    --network host \
    $IMAGE \
    $HF_MODEL_NAME --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

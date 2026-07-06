# catkv-release

`catkv-release` groups the CatKV release components. It is not a single Python
package; each source directory has its own package layout and entry points.

## Components

- `csrc/`: C++ and CUDA extensions for CatKV, imported in Python as
  `catkv_ops`.
- `catkv-transformers/`: CatKV evaluation code based on Hugging Face
  Transformers.
- `catkv-vllm/`: service-side KV cache code based on LMCache and vLLM.
- `dataset/`: small release datasets used by the demos, including
  `wikimqa_s.jsonl`.

## Install

Use Python 3.12 for the Transformers experiments. The native extension and the
vLLM integration also need a working CUDA toolkit, a CUDA-compatible PyTorch
build, and the system development libraries for libcurl and OpenSSL.

On Ubuntu-like systems, install the native build dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential ninja-build libcurl4-openssl-dev libssl-dev
```

Create and activate a Python environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja packaging
```

Install PyTorch first, choosing the wheel that matches the CUDA version on the
machine. For CUDA 12.8:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
```

Install the CatKV native extension:

```bash
pip install -e ./csrc --no-build-isolation
```

Install the Transformers experiment package:

```bash
pip install -e ./catkv-transformers
```

Install the vLLM/LMCache integration:

```bash
pip install -e ./catkv-vllm --no-build-isolation
```

Verify the Python packages:

```bash
python - <<'PY'
import catkv_ops
import lmcache
print("catkv_ops:", catkv_ops.__file__)
print("lmcache:", lmcache.__file__)
PY
```

## S3/MinIO Config

Remote KV-cache upload in `catkv_ops` reads an INI file with an `[s3]` section.
Start from the example config:

```bash
cp csrc/config/s3.ini.example /path/to/s3.ini
```

Edit these fields:

```ini
[s3]
endpoint = 127.0.0.1:9000
access_key = minioadmin
secret_key = minioadmin
bucket = catkv-ops
region = us-east-1
use_ssl = false
```

- `endpoint`: S3 or MinIO host and port, without `http://` or `https://`.
- `access_key` / `secret_key`: credentials with read and write access.
- `bucket`: destination bucket for remote KV-cache objects. Create it before
  running remote upload.
- `region`: S3 region. `us-east-1` is commonly used for local MinIO.
- `use_ssl`: `true` for HTTPS endpoints, `false` for plain HTTP.

Use the config path when enabling remote upload:

```python
from catkv_ops import CPUMemoryStore, S3Manager

s3_config = "/path/to/s3.ini"
store = CPUMemoryStore(pin_memory=True, offload_workers=2)
store.enable_remote_upload(
    s3_config,
    ratio=0.2,
    num_workers=4,
)

manager = S3Manager(s3_config)
```

Remote upload writes objects to the configured bucket in the background. The
same config file is used by `S3Manager` for direct save/load helpers.

## What `csrc/` Does

`csrc/` builds the native `catkv-ops` extension, imported as `catkv_ops`.
It provides the low-level runtime used by the higher-level demos:

- GPU -> CPU KV cache offload for `torch.bfloat16` tensors.
- `CPUMemoryStore` for async offload, local CPU cache storage, and loading
  cached tensors back to CPU or GPU.
- Optional remote pipeline: after local offload, KV tensors can be compressed
  and uploaded to S3/MinIO in the background.
- S3/MinIO tensor save/load helpers used by remote KV cache paths.
- CUDA fused dequant kernels for compressed KV cache recovery.

In short, `csrc/` is the native offload/storage/compression layer; the
Transformers and vLLM code call into it when they need fast KV cache movement
or remote cache persistence.

## Demo: `exp.eval` in Transformers

Run the minimal WikiMQA evaluation through the Transformers-based experiment
entry point:

```bash
cd catkv-transformers
python -m exp.eval \
  -m qwen-2.5-3b-instruct.yaml \
  -t epic.yaml
```

This uses:

- model config: `catkv-transformers/config/models/qwen-2.5-3b-instruct.yaml`
- task config: `catkv-transformers/config/task/epic.yaml`
- dataset entry: `wikimqa` from
  `catkv-transformers/config/dataset/dataset2path.json`

The task writes results under the configured result directory and uses
`cuda:0` from `epic.yaml` by default.

## Demo: `run_wikimqa_outputs.py` in vLLM

Run the WikiMQA output demo through the vLLM/LMCache integration:

```bash
cd catkv-vllm
python examples/blend_kv_v1/run_wikimqa_outputs.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset ../dataset/wikimqa_s.jsonl \
  --num-samples 5 \
  --cuda-visible-devices 0
```

The script warms WikiMQA contexts, generates answers, and writes JSONL output
under `catkv-vllm/examples/blend_kv_v1/results/` unless `--output` is provided.

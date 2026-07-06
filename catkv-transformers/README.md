# CatKV

CatKV is a research project focused on efficient key-value cache management for large language models, implementing various caching strategies and attention mechanisms to optimize memory usage and inference performance.

## Features

- **Efficient KV Cache Management**: Advanced key-value cache strategies including LRU and custom blend cache implementations
- **Multiple Attention Mechanisms**: Support for various attention implementations including cascade dot-product attention
- **GPU Memory Optimization**: CUDA-based cache management with asynchronous operations
- **Flexible Model Support**: Compatible with multiple model architectures (Llama, Mistral, Qwen series)
- **Performance Profiling**: Built-in NVTX annotations for detailed performance analysis
- **Multiple Datasets**: Support for various benchmarks including SamSum, TriviaQA, RepoBench, and more

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install the package:
```bash
pip install -e .
```

## Quick Start

### Running Experiments

The project includes several test scripts for different datasets:

```bash
python -m exp.eval -m config/qwen-2.5-7b-instruct.yaml -r result -t config/task/epic.yaml  [config_path] [result_dir]
```


### Configuration

Model configurations are available in the [`config/`](config/) directory:
- [`llama-3.1-8b-instruct.yaml`](config/llama-3.1-8b-instruct.yaml)
- [`mistral-7b-instruct-v0.2.yaml`](config/mistral-7b-instruct-v0.2.yaml)
- [`qwen-2.5-7b-instruct.yaml`](config/qwen-2.5-7b-instruct.yaml)
- And more...


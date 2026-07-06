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

1. Clone the repository:
```bash
git clone <repository-url>
cd CatKV
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
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


## Experiment Parameters

Common parameters for experiments:
- `--config_path`: Path to model configuration file
- `--result_dir`: Directory to save results
- `--dataset_path`: Path to dataset file
- `--max_output`: Maximum output length
- `--select_type`: Cache selection strategy ("value", "key", "epic")
- `--recompute_type`: Recomputation strategy ("token", "block")
- `--recompute_ratio`: Ratio for recomputation (0.0-1.0)
- `--precompute`: Enable precomputation

## Results

Results are saved in JSONL format in the [`result/`](result/) directory, organized by dataset and model. Each result entry includes:
- Prediction text
- Time to First Token (TTFT)
- Ground truth answer
- Evaluation score

## Testing

Run the test suite:
```bash
./test.sh
```

Individual tests are available in the [`test/`](test/) directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use CatKV in your research, please cite:

```bibtex
@misc{CatKV2024,
  title={CatKV: Efficient Key-Value Cache Management for Large Language Models},
  author={...},
  year={2024},
  url={...}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
#!/usr/bin/env python3
"""
Custom benchmark script that combines vllm/benchmarks/datasets.py dataset loading
with vllm/benchmarks/serve.py request sending functionality.

This script allows you to use custom datasets while sending requests like serve.py,
but without manually tracking latency metrics - designed for use with Prometheus
monitoring.
"""

import argparse
import asyncio
import os
import random
from datetime import datetime
from typing import Optional

import numpy as np  # Still needed for gamma distribution in get_request_with_burstiness
from tqdm.asyncio import tqdm

# Import from vllm benchmarks
from vllm.benchmarks.endpoint_request_func import (ASYNC_REQUEST_FUNCS,
                                                   RequestFuncInput,
                                                   RequestFuncOutput)
from vllm.benchmarks.datasets import (
    BenchmarkDataset, RandomDataset, ShareGPTDataset, SonnetDataset,
    BurstGPTDataset, ConversationDataset, VisionArenaDataset,
    InstructCoderDataset, AIMODataset, NextEditPredictionDataset,
    InfiniteBenchDataset, SampleRequest
)
from vllm.transformers_utils.tokenizer import get_tokenizer


# Dataset registry mapping dataset type names to classes
DATASET_REGISTRY = {
    'random': RandomDataset,
    'sharegpt': ShareGPTDataset,
    'sonnet': SonnetDataset,
    'burstgpt': BurstGPTDataset,
    'conversation': ConversationDataset,
    'vision-arena': VisionArenaDataset,
    'instruct-coder': InstructCoderDataset,
    'aimo': AIMODataset,
    'next-edit-prediction': NextEditPredictionDataset,
    'infinitebench': InfiniteBenchDataset,
}


def load_dataset(dataset_type: str,
                 dataset_path: Optional[str] = None,
                 dataset_split: str = "train",
                 dataset_subset: Optional[str] = None,
                 random_seed: int = 0) -> BenchmarkDataset:
    """Load a dataset based on type and path."""
    
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. "
                        f"Supported types: {list(DATASET_REGISTRY.keys())}")
    
    dataset_class = DATASET_REGISTRY[dataset_type]
    
    # Handle HuggingFace datasets that need special parameters
    if dataset_type in ['conversation', 'vision-arena', 'instruct-coder', 
                       'aimo', 'next-edit-prediction']:
        if not dataset_path:
            raise ValueError(f"dataset_path is required for {dataset_type}")
        return dataset_class(
            dataset_path=dataset_path,
            dataset_split=dataset_split,
            dataset_subset=dataset_subset,
            random_seed=random_seed
        )
    elif dataset_type == 'infinitebench':
        if not dataset_path:
            raise ValueError(f"dataset_path is required for {dataset_type}")
        return dataset_class(
            dataset_path=dataset_path,
            dataset_split=dataset_split,  # For InfiniteBench, this should be a task name
            dataset_subset=dataset_subset,
            random_seed=random_seed
        )
    else:
        # Regular datasets
        return dataset_class(
            dataset_path=dataset_path,
            random_seed=random_seed
        )


async def get_request_with_burstiness(
    requests: list[SampleRequest],
    request_rate: float,
    burstiness: float = 1.0,
):
    """
    Generate requests at specified rate with burstiness control.
    
    Args:
        requests: List of sample requests
        request_rate: Requests per second
        burstiness: Burstiness factor (1.0 = Poisson, <1 = bursty, >1 = uniform)
    """
    if request_rate == float("inf"):
        # Unlimited rate - yield all requests immediately
        for req in requests:
            yield req
        return
    
    # Calculate scale parameter theta to maintain the desired request_rate
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}.")
    theta = 1.0 / (request_rate * burstiness)
    
    for req in requests:
        yield req
        
        # Sample the request interval from the gamma distribution
        # If burstiness is 1, it follows exponential distribution
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(interval)


async def send_requests(
    endpoint_type: str,
    api_url: str,
    model_id: str,
    model_name: Optional[str],
    requests: list[SampleRequest],
    request_rate: float = float("inf"),
    burstiness: float = 1.0,
    max_concurrency: Optional[int] = None,
    disable_tqdm: bool = False,
    logprobs: Optional[int] = None,
    best_of: int = 1,
    ignore_eos: bool = False
) -> None:
    """Send requests to the endpoint without collecting metrics."""

    if endpoint_type not in ASYNC_REQUEST_FUNCS:
        raise ValueError(f"Unknown endpoint_type: {endpoint_type}")

    request_func = ASYNC_REQUEST_FUNCS[endpoint_type]

    print(f"Sending {len(requests)} requests to {api_url}")
    print(f"Request rate: {request_rate}")
    print(f"Burstiness: {burstiness}")
    print(f"Max concurrency: {max_concurrency}")

    # Test with first request
    if requests:
        test_req = requests[0]
        test_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=test_req.prompt,
            api_url=api_url,
            prompt_len=test_req.prompt_len,
            output_len=test_req.expected_output_len,
            logprobs=logprobs,
            best_of=best_of,
            multi_modal_content=test_req.multi_modal_data,
            ignore_eos=ignore_eos,
        )

        print("Testing connection with first request...")
        test_output = await request_func(request_func_input=test_input)
        if not test_output.success:
            raise ValueError(f"Test request failed: {test_output.error}")
        print("Connection test successful!")

    # Prepare semaphore for concurrency control
    semaphore = (asyncio.Semaphore(max_concurrency)
                if max_concurrency else None)

    async def limited_request_func(request_input, pbar):
        if semaphore is None:
            return await request_func(request_func_input=request_input,
                                      pbar=pbar)
        async with semaphore:
            return await request_func(request_func_input=request_input,
                                      pbar=pbar)

    # Progress bar
    pbar = None if disable_tqdm else tqdm(total=len(requests))

    # Send all requests with burstiness control
    tasks = []

    async for req in get_request_with_burstiness(requests, request_rate,
                                                 burstiness):
        request_input = RequestFuncInput(
            model=model_id,
            model_name=model_name,
            prompt=req.prompt,
            api_url=api_url,
            prompt_len=req.prompt_len,
            output_len=req.expected_output_len,
            logprobs=logprobs,
            best_of=best_of,
            multi_modal_content=req.multi_modal_data,
            ignore_eos=ignore_eos,
        )

        tasks.append(
            asyncio.create_task(
                limited_request_func(request_input=request_input, pbar=pbar)
            )
        )

    # Wait for all requests to complete
    outputs = await asyncio.gather(*tasks, return_exceptions=True)

    if pbar is not None:
        pbar.close()

    # Filter out successful outputs and count
    successful_count = 0
    for output in outputs:
        if isinstance(output, RequestFuncOutput) and output.success:
            successful_count += 1
        elif isinstance(output, Exception):
            print(f"Request failed with exception: {output}")

    print(f"Completed {successful_count}/{len(requests)} requests successfully")

    
def add_cli_args(parser: argparse.ArgumentParser):
    """Add command line arguments for custom benchmark."""
    
    # Dataset arguments
    parser.add_argument(
        "--dataset-type",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Type of dataset to use"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the dataset file (required for most dataset types)"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split for HuggingFace datasets"
    )
    parser.add_argument(
        "--dataset-subset",
        type=str,
        help="Dataset subset for HuggingFace datasets"
    )
    
    # Request parameters
    parser.add_argument(
        "--num-requests",
        type=int,
        default=100,
        help="Number of requests to send"
    )
    
    # Server connection
    parser.add_argument(
        "--endpoint-type",
        type=str,
        default="openai-comp",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
        help="Type of endpoint to use"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Server base URL"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint path"
    )
    
    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name"
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        help="Model name as served by the API"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="auto",
        choices=['auto', 'slow', 'mistral', 'custom'],
        help="Tokenizer mode"
    )
    
    # Request control
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf for unlimited)"
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor of request generation. "
        "Default=1.0 (Poisson process). "
        "<1.0 = more bursty, >1.0 = more uniform"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        help="Maximum concurrent requests"
    )
    
    # Other options
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code for tokenizer"
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable progress bar"
    )
    parser.add_argument(
        "--logprobs",
        type=int,
        help="Number of logprobs to return"
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generate best_of sequences and return the best"
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS token"
    )

    # Dataset-specific arguments for RandomDataset
    random_group = parser.add_argument_group("random dataset options")
    random_group.add_argument("--input-len", type=int, default=1024)
    random_group.add_argument("--output-len", type=int, default=128)
    random_group.add_argument("--prefix-len", type=int, default=0)
    random_group.add_argument("--range-ratio", type=float, default=0.0)


def main(args: argparse.Namespace):
    """Main function for custom benchmark."""
    async def run_benchmark():
        # Set random seeds
        random.seed(args.seed)
        np.random.seed(args.seed)

        # Setup URLs
        if args.base_url is not None:
            api_url = f"{args.base_url}{args.endpoint}"
        else:
            api_url = f"http://{args.host}:{args.port}{args.endpoint}"

        # Setup tokenizer
        tokenizer_id = args.tokenizer or args.model
        tokenizer = get_tokenizer(
            tokenizer_id,
            tokenizer_mode=args.tokenizer_mode,
            trust_remote_code=args.trust_remote_code
        )

        # Load dataset
        print(f"Loading {args.dataset_type} dataset...")
        dataset = load_dataset(
            dataset_type=args.dataset_type,
            dataset_path=args.dataset_path,
            dataset_split=args.dataset_split,
            dataset_subset=args.dataset_subset,
            random_seed=args.seed
        )

        # Sample requests from dataset
        print(f"Sampling {args.num_requests} requests from dataset...")

        # Prepare kwargs for sampling based on dataset type
        sample_kwargs = {}
        if args.dataset_type == 'random':
            sample_kwargs.update({
                'input_len': args.input_len,
                'output_len': args.output_len,
                'prefix_len': args.prefix_len,
                'range_ratio': args.range_ratio,
            })

        requests = dataset.sample(
            tokenizer=tokenizer,
            num_requests=args.num_requests,
            **sample_kwargs
        )

        print(f"Loaded {len(requests)} requests")

        # Send requests (no metrics collection)
        await send_requests(
            endpoint_type=args.endpoint_type,
            api_url=api_url,
            model_id=args.model,
            model_name=args.served_model_name,
            requests=requests,
            request_rate=args.request_rate,
            burstiness=args.burstiness,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            logprobs=args.logprobs,
            best_of=args.best_of,
            ignore_eos=args.ignore_eos
        )

        print("Benchmark completed!")
        print("Note: Metrics are collected via Prometheus, not in this script")

    # Run the benchmark
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Custom benchmark using datasets.py with serve.py "
        "request sending"
    )
    add_cli_args(parser)
    args = parser.parse_args()
    main(args)
    
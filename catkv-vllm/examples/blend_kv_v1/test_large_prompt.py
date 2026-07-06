#!/usr/bin/env python3
"""Large-prompt test for observing real cache speedup."""
import argparse
import asyncio
import json
import time
from statistics import mean, median

import aiohttp
import numpy as np
from transformers import AutoTokenizer


api_base = "http://localhost:12345"
model = "mistralai/Mistral-7B-Instruct-v0.3"


def parse_args():
    parser = argparse.ArgumentParser(description="Blend KV large prompt test")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Request rate (RPS). Use inf for unlimited.",
    )
    parser.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="Burstiness factor. 1.0 corresponds to a Poisson process.",
    )
    return parser.parse_args()


def _format_ms(value):
    return f"{value * 1000:.2f} ms" if value is not None else "N/A"


async def send_requests_with_rate_limit(prompts, request_rate, burstiness):
    prompts = list(prompts)
    if not prompts:
        return
    if request_rate == float("inf"):
        for prompt in prompts:
            yield prompt
        return
    assert burstiness > 0, (
        f"A positive burstiness factor is expected, but given {burstiness}."
    )
    theta = 1.0 / (request_rate * burstiness)
    total = len(prompts)
    for idx, prompt in enumerate(prompts):
        yield prompt
        if idx == total - 1:
            break
        interval = np.random.gamma(shape=burstiness, scale=theta)
        await asyncio.sleep(float(interval))


async def stream_completion(session, prompt, request_name):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_time = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=120)
    async with session.post(
        f"{api_base}/v1/completions", json=payload, timeout=timeout
    ) as response:
        response.raise_for_status()
        print(f"{request_name} Response received, streaming...")
        first_token_time = None
        most_recent_timestamp = start_time
        generated_chunks = []
        itl = []
        prompt_tokens = None
        completion_tokens = None
        success = False
        buffer = ""
        done = False
        async for raw_chunk in response.content:
            if not raw_chunk:
                continue
            buffer += raw_chunk.decode("utf-8")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                choices = chunk.get("choices")
                usage = chunk.get("usage")
                if choices:
                    timestamp = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = timestamp
                        success = True
                    else:
                        itl.append(timestamp - most_recent_timestamp)
                    most_recent_timestamp = timestamp
                    text = choices[0].get("text") or ""
                    generated_chunks.append(text)
                elif usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get(
                        "completion_tokens", completion_tokens
                    )
            if done:
                break
    ttft = first_token_time - start_time if first_token_time is not None else None
    latency = most_recent_timestamp - start_time
    metrics = {
        "generated_text": "".join(generated_chunks),
        "ttft": ttft,
        "itl": itl,
        "latency": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "success": success,
    }
    return metrics


def print_metrics(metrics):
    print(metrics["generated_text"])
    if not metrics["success"]:
        print("Warning: no chunk with choices was received; metrics may be invalid.")
    print(f"TTFT: {_format_ms(metrics['ttft'])}")
    if metrics["itl"]:
        mean_itl = mean(metrics["itl"])
        median_itl = median(metrics["itl"])
    else:
        mean_itl = None
        median_itl = None
    print(f"Mean ITL: {_format_ms(mean_itl)}, Median ITL: {_format_ms(median_itl)}")
    print(f"Total latency: {_format_ms(metrics['latency'])}")
    prompt_tokens = metrics["prompt_tokens"]
    completion_tokens = metrics["completion_tokens"]
    prompt_tokens_str = prompt_tokens if prompt_tokens is not None else "N/A"
    completion_tokens_str = (
        completion_tokens if completion_tokens is not None else "N/A"
    )
    print(
        f"Prompt tokens: {prompt_tokens_str} | Completion tokens: {completion_tokens_str}"
    )


async def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Larger chunks, each about 2000 tokens.
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1 = tokenizer.encode("Hello, how are you doing today? " * 300)[1:]
    chunk2 = tokenizer.encode("What's your favorite color? " * 300)[1:]
    chunk3 = tokenizer.encode("Tell me about yourself. " * 300)[1:]
    sep = tokenizer.encode(" # # ")[1:]
    question1 = tokenizer.encode("Please help me")[1:]
    question2 = tokenizer.encode("What do you think?")[1:]
    question3 = tokenizer.encode("Can you assist?")[1:]

    # prompt1 = chunk1
    # prompt2 = chunk2
    # prompt3 = chunk3

    prompt1 = sys_prompt  + chunk1  + chunk2  + chunk3  + question1
    prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + question2
    prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + question3

    request_info = [
        ("[Request 1] cold start", prompt1),
        ("[Request 2] different order, should reuse cache", prompt2),
        ("[Request 3] another different order", prompt3),
    ]

    print("=" * 60)
    print("Large prompt test, about 6K+ tokens each")
    print("=" * 60)
    print(f"Prompt1 total length: {len(prompt1)} tokens")
    print(f"Request rate (RPS): {args.request_rate}")
    print(f"Burstiness: {args.burstiness}")
    print()

    async with aiohttp.ClientSession() as session:
        # Warmup phase: store cache first and wait for the service to stabilize.
        print("=" * 60)
        print("Warmup phase: storing cache...")
        print("=" * 60)
        warmup_metrics = await stream_completion(
            session=session,
            prompt=prompt1,
            request_name="[Warmup-Store]"
        )
        print("[Warmup-Store] completed")
        print_metrics(warmup_metrics)
        print()

        print("Waiting 3 seconds before the retrieval benchmark...")
        await asyncio.sleep(3)
        print()

        print("=" * 60)
        print("Starting benchmark for prompt2 and prompt3 retrieval performance")
        print("=" * 60)

        # Only test prompt2 and prompt3 because they retrieve cache.
        prompts = [info[1] for info in request_info[1:]]  # Skip prompt1.
        tasks = []
        idx = 1  # Start from request_info[1].
        async for prompt in send_requests_with_rate_limit(
            prompts, args.request_rate, args.burstiness
        ):
            request_name = request_info[idx][0]
            tasks.append(
                asyncio.create_task(
                    stream_completion(
                        session=session, prompt=prompt, request_name=request_name
                    )
                )
            )
            idx += 1

        metrics_list = await asyncio.gather(*tasks)

    print()
    print("=" * 60)
    print("Benchmark results")
    print("=" * 60)

    metrics2, metrics3 = metrics_list

    print(request_info[1][0])
    print_metrics(metrics2)
    if warmup_metrics["ttft"] is not None and metrics2["ttft"] is not None:
        print(f"Speedup vs warmup: {warmup_metrics['ttft']/metrics2['ttft']:.2f}x")
    else:
        print("Speedup: cannot compute because TTFT data is missing")
    print()

    print(request_info[2][0])
    print_metrics(metrics3)
    if warmup_metrics["ttft"] is not None and metrics3["ttft"] is not None:
        print(f"Speedup vs warmup: {warmup_metrics['ttft']/metrics3['ttft']:.2f}x")
    else:
        print("Speedup: cannot compute because TTFT data is missing")
    print()

    print("=" * 60)
    print("Expected: larger prompts should show clearer speedup")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

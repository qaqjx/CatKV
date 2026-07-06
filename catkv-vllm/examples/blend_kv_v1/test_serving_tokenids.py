#!/usr/bin/env python3
"""
CacheBlend serving test client using the vLLM Python API.

Run this script after serving has started. It uses the vLLM Python SDK.
"""

import os
import time
import requests
from transformers import AutoTokenizer


def test_with_openai_api():
    """Test through the OpenAI API over HTTP."""
    api_base = "http://localhost:8000"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    print("=" * 60)
    print("Method 1: send token IDs directly through the vLLM HTTP API")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Prepare token IDs and skip the BOS token where needed.
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1 = tokenizer.encode("Hello, how are you?" * 500)[1:]
    chunk2 = tokenizer.encode("Hello, what's up?" * 500)[1:]
    chunk3 = tokenizer.encode("Hi, what are you up to?" * 500)[1:]
    sep = tokenizer.encode(" # # ")[1:]  # Skip BOS.
    question1 = tokenizer.encode("Hello, my name is")[1:]
    question2 = tokenizer.encode("Hello, how are you?")[1:]
    question3 = tokenizer.encode("Hello, what's up?")[1:]

    # Build three prompts with different chunk orders.
    prompt1 = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + question1
    prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + question2
    prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + question3

    print(f"Prompt 1 length: {len(prompt1)} tokens")
    print(f"Prompt 2 length: {len(prompt2)} tokens")
    print(f"Prompt 3 length: {len(prompt3)} tokens")
    print(f"Separator tokens: {sep}")
    print()

    # First request.
    print("[Request 1] order: sys + chunk1 + chunk2 + chunk3")
    start = time.time()
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": model,
            "prompt": prompt1,
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60
    )
    ttft1 = time.time() - start
    result1 = response.json()
    print(f"Generated text: {result1['choices'][0]['text']}")
    print(f"TTFT: {ttft1:.2f} s")
    print()

    time.sleep(1)

    # Second request.
    print("[Request 2] order: sys + chunk2 + chunk1 + chunk3 (should reuse cache)")
    start = time.time()
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": model,
            "prompt": prompt2,
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60
    )
    ttft2 = time.time() - start
    result2 = response.json()
    print(f"Generated text: {result2['choices'][0]['text']}")
    print(f"TTFT: {ttft2:.2f} s")
    print(f"Speedup: {ttft1/ttft2:.2f}x")
    print()

    time.sleep(1)

    # Third request.
    print("[Request 3] order: sys + chunk3 + chunk1 + chunk2 (should reuse cache)")
    start = time.time()
    response = requests.post(
        f"{api_base}/v1/completions",
        json={
            "model": model,
            "prompt": prompt3,
            "max_tokens": 10,
            "temperature": 0,
        },
        timeout=60
    )
    ttft3 = time.time() - start
    result3 = response.json()
    print(f"Generated text: {result3['choices'][0]['text']}")
    print(f"TTFT: {ttft3:.2f} s")
    print(f"Speedup: {ttft1/ttft3:.2f}x")
    print()

    print("=" * 60)
    print("Test completed.")
    print("If requests 2 and 3 are clearly faster (>2x), CacheBlend is working.")
    print("=" * 60)


if __name__ == "__main__":
    test_with_openai_api()

#!/usr/bin/env python3
"""
CacheBlend serving test client.

Demonstrates how to call LMCache CacheBlend serving through the OpenAI API.
"""

import os
import time
from openai import OpenAI


def main():
    # Configuration.
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    model = os.getenv("MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    # Separator. This must match LMCACHE_BLEND_SPECIAL_STR on the server side.
    sep = " # # "

    # Initialize client.
    client = OpenAI(
        api_key="EMPTY",
        base_url=api_base
    )

    print("=" * 60)
    print("LMCache CacheBlend serving test")
    print("=" * 60)
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"Separator: '{sep}'")
    print("=" * 60)

    # Define shared chunks.
    sys_prompt = "You are a very helpful assistant."
    chunk1 = "Hello, how are you?" * 500
    chunk2 = "Hello, what's up?" * 500
    chunk3 = "Hi, what are you up to?" * 500

    # First request.
    print("\n[Request 1] order: sys + chunk1 + chunk2 + chunk3")
    prompt1 = f"{sys_prompt}{sep}{chunk1}{sep}{chunk2}{sep}{chunk3}{sep}Hello, my name is"
    start = time.time()
    response1 = client.completions.create(
        model=model,
        prompt=prompt1,
        max_tokens=10,
        temperature=0
    )
    ttft1 = time.time() - start
    print(f"Generated text: {response1.choices[0].text}")
    print(f"TTFT: {ttft1:.2f} s")

    time.sleep(1)

    # Second request with a different order, expected to reuse cache.
    print("\n[Request 2] order: sys + chunk2 + chunk1 + chunk3 (should reuse cache)")
    prompt2 = f"{sys_prompt}{sep}{chunk2}{sep}{chunk1}{sep}{chunk3}{sep}Hello, how are you?"
    start = time.time()
    response2 = client.completions.create(
        model=model,
        prompt=prompt2,
        max_tokens=10,
        temperature=0
    )
    ttft2 = time.time() - start
    print(f"Generated text: {response2.choices[0].text}")
    print(f"TTFT: {ttft2:.2f} s")
    print(f"Speedup: {ttft1/ttft2:.2f}x")

    time.sleep(1)

    # Third request with another different order.
    print("\n[Request 3] order: sys + chunk3 + chunk1 + chunk2 (should reuse cache)")
    prompt3 = f"{sys_prompt}{sep}{chunk3}{sep}{chunk1}{sep}{chunk2}{sep}Hello, what's up?"
    start = time.time()
    response3 = client.completions.create(
        model=model,
        prompt=prompt3,
        max_tokens=10,
        temperature=0
    )
    ttft3 = time.time() - start
    print(f"Generated text: {response3.choices[0].text}")
    print(f"TTFT: {ttft3:.2f} s")
    print(f"Speedup: {ttft1/ttft3:.2f}x")

    print("\n" + "=" * 60)
    print("Test completed.")
    print("If TTFT drops clearly, CacheBlend is working correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()

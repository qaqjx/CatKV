#!/usr/bin/env python3
"""
CacheBlend test with a smaller chunk size.
"""

import time
import requests
from transformers import AutoTokenizer


def main():
    api_base = "http://localhost:8000"
    model = "mistralai/Mistral-7B-Instruct-v0.2"

    print("=" * 60)
    print("CacheBlend test with adjusted chunk size")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Reduce repetitions from 500 to 50.
    sys_prompt = tokenizer.encode("You are a very helpful assistant.")
    chunk1 = tokenizer.encode("Hello, how are you?" * 50)[1:]  # Reduced to 50.
    chunk2 = tokenizer.encode("Hello, what's up?" * 50)[1:]
    chunk3 = tokenizer.encode("Hi, what are you up to?" * 50)[1:]
    sep = tokenizer.encode(" # # ")[1:]
    question1 = tokenizer.encode("Hello, my name is")[1:]
    question2 = tokenizer.encode("Hello, how are you?")[1:]
    question3 = tokenizer.encode("Hello, what's up?")[1:]

    prompt1 = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + question1
    prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + question2
    prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + question3

    print(f"Chunk1 length: {len(chunk1)} tokens")
    print(f"Chunk2 length: {len(chunk2)} tokens")
    print(f"Chunk3 length: {len(chunk3)} tokens")
    print(f"Prompt1 total length: {len(prompt1)} tokens")
    print()

    # Request 1.
    print("[Request 1] cold start")
    start = time.time()
    r1 = requests.post(f"{api_base}/v1/completions", json={
        "model": model, "prompt": prompt1, "max_tokens": 10, "temperature": 0
    }, timeout=60)
    ttft1 = time.time() - start
    print(f"Generated: {r1.json()['choices'][0]['text']}")
    print(f"TTFT: {ttft1:.2f} s\n")

    time.sleep(1)

    # Request 2.
    print("[Request 2] different order")
    start = time.time()
    r2 = requests.post(f"{api_base}/v1/completions", json={
        "model": model, "prompt": prompt2, "max_tokens": 10, "temperature": 0
    }, timeout=60)
    ttft2 = time.time() - start
    print(f"result r2 :{r2.json()}")
    print(f"Generated: {r2.json()['choices'][0]['text']}")
    print(f"TTFT: {ttft2:.2f} s")
    print(f"Speedup: {ttft1/ttft2:.2f}x\n")

    time.sleep(1)

    # Request 3.
    print("[Request 3] another different order")
    start = time.time()
    r3 = requests.post(f"{api_base}/v1/completions", json={
        "model": model, "prompt": prompt3, "max_tokens": 10, "temperature": 0
    }, timeout=60)
    ttft3 = time.time() - start
    print(f"Generated: {r3.json()['choices'][0]['text']}")
    print(f"TTFT: {ttft3:.2f} s")
    print(f"Speedup: {ttft1/ttft3:.2f}x\n")

    print("=" * 60)
    print("Expected: requests 2 and 3 should be clearly faster")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Baseline comparison: with LMCache vs without LMCache."""
import time
import requests

api_base = "http://localhost:8000"
model = "mistralai/Mistral-7B-Instruct-v0.2"

# Simple repeated prompt that simulates a cache reuse scenario.
prompt = "Hello, how are you? " * 200  # About 1200 tokens.

print("=" * 60)
print("Baseline test: three identical requests")
print("=" * 60)

times = []
for i in range(3):
    print(f"\n[Request {i+1}]")
    start = time.time()
    r = requests.post(f"{api_base}/v1/completions", json={
        "model": model,
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0
    }, timeout=60)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Elapsed: {elapsed:.2f} s")
    time.sleep(0.5)

print("\n" + "=" * 60)
print(f"Request 1: {times[0]:.2f} s (cold start)")
print(f"Request 2: {times[1]:.2f} s (speedup: {times[0]/times[1]:.2f}x)")
print(f"Request 3: {times[2]:.2f} s (speedup: {times[0]/times[2]:.2f}x)")
print("=" * 60)

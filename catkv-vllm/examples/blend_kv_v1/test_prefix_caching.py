#!/usr/bin/env python3
"""Test prefix reuse when the prefix remains at the same position."""
import time
import requests
from transformers import AutoTokenizer

api_base = "http://localhost:8000"
model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model)

# Shared prefix plus different suffixes for a same-position test.
prefix = tokenizer.encode("You are a helpful assistant. " * 100)  # ~2000 tokens
sep = tokenizer.encode(" # # ")[1:]
suffix1 = tokenizer.encode("Question 1: What is AI?" * 100)[1:]
suffix2 = tokenizer.encode("Question 2: What is ML?" * 100)[1:]
suffix3 = tokenizer.encode("Question 3: What is DL?" * 100)[1:]

prompt1 = prefix + sep + suffix1
prompt2 = prefix + sep + suffix2
prompt3 = prefix + sep + suffix3

print("=" * 60)
print("Prefix reuse test, same position with different content")
print("=" * 60)
print(f"Shared prefix: {len(prefix)} tokens")
print(f"Prompt1 total length: {len(prompt1)} tokens\n")

times = []

# Request 1.
print("[Request 1] cold start")
start = time.time()
r1 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt1, "max_tokens": 5, "temperature": 0
}, timeout=120)
t1 = time.time() - start
times.append(t1)
print(f"TTFT: {t1:.3f} s\n")
time.sleep(1)

# Request 2 with the same prefix.
print("[Request 2] same prefix, different suffix")
start = time.time()
r2 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt2, "max_tokens": 5, "temperature": 0
}, timeout=120)
t2 = time.time() - start
times.append(t2)
print(f"TTFT: {t2:.3f} s")
print(f"Speedup: {t1/t2:.2f}x\n")
time.sleep(1)

# Request 3 with the same prefix.
print("[Request 3] same prefix, another different suffix")
start = time.time()
r3 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt3, "max_tokens": 5, "temperature": 0
}, timeout=120)
t3 = time.time() - start
times.append(t3)
print(f"TTFT: {t3:.3f} s")
print(f"Speedup: {t1/t3:.2f}x\n")

print("=" * 60)
print("Summary:")
print(f"  Cold start: {t1:.3f} s")
print(f"  Average warm start: {(t2+t3)/2:.3f} s")
print(f"  Average speedup: {t1/((t2+t3)/2):.2f}x")
print("Expected: prefix reuse should provide clear speedup")
print("=" * 60)

#!/usr/bin/env python3
"""Medium-size prompt test that stays within a safer range."""
import time
import requests
from transformers import AutoTokenizer

api_base = "http://localhost:8000"
model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model)

# Medium-size chunks, each about 800 tokens in the safer range.
sys_prompt = tokenizer.encode("You are a helpful assistant.")
chunk1 = tokenizer.encode("Hello, how are you? " * 120)[1:]  # ~800 tokens
chunk2 = tokenizer.encode("What's up? " * 120)[1:]
chunk3 = tokenizer.encode("How can I help? " * 120)[1:]
sep = tokenizer.encode(" # # ")[1:]
q1 = tokenizer.encode("Please help me")[1:]
q2 = tokenizer.encode("Thank you")[1:]
q3 = tokenizer.encode("Great")[1:]

prompt1 = sys_prompt + sep + chunk1 + sep + chunk2 + sep + chunk3 + sep + q1
prompt2 = sys_prompt + sep + chunk2 + sep + chunk1 + sep + chunk3 + sep + q2
prompt3 = sys_prompt + sep + chunk3 + sep + chunk1 + sep + chunk2 + sep + q3

print("=" * 60)
print("Medium prompt test, about 2.5K tokens in a safer range")
print("=" * 60)
print(f"Prompt1 length: {len(prompt1)} tokens")
print()

times = []

# Request 1.
print("[Request 1] cold start")
start = time.time()
r1 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt1, "max_tokens": 5, "temperature": 0
}, timeout=60)
t1 = time.time() - start
times.append(t1)
print(f"TTFT: {t1:.3f} s\n")
time.sleep(1)

# Request 2.
print("[Request 2] different order")
start = time.time()
r2 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt2, "max_tokens": 5, "temperature": 0
}, timeout=60)
t2 = time.time() - start
times.append(t2)
print(f"TTFT: {t2:.3f} s")
print(f"Speedup: {t1/t2:.2f}x\n")
time.sleep(1)

# Request 3.
print("[Request 3] another different order")
start = time.time()
r3 = requests.post(f"{api_base}/v1/completions", json={
    "model": model, "prompt": prompt3, "max_tokens": 5, "temperature": 0
}, timeout=60)
t3 = time.time() - start
times.append(t3)
print(f"TTFT: {t3:.3f} s")
print(f"Speedup: {t1/t3:.2f}x\n")

print("=" * 60)
print("Summary:")
print(f"  Cold start: {t1:.3f} s")
print(f"  Average warm start: {(t2+t3)/2:.3f} s")
print(f"  Average speedup: {t1/((t2+t3)/2):.2f}x")
print("=" * 60)

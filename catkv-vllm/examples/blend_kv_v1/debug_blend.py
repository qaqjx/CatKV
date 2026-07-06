#!/usr/bin/env python3
"""Debug blend_special_str and segment hashing."""
from transformers import AutoTokenizer

model = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model)

# Test special_str encoding.
special_str = " # # "
sep_tokens_full = tokenizer.encode(special_str)
sep_tokens = tokenizer.encode(special_str)[1:]  # LMCache convention.

print("=" * 60)
print("Special string encoding test")
print("=" * 60)
print(f"Special String: '{special_str}'")
print(f"Full tokens: {sep_tokens_full}")
print(f"Tokens[1:]: {sep_tokens}")
print(f"Decoded: '{tokenizer.decode(sep_tokens)}'")
print()

# Test prompt structure.
sys_prompt = tokenizer.encode("You are a very helpful assistant.")
chunk1 = tokenizer.encode("Hello, how are you doing today? " * 10)[1:]
chunk2 = tokenizer.encode("What's your favorite color? " * 10)[1:]
sep = tokenizer.encode(" # # ")[1:]

prompt1 = sys_prompt + sep + chunk1 + sep + chunk2
prompt2 = sys_prompt + sep + chunk2 + sep + chunk1

print("Prompt structure test")
print("=" * 60)
print(f"sys_prompt length: {len(sys_prompt)}")
print(f"chunk1 length: {len(chunk1)}")
print(f"chunk2 length: {len(chunk2)}")
print(f"sep length: {len(sep)}")
print(f"sep tokens: {sep}")
print()
print(f"prompt1 length: {len(prompt1)}")
print(f"prompt2 length: {len(prompt2)}")
print()

# Check whether sep appears in the prompt.
prompt1_tensor = prompt1
sep_indices_1 = []
for i in range(len(prompt1) - len(sep) + 1):
    if prompt1[i:i+len(sep)] == sep:
        sep_indices_1.append(i)

print(f"sep positions in prompt1: {sep_indices_1}")
print(f"Expected positions: {[len(sys_prompt), len(sys_prompt) + len(sep) + len(chunk1)]}")
print()

# Check tokenizer decoding.
print("Decoded segments:")
print(f"sys_prompt: '{tokenizer.decode(sys_prompt[:50])}...'")
print(f"sep: '{tokenizer.decode(sep)}'")
print(f"chunk1 start: '{tokenizer.decode(chunk1[:30])}...'")
print("=" * 60)

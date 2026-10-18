#!/usr/bin/env python3
"""Measure token generation throughput."""
import json
import time
import urllib.request

url = "http://172.18.0.12:8000/v1/completions"
payload = {
    "model": "Qwen3.5-4B-GGUF",
    "prompt": "Explain quantum computing in detail. Cover the history, key concepts, and future applications.",
    "max_tokens": 256,
    "temperature": 0,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)

t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=300) as resp:
    data = json.loads(resp.read())
t1 = time.perf_counter()

usage = data.get("usage", {})
completion_tokens = usage.get("completion_tokens", 0)
prompt_tokens = usage.get("prompt_tokens", 0)
elapsed = t1 - t0

print(f"Prompt tokens: {prompt_tokens}")
print(f"Completion tokens: {completion_tokens}")
print(f"Total time: {elapsed:.2f}s")
if completion_tokens > 0:
    print(f"Generation speed: {completion_tokens / elapsed:.2f} tok/s")
    print(f"Time to first token: included in total (non-streaming)")

content = data["choices"][0]["text"]
print(f"\nOutput preview: {content[:300]}")

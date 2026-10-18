#!/usr/bin/env python3
"""Profile a single request and capture the vLLM server-side metrics."""
import json
import time
import urllib.request

# Streaming request to measure TTFT and per-token decode latency
url = "http://172.18.0.12:8000/v1/completions"

# First: a short prompt, long generation to isolate decode speed
payload = {
    "model": "Qwen3.5-4B-GGUF",
    "prompt": "Count from 1 to 100:",
    "max_tokens": 128,
    "temperature": 0,
    "stream": True,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)

tokens = []
t0 = time.perf_counter()
first_token_time = None

with urllib.request.urlopen(req, timeout=300) as resp:
    buf = b""
    while True:
        chunk = resp.read(4096)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line or line == b"data: [DONE]":
                continue
            if line.startswith(b"data: "):
                try:
                    data = json.loads(line[6:])
                    text = data["choices"][0].get("text", "")
                    if text:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        tokens.append(now)
                except (json.JSONDecodeError, KeyError):
                    pass

t1 = time.perf_counter()

total = t1 - t0
ttft = first_token_time - t0 if first_token_time else 0
n_tokens = len(tokens)

print(f"Total tokens: {n_tokens}")
print(f"Total time: {total:.2f}s")
print(f"Time to first token (TTFT): {ttft:.3f}s")

if n_tokens > 1:
    decode_time = tokens[-1] - tokens[0]
    decode_tokens = n_tokens - 1
    print(f"Decode tokens: {decode_tokens}")
    print(f"Decode time: {decode_time:.2f}s")
    print(f"Decode speed: {decode_tokens / decode_time:.2f} tok/s")

    # Per-token intervals
    intervals = [tokens[i+1] - tokens[i] for i in range(len(tokens)-1)]
    avg_interval = sum(intervals) / len(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    print(f"\nPer-token interval: avg={avg_interval*1000:.1f}ms min={min_interval*1000:.1f}ms max={max_interval*1000:.1f}ms")

# Now test with a longer prompt to measure prefill speed
print("\n--- Prefill test ---")
long_prompt = "Repeat after me: " + " ".join(f"word{i}" for i in range(200))
payload2 = {
    "model": "Qwen3.5-4B-GGUF",
    "prompt": long_prompt,
    "max_tokens": 1,
    "temperature": 0,
}

req2 = urllib.request.Request(
    url,
    data=json.dumps(payload2).encode(),
    headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
)

t2 = time.perf_counter()
with urllib.request.urlopen(req2, timeout=300) as resp2:
    data2 = json.loads(resp2.read())
t3 = time.perf_counter()

prompt_tokens = data2.get("usage", {}).get("prompt_tokens", 0)
prefill_time = t3 - t2
print(f"Prompt tokens: {prompt_tokens}")
print(f"Prefill time: {prefill_time:.3f}s")
if prompt_tokens > 0:
    print(f"Prefill speed: {prompt_tokens / prefill_time:.1f} tok/s")

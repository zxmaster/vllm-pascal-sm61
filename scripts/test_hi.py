#!/usr/bin/env python3
"""Send a simple 'hi' to vLLM to check model response."""
import urllib.request
import json

data = json.dumps({
    "model": "Qwen3.5-4B-GGUF",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 600,
}).encode()

req = urllib.request.Request(
    "http://172.18.100.6:8000/v1/chat/completions",
    data=data,
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    },
)
resp = urllib.request.urlopen(req, timeout=120)
result = json.loads(resp.read())
choice = result["choices"][0]
content = choice.get("message", {}).get("content") or ""
print("=== FULL CHOICE ===")
print(json.dumps(choice, indent=2)[:2000])
print("=== USAGE ===")
print(json.dumps(result.get("usage", {}), indent=2))

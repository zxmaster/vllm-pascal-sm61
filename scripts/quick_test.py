#!/usr/bin/env python3
"""Send multiple test prompts and print the generated text."""
import json, urllib.request

prompts = [
    "The capital of France is",
    "1+1=",
    "Hello, how are you",
    "The meaning of life is",
]

for prompt in prompts:
    url = "http://172.18.0.12:8000/v1/completions"
    payload = {
        "model": "Qwen3.5-4B-GGUF",
        "prompt": prompt,
        "max_tokens": 20,
        "temperature": 0,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
        content = data['choices'][0]['text']
        print(f"PROMPT: {prompt!r}")
        print(f"OUTPUT: {content[:200]!r}")
        print()

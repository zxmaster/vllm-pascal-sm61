#!/usr/bin/env python3
"""Battery for MiniCPM5-2B pilot (run inside vllm-minicpm5)."""
import json
import urllib.request
import time

URL = 'http://127.0.0.1:8000/v1/chat/completions'
HDRS = {'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'}

TESTS = [
    ("capital", "What is the capital of France? Reply with one word.", 136, 0.0),
    ("math23", "What is 2 + 3? Reply with the number only.", 128, 0.0),
    ("hello", "Say 'hello' and nothing else.", 128, 0.2),
    ("sky", "What color is the sky on a clear day? Reply with one word.", 128, 0.0),
    ("count", "Count from 1 to 5, separated by commas. Reply with the numbers only.", 144, 0.2),
    ("germany", "What is the capital of Germany? Reply with one word.", 136, 0.0),
    ("planets", "List the first three planets from the Sun, comma-separated.", 160, 0.0),
    ("cats", "Write one short sentence about cats.", 160, 0.2),
    ("sum78", "What is 7 * 8? Reply with the number only.", 128, 0.0),
    ("fr", "Translate the word cat to French. Reply with one word.", 128, 0.0),
    ("name", "My name is Alice. What is my name? One word.", 132, 0.0),
    ("plus", "1 + 1 = ? Number only.", 128, 0.0),
]

for name, prompt, max_tok, temp in TESTS:
    payload = {
        "model": "MiniCPM5-2B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tok,
        "temperature": temp,
        "top_p": 0.9,
    }
    t0 = time.time()
    try:
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDRS)
        resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
        dt = time.time() - t0
        ch = resp['choices'][0]
        msg = ch['message']
        text = msg.get('content') or ''
        reasoning = msg.get('reasoning_content') or msg.get('reasoning') or ''
        extra = f" reasoning={reasoning[:40]!r}" if reasoning else ""
        print(f"[{name}] {dt:.1f}s finish={ch.get('finish_reason')!r} -> {text!r}{extra}", flush=True)
    except Exception as e:
        print(f"[{name}] ERROR: {e}", flush=True)
print('BATTERY DONE')

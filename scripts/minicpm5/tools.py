#!/usr/bin/env python3
"""Tool-calling test for MiniCPM5-2B (agentic workload). Run inside container."""
import json
import urllib.request
import time

URL = 'http://127.0.0.1:8000/v1/chat/completions'
HDRS = {'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'}

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]

print("=== TEST 1: direct request with tool ===")
payload = {
    "model": "MiniCPM5-2B",
    "messages": [{"role": "user", "content": "What is the weather in Bangkok right now?"}],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 300,
    "temperature": 0.0,
}
t0 = time.time()
req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDRS)
resp = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
print(f"({time.time()-t0:.1f}s)")
ch = resp['choices'][0]
msg = ch['message']
print("finish:", ch.get('finish_reason'))
print("content:", repr((msg.get('content') or '')[:300]))
tc = msg.get('tool_calls')
print("tool_calls:", json.dumps(tc, indent=2)[:600] if tc else None)

if tc:
    print()
    print("=== TEST 2: multi-turn — feed tool result back ===")
    messages = payload["messages"] + [
        {"role": "assistant", "tool_calls": tc},
        {"role": "tool", "tool_call_id": tc[0].get('id') or 'call_1',
         "content": json.dumps({"city": "Bangkok", "temp_c": 33, "condition": "sunny"})},
    ]
    payload2 = dict(payload, messages=messages, max_tokens=200)
    t0 = time.time()
    req2 = urllib.request.Request(URL, data=json.dumps(payload2).encode(), headers=HDRS)
    resp2 = json.loads(urllib.request.urlopen(req2, timeout=600).read().decode())
    ch2 = resp2['choices'][0]
    msg2 = ch2['message']
    print(f"({time.time()-t0:.1f}s) finish:", ch2.get('finish_reason'))
    print("final answer:", repr((msg2.get('content') or '')[:400]))

print()
print("=== TEST 3: second tool available, pick correct one ===")
tools3 = tools + [{
    "type": "function",
    "function": {
        "name": "search_flights",
        "description": "Search available flights between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {"type": "string"},
                "to_city": {"type": "string"},
                "date": {"type": "string"},
            },
            "required": ["from_city", "to_city"],
        },
    },
}]
payload3 = {
    "model": "MiniCPM5-2B",
    "messages": [{"role": "user", "content": "Find me a flight from Bangkok to Paris on 2026-10-01."}],
    "tools": tools3,
    "tool_choice": "auto",
    "max_tokens": 300,
    "temperature": 0.0,
}
t0 = time.time()
req3 = urllib.request.Request(URL, data=json.dumps(payload3).encode(), headers=HDRS)
resp3 = json.loads(urllib.request.urlopen(req3, timeout=600).read().decode())
ch3 = resp3['choices'][0]
msg3 = ch3['message']
print(f"({time.time()-t0:.1f}s) finish:", ch3.get('finish_reason'))
print("content:", repr((msg3.get('content') or '')[:200]))
tc3 = msg3.get('tool_calls')
print("tool_calls:", json.dumps(tc3, indent=2)[:600] if tc3 else None)
print('TOOLS DONE')

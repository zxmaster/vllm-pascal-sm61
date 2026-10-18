#!/usr/bin/env python3
"""A/B-прогон локального бэкенда: двухшаговый агентский сценарий (tool_call → синтез).

Использование: python3 ab-local-model.py [model] [label]
"""
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8800/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "MiniCPM5-2B"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
TEMP = float(os.environ.get("TEMP", "0.3"))
TOP_P = float(os.environ.get("TOP_P", "1.0"))

TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Поиск в интернете",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "поисковый запрос"}},
                       "required": ["query"]},
    },
}]

SYSTEM = ("Ты — ассистентка. Отвечай по-русски, коротко. Для поиска информации используй web_search.\n"
          "Сегодня 28 сентября 2026, Паттайя, Таиланд.")

WEATHER_RESULT = ("Результаты поиска по запросу «погода Паттайя сегодня»:\n"
                  "Паттайя, 28 сентября 2026: +30 °C, ощущается как +35 °C, влажность 89%, "
                  "кратковременные дожди во второй половине дня, ветер 3 м/с.")


def chat(messages, tools=True):
    payload = {"model": MODEL, "messages": messages, "max_tokens": 900, "temperature": TEMP, "top_p": TOP_P}
    if tools:
        payload["tools"] = TOOLS
    req = urllib.request.Request(BASE, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local-noauth"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["choices"][0]["message"], d.get("usage", {}), dt


def show(tag, m):
    print(f"  [{tag}] content  : {m.get('content')!r}")
    if m.get("tool_calls"):
        for tc in m["tool_calls"]:
            print(f"  [{tag}] tool_call: {tc['function']['name']} args={tc['function']['arguments']}")
    if m.get("reasoning_content"):
        print(f"  [{tag}] reasoning: {m['reasoning_content'][:120]!r}")


print(f"### {LABEL} — модель {MODEL}")
msgs = [{"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Какая сегодня погода в Паттайе? Найди в интернете и ответь одним предложением по-русски."}]

m1, u1, dt1 = chat(msgs)
show("шаг1", m1)
print(f"  шаг1: {dt1:.1f}s, токенов {u1.get('total_tokens')}")

msgs.append(m1)
msgs.append({"role": "tool", "tool_call_id": (m1.get("tool_calls") or [{}])[0].get("id", "call_x"),
             "content": WEATHER_RESULT})
m2, u2, dt2 = chat(msgs)
show("шаг2", m2)
print(f"  шаг2: {dt2:.1f}s, токенов {u2.get('total_tokens')}")
print(f"  ИТОГ ({LABEL}): {m2.get('content')!r}")

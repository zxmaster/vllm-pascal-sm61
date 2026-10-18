#!/usr/bin/env python3
"""
test_baseline.py — базлайн-тесты vLLM Pascal: короткие промпты, без перегрева.

Каждый тест: маленький max_tokens (2-3 предложения), пауза между запросами.
Результат: JSON + краткий вывод. Использование:
  python3 scripts/test_baseline.py [--base http://172.18.100.6:8000] [--out logs/tests/baseline.json]
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

def api(base, path, payload=None, timeout=600, api_key="EMPTY"):
    url = base + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    })
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0

def chat(base, messages, max_tokens=40, extra=None):
    payload = {"model": "Qwen3.5-4B-GGUF", "messages": messages, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False}}
    if extra:
        payload.update(extra)
    body, dt = api(base, "/v1/chat/completions", payload)
    msg = body["choices"][0]["message"]
    return {
        "reply": msg.get("content", "")[:400],
        "finish": body["choices"][0].get("finish_reason"),
        "latency_s": round(dt, 2),
        "tokens": body.get("usage", {}),
    }

def completion_logprobs(base, prompt, max_tokens=1, top=10):
    payload = {
        "model": "Qwen3.5-4B-GGUF", "prompt": prompt, "max_tokens": max_tokens,
        "logprobs": top, "echo": False,
    }
    body, dt = api(base, "/v1/completions", payload)
    lp = body["choices"][0].get("logprobs") or {}
    top_toks = (lp.get("top_logprobs") or [{}])[0] if lp.get("top_logprobs") else {}
    return {"top": top_toks, "latency_s": round(dt, 2)}, body["choices"][0].get("text", "")[:100]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://172.18.100.6:8000")
    ap.add_argument("--out", default="")
    ap.add_argument("--pause", type=float, default=8.0, help="пауза между тестами, сек")
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--max-tokens", type=int, default=30, help="max_tokens для chat-тестов")
    args = ap.parse_args()

    results = {"base": args.base, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tests": {}}
    def record(name, ok, detail):
        results["tests"][name] = {"ok": ok, **detail}
        print(f"[{'OK ' if ok else 'FAIL'}] {name}: {json.dumps(detail, ensure_ascii=False)[:300]}")

    # 0. модели
    try:
        body, dt = api(args.base, "/v1/models")
        record("models", True, {"count": len(body.get("data", [])), "latency_s": round(dt, 2)})
    except Exception as e:
        record("models", False, {"error": str(e)[:200]}); sys.exit(1)
    time.sleep(args.pause)

    # 1. короткий чат
    try:
        r = chat(args.base, [{"role": "user", "content": "What is the capital of France? Answer in one short sentence."}], args.max_tokens)
        record("chat_short", r["finish"] == "stop" and bool(r["reply"]), r)
    except Exception as e:
        record("chat_short", False, {"error": str(e)[:200]})
    time.sleep(args.pause)

    # 2. два предложения
    try:
        r = chat(args.base, [{"role": "user", "content": "Explain quantum entanglement in exactly two sentences."}], args.max_tokens)
        record("chat_two_sentences", r["finish"] == "stop" and bool(r["reply"]), r)
    except Exception as e:
        record("chat_two_sentences", False, {"error": str(e)[:200]})
    time.sleep(args.pause)

    # 3. качество: логпроб Paris (эталон из MEMORIES: logit ~18.05)
    try:
        lp, text = completion_logprobs(args.base, "The capital of France is", top=10)
        record("logprob_paris", bool(lp["top"]), {"top_tokens": {k: round(v, 4) for k, v in sorted(lp["top"].items(), key=lambda x: -x[1])[:5]}, "latency_s": lp["latency_s"]})
    except Exception as e:
        record("logprob_paris", False, {"error": str(e)[:200]})
    time.sleep(args.pause)

    # 4. длинный контекст ~4k токенов (TQ long-context проверка)
    try:
        filler = "The quick brown fox jumps over the lazy dog. " * 350  # ~4.9k токенов
        r = chat(args.base, [{"role": "user", "content": filler + "\nNow answer briefly: what animal jumps?"}], args.max_tokens)
        record("long_ctx_5k", r["finish"] == "stop" and bool(r["reply"]), {"latency_s": r["latency_s"], "reply": r["reply"][:120], "tokens": r["tokens"]})
    except Exception as e:
        record("long_ctx_5k", False, {"error": str(e)[:200]})
    time.sleep(args.pause)

    # 5. tool call (если включён парсер)
    try:
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        payload = {"model": "Qwen3.5-4B-GGUF", "messages": [{"role": "user", "content": "What is the weather in Bangkok? Use the tool."}], "max_tokens": args.max_tokens, "tools": tools,
                   "chat_template_kwargs": {"enable_thinking": False}}
        body, dt = api(args.base, "/v1/chat/completions", payload)
        msg = body["choices"][0]["message"]
        record("tool_call", bool(msg.get("tool_calls")), {"tool_calls": msg.get("tool_calls"), "latency_s": round(dt, 2)})
    except Exception as e:
        record("tool_call", False, {"error": str(e)[:200]})

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.out}")
    ok_count = sum(1 for t in results["tests"].values() if t["ok"])
    print(f"\nSummary: {ok_count}/{len(results['tests'])} tests OK")

if __name__ == "__main__":
    main()

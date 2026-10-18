#!/usr/bin/env python3
"""
Save llama.cpp reference logits for comparison with vLLM.
"""
from llama_cpp import Llama
import numpy as np

MODEL_PATH = "./volumes/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"

print("Loading model...", flush=True)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    verbose=False,
    logits_all=True,
)

prompt = "The capital of France is"
tokens = llm.tokenize(prompt.encode(), add_bos=True)
print(f"Tokens: {tokens}", flush=True)

llm.reset()
llm.eval(tokens)

# Save logits for ALL token positions
all_logits = {}
for i in range(len(tokens)):
    logits = np.array(llm.scores[i], dtype=np.float32)
    all_logits[f'pos_{i}'] = logits
    top5_idx = np.argsort(logits)[::-1][:5]
    top5_str = [(int(idx), float(logits[idx]), llm.detokenize([int(idx)]).decode('utf-8', errors='replace')) for idx in top5_idx]
    print(f"Position {i} top5: {top5_str}")

np.savez('/tmp/llama_ref_logits.npz', **all_logits)
print(f"\nSaved logits to /tmp/llama_ref_logits.npz", flush=True)
print(f"Vocab size: {len(all_logits['pos_0'])}", flush=True)

# Also save per-token hidden state statistics for comparison
# (llama-cpp-python doesn't expose hidden states, but logits are enough)

# Run additional prompts for comparison
extra_prompts = [
    "1+1=",
    "A B C D E F G H I J K",
    "The color of the sky is",
]
for p in extra_prompts:
    toks = llm.tokenize(p.encode(), add_bos=True)
    llm.reset()
    llm.eval(toks)
    last_logits = np.array(llm.scores[len(toks)-1], dtype=np.float32)
    top5_idx = np.argsort(last_logits)[::-1][:5]
    print(f"\nPrompt: {p!r}")
    for idx in top5_idx:
        tok_str = llm.detokenize([int(idx)]).decode('utf-8', errors='replace')
        print(f"  token={idx:6d} logit={last_logits[idx]:+10.4f} text={tok_str!r}")

    output = llm.create_completion(p, max_tokens=10, temperature=0)
    print(f"  Output: {output['choices'][0]['text']!r}")

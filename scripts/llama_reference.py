#!/usr/bin/env python3
"""
Get reference logits from llama-cpp-python for the same GGUF model.
Uses CPU inference to get a gold standard reference.
"""
import sys

from llama_cpp import Llama
import numpy as np

MODEL_PATH = "./volumes/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"

print("Loading model (CPU)...", flush=True)
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=4,
    verbose=False,
    logits_all=True,
)

prompt = "The capital of France is"
print(f"Prompt: {prompt!r}", flush=True)

# Tokenize
tokens = llm.tokenize(prompt.encode(), add_bos=True)
print(f"Tokens: {tokens}", flush=True)

# Evaluate to get logits
llm.reset()
llm.eval(tokens)

# Get logits for the last token position (predicting next token)
logits = llm.scores[len(tokens) - 1]
logits_np = np.array(logits)

# Find top-20 predictions
top_indices = np.argsort(logits_np)[::-1][:20]
print("\nTop-20 predictions for next token after 'The capital of France is':")
for idx in top_indices:
    token_str = llm.detokenize([idx]).decode('utf-8', errors='replace')
    print(f"  token={idx:6d} logit={logits_np[idx]:+10.4f} text={token_str!r}")

# Check specific tokens
paris_tokens = []
for t_str in [" Paris", "Paris", " paris"]:
    toks = llm.tokenize(t_str.encode(), add_bos=False)
    if toks:
        paris_tokens.append((t_str, toks[0]))

print("\nSpecific token logits:")
for name, tid in paris_tokens:
    print(f"  {name!r} (id={tid}): logit={logits_np[tid]:+10.4f}")

# Also check articles
for t_str in [" a", " the", " an"]:
    toks = llm.tokenize(t_str.encode(), add_bos=False)
    if toks:
        tid = toks[0]
        print(f"  {t_str!r} (id={tid}): logit={logits_np[tid]:+10.4f}")

# Generate completion
print("\nGenerated completion:", flush=True)
output = llm.create_completion(
    prompt,
    max_tokens=20,
    temperature=0,
)
print(f"  OUTPUT: {output['choices'][0]['text']!r}", flush=True)

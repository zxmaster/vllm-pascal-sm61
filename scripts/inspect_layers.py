"""Check Qwen3.5 layer structure - which layers are full attention vs GDN."""
import os, re

qwen_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
with open(qwen_path) as f:
    text = f.read()

# Find layer types
for pattern in [
    r"class Qwen3.*Layer",
    r"class Qwen3.*Attention",
    r"class Qwen3.*Model",
    r"linear_attn_layer_indices",
    r"full_attn_layer_indices",
    r"attn_layer_indices",
    r"_LAYER_TYPE",
    r"layer_type",
    r"is_gdn\b",
    r"is_full_attn",
    r"Qwen3NextAttention",
    r"Qwen3_5DecoderLayer",
]:
    for m in re.finditer(pattern, text):
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 200)
        # Find line number
        line_num = text[:m.start()].count("\n") + 1
        snippet = text[m.start():m.end()].split("\n")[0]
        print(f"L{line_num}: {snippet[:150]}")

print("\n--- HF config layer info ---")
# Check the actual hf config stored
config_paths = [
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/",
]
for cp in config_paths:
    if os.path.isdir(cp):
        for root, dirs, files in os.walk(cp):
            for fn in files:
                if fn == "config.json":
                    full = os.path.join(root, fn)
                    with open(full) as f:
                        import json
                        cfg = json.load(f)
                    # Print relevant keys
                    for k in sorted(cfg.keys()):
                        if any(s in k.lower() for s in ["layer", "head", "hidden", "attn", "gdn", "linear"]):
                            print(f"  {k}: {cfg[k]}")
                    break

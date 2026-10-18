"""Inspect Qwen3.5 model structure for TurboQuant integration points."""
import os, sys

# Read the flash_attn backend to understand forward signature
flash_path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flash_attn.py"
if os.path.isfile(flash_path):
    with open(flash_path) as f:
        lines = f.readlines()
    # Find key method signatures
    for i, line in enumerate(lines):
        if "def forward" in line or "class " in line or "def do_kv_cache" in line:
            print(f"flash_attn.py:{i+1}: {line.rstrip()}")

# Read the GDN attention backend
gdn_path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/gdn_attn.py"
if os.path.isfile(gdn_path):
    with open(gdn_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "def forward" in line or "class " in line or "def do_kv_cache" in line:
            print(f"gdn_attn.py:{i+1}: {line.rstrip()}")

# Read the Attention layer wrapper
attn_layer_paths = [
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backend.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py",
]
for p in attn_layer_paths:
    if os.path.isfile(p):
        with open(p) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "def forward" in line or "class " in line or "def do_kv_cache" in line:
                print(f"{os.path.basename(p)}:{i+1}: {line.rstrip()}")

# Read the Qwen3.5 model to find self_attn layers
qwen_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
if os.path.isfile(qwen_path):
    with open(qwen_path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "self_attn" in line and "class " not in line:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                print(f"qwen3_5.py:{i+1}: {stripped[:120]}")

# Check the start-vllm.sh Pascal SDPA patches
start_path = "/workspace/scripts/start-vllm.sh"
if os.path.isfile(start_path):
    with open(start_path) as f:
        text = f.read()
    if "sdpa" in text.lower() or "scaled_dot_product" in text.lower():
        print("start-vllm.sh mentions SDPA/scaled_dot_product")
    if "pascal" in text.lower() or "sm_61" in text.lower() or "sm61" in text.lower():
        print("start-vllm.sh mentions Pascal/SM61")

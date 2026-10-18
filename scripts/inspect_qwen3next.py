"""Read Qwen3NextAttention from qwen3_next.py."""
import os

path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"
with open(path) as f:
    lines = f.readlines()

# Find Qwen3NextAttention class
in_class = False
brace_depth = 0
class_lines = []
for i, line in enumerate(lines):
    if "class Qwen3NextAttention" in line:
        in_class = True
    if in_class:
        class_lines.append(f"{i+1}: {line.rstrip()}")
        if len(class_lines) > 100:
            break

for cl in class_lines:
    print(cl)

print("\n\n--- config.layer_types ---")
# Search for layer_types in config
for i, line in enumerate(lines):
    if "layer_type" in line:
        print(f"{i+1}: {line.rstrip()}")

# Also check the Attention class forward signature
attn_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py"
with open(attn_path) as f:
    attn_lines = f.readlines()

print("\n\n--- Attention.forward method ---")
in_forward = False
forward_lines = []
for i, line in enumerate(attn_lines):
    if "def forward" in line and i > 350:
        in_forward = True
    if in_forward:
        forward_lines.append(f"{i+1}: {line.rstrip()}")
        if len(forward_lines) > 60:
            break

for fl in forward_lines:
    print(fl)

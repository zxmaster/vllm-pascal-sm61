"""Read specific lines from qwen3_5.py to understand layer structure."""
import os

qwen_path = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py"
with open(qwen_path) as f:
    lines = f.readlines()

# Print lines 220-270 (around Qwen3_5DecoderLayer)
for i in range(219, min(280, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")

print("\n--- Lines 80-110 (Qwen3NextAttention) ---")
for i in range(79, min(115, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")

print("\n--- Lines 335-360 (Qwen3_5Model) ---")
for i in range(334, min(365, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")

"""Find the load_model method in gpu_model_runner.py."""
import os

path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
with open(path) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def load_model" in line or "def _load_model" in line:
        # Print context around it
        start = max(0, i - 2)
        end = min(len(lines), i + 30)
        for j in range(start, end):
            print(f"{j+1}: {lines[j].rstrip()}")
        print("---")

# Also find where model is loaded and initialized
for i, line in enumerate(lines):
    if "model.load_weights" in line or "self.model =" in line or "self.model=" in line:
        start = max(0, i - 2)
        end = min(len(lines), i + 5)
        for j in range(start, end):
            print(f"{j+1}: {lines[j].rstrip()}")
        print("...")

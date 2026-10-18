"""Read lines around model load completion in gpu_model_runner."""
path = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py"
with open(path) as f:
    lines = f.readlines()

# Show lines 4530-4560 to see what happens after model load
for i in range(4529, min(4570, len(lines))):
    print(f"{i+1}: {lines[i].rstrip()}")

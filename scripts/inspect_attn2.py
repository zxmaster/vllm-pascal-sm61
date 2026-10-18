"""Inspect vllm module structure for attention backend paths."""
import importlib
import pkgutil

# Find attention-related modules
try:
    import vllm
    print("vllm version:", getattr(vllm, "__version__", "unknown"))

    # Check v1 vs legacy
    try:
        from vllm.v1.attention.backends import torch_sdpa
        print("v1 torch_sdpa backend: FOUND")
        print("  attrs:", [a for a in dir(torch_sdpa) if not a.startswith("__")])
    except ImportError as e:
        print(f"v1 torch_sdpa: {e}")

    try:
        from vllm.v1.attention.backends import flash_attn
        print("v1 flash_attn: FOUND")
    except ImportError as e:
        print(f"v1 flash_attn: {e}")

    # Check what's in v1/attention
    try:
        import vllm.v1.attention as v1attn
        print("v1.attention dir:", [a for a in dir(v1attn) if not a.startswith("__")])
    except ImportError as e:
        print(f"v1.attention: {e}")

    # Check model_executor attention
    try:
        import vllm.model_executor.layers.attention as mela
        print("model_executor.layers.attention dir:", [a for a in dir(mela) if not a.startswith("__")])
    except ImportError as e:
        print(f"model_executor.layers.attention: {e}")

    # List attention backends
    import os
    attn_paths = [
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/",
        "/usr/local/lib/python3.12/dist-packages/vllm/attention/backends/",
    ]
    for p in attn_paths:
        if os.path.isdir(p):
            files = os.listdir(p)
            print(f"  {p}: {files}")
        else:
            print(f"  {p}: NOT FOUND")

except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()

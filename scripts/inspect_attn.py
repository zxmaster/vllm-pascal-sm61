"""Inspect the running model's attention layers for TurboQuant hook points."""
import json

try:
    from vllm.attention import Attention
    print("Attention imported OK")
    print("Attention attrs:", [a for a in dir(Attention) if not a.startswith("__")])
except Exception as e:
    print(f"Attention import failed: {e}")

try:
    from vllm.attention.backends.torch_sdpa import TorchSDPABackendImpl
    print("TorchSDPABackendImpl imported OK")
    print("TorchSDPA attrs:", [a for a in dir(TorchSDPABackendImpl) if not a.startswith("__")])
except Exception as e:
    print(f"TorchSDPA import failed: {e}")

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
    print("Qwen3_5ForCausalLM imported OK")
except Exception as e:
    print(f"Qwen3.5 import failed: {e}")

# Check what layers exist in the model config
try:
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    d = config.to_dict()
    keys_of_interest = [
        "num_hidden_layers", "hidden_size", "num_attention_heads",
        "num_key_value_heads", "head_dim",
        "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim",
        "gdn_layer_indices", "full_attn_layer_indices",
    ]
    for k in keys_of_interest:
        print(f"  {k}: {d.get(k, 'NOT SET')}")
except Exception as e:
    print(f"Config load failed: {e}")

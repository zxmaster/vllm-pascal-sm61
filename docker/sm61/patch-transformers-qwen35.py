from pathlib import Path


def replace_once(text: str, old: str, new: str, error: str) -> str:
    if old in text:
        return text.replace(old, new, 1)
    if new in text:
        return text
    raise SystemExit(error)


loader_path = Path(
    "/usr/local/lib/python3.12/dist-packages/transformers/"
    "modeling_gguf_pytorch_utils.py"
)
loader_text = loader_path.read_text()
loader_marker = (
    '    elif architecture == "qwen35":\n'
    '        updated_architecture = "qwen3_5"\n'
)
if loader_marker not in loader_text:
    loader_anchor = (
        '    # For stablelm architecture, we need to set qkv_bias and '
        'use_parallel_residual from tensors\n'
    )
    loader_text = replace_once(
        loader_text,
        loader_anchor,
        loader_marker + "\n" + loader_anchor,
        "Unable to patch GGUF loader for qwen35 alias",
    )
    loader_path.write_text(loader_text)


ggml_path = Path(
    "/usr/local/lib/python3.12/dist-packages/transformers/integrations/ggml.py"
)
ggml_text = ggml_path.read_text()
mapping_marker = '    "qwen3_5": {\n'
if mapping_marker not in ggml_text:
    mapping_anchor = '''    "qwen3": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
'''
    mapping_insertion = '''    "qwen3": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "qwen3_5": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.key_length": "linear_key_head_dim",
        "attention.value_length": "linear_value_head_dim",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "ssm.conv_kernel": "linear_conv_kernel_dim",
        "ssm.group_count": "linear_num_key_heads",
        "full_attention_interval": "full_attention_interval",
        "vocab_size": "vocab_size",
    },
'''
    ggml_text = replace_once(
        ggml_text,
        mapping_anchor,
        mapping_insertion,
        "Unable to patch GGUF config mapping for qwen3_5",
    )

converter_marker = '    "qwen3_5": GGUFQwen2Converter,\n'
if converter_marker not in ggml_text:
    converter_anchor = '''    "qwen3": GGUFQwen2Converter,
    "qwen3_moe": GGUFQwen2Converter,
'''
    converter_insertion = '''    "qwen3": GGUFQwen2Converter,
    "qwen3_5": GGUFQwen2Converter,
    "qwen3_moe": GGUFQwen2Converter,
'''
    ggml_text = replace_once(
        ggml_text,
        converter_anchor,
        converter_insertion,
        "Unable to patch GGUF converter registry for qwen3_5",
    )

ggml_path.write_text(ggml_text)

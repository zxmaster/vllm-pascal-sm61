#!/usr/bin/env python3
"""Build an HF config + tokenizer bundle for Bonsai-27B (Q1_0 GGUF) for vLLM.

Bonsai is a qwen35 hybrid (GDN + attention) dense 27B model. vLLM's GGUF plugin
needs an HF config next to / pointed at the model; we derive every value from
the GGUF metadata instead of guessing.

Usage:
    python3 scripts/make-bonsai-config.py \
        --gguf volumes/models/Bonsai-27B-Q1_0.gguf \
        --tokenizer-src volumes/hf-cache/Qwen3.5-4B-hf \
        --out volumes/models/Bonsai-27B-GGUF
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import gguf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer-src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = gguf.GGUFReader(args.gguf)
    f = r.fields

    def g(key: str):
        return f[key].contents() if key in f else None

    n_layers = int(g("qwen35.block_count"))
    hidden = int(g("qwen35.embedding_length"))
    ffn = int(g("qwen35.feed_forward_length"))
    n_heads = int(g("qwen35.attention.head_count"))
    n_kv = int(g("qwen35.attention.head_count_kv"))
    head_dim = int(g("qwen35.attention.key_length"))
    interval = int(g("qwen35.full_attention_interval"))
    ctx = int(g("qwen35.context_length"))
    rope_base = float(g("qwen35.rope.freq_base"))

    ssm_groups = int(g("qwen35.ssm.group_count"))
    ssm_rank = int(g("qwen35.ssm.time_step_rank"))
    ssm_inner = int(g("qwen35.ssm.inner_size"))
    ssm_state = int(g("qwen35.ssm.state_size"))
    ssm_conv = int(g("qwen35.ssm.conv_kernel"))

    vocab = None
    for key in ("qwen35.vocab_size", "general.vocab_size"):
        if g(key):
            vocab = int(g(key))
            break
    if vocab is None:
        vocab = int(len(f["tokenizer.ggml.tokens"].contents()))

    layer_types = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(n_layers)
    ]

    text_config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "dtype": "float16",
        "full_attention_interval": interval,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "hidden_size": hidden,
        "initializer_range": 0.02,
        "intermediate_size": ffn,
        "layer_types": layer_types,
        "linear_conv_kernel_dim": ssm_conv,
        "linear_key_head_dim": ssm_state,
        "linear_num_key_heads": ssm_groups,
        "linear_num_value_heads": ssm_rank,
        "linear_value_head_dim": ssm_inner // ssm_rank,
        "max_position_embeddings": ctx,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "num_attention_heads": n_heads,
        "num_hidden_layers": n_layers,
        "num_key_value_heads": n_kv,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_scaling": None,
        "rope_theta": rope_base,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": vocab,
    }

    config = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5",
        "text_config": text_config,
    }
    # eos token from GGUF metadata when available
    eos = g("tokenizer.ggml.eos_token_id")
    if eos is not None:
        config["eos_token_id"] = int(eos)
        text_config["eos_token_id"] = int(eos)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    src = Path(args.tokenizer_src)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "generation_config.json",
    ):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    print(f"wrote {out}/config.json")
    print(
        f"layers={n_layers} hidden={hidden} heads={n_heads}/{n_kv} head_dim={head_dim} "
        f"ffn={ffn} interval={interval} vocab={vocab} ctx={ctx}"
    )
    print(
        f"ssm: groups={ssm_groups} rank={ssm_rank} inner={ssm_inner} "
        f"state={ssm_state} conv={ssm_conv} -> value_head_dim={ssm_inner // ssm_rank}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
pascal_runtime_patch.py — runtime patches for vLLM v0.28.0 on Pascal (sm_61).

Applied at container startup by start-vllm-v028.sh. Idempotent (safe to run
multiple times). Two patches:

1. GDN torch fallback (VLLM_PASCAL_TORCH_GDN=1):
   - rebind fused_post_conv_prep / fused_sigmoid_gating_delta_rule_update
     in qwen_gdn_linear_attn.py to pascal_gdn_torch implementations
   - add ChunkGatedDeltaRule.forward_torch and force it when the env is set

2. Triton attention SDPA fallback (always, when triton_attn.py is used):
   - replace the unified_attention Triton call with a chunked PyTorch SDPA
     implementation (Triton tl.dot is broken on sm_61, see MEMORIES.md §6/§13)
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path

SITE = Path("/usr/local/lib/python3.12/dist-packages")
GDN = SITE / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
TRITON_ATTN = SITE / "vllm/v1/attention/backends/triton_attn.py"
GGUF_ADAPTER = SITE / "vllm_gguf_plugin/weights_adapter/default.py"
SCRIPTS = "/workspace/scripts"


def patch_gdn_torch() -> bool:
    if os.environ.get("VLLM_PASCAL_TORCH_GDN", "0") != "1":
        return False
    text = GDN.read_text()
    changed = False

    # --- 1. rebind the two module-level Triton functions -------------------
    anchor = "logger = init_logger(__name__)\n"
    if anchor in text and "pascal_gdn_torch" not in text:
        inject = anchor + textwrap.dedent(f'''\
            if os.environ.get("VLLM_PASCAL_TORCH_GDN", "0") == "1":
                import sys
                sys.path.insert(0, "{SCRIPTS}")
                import pascal_gdn_torch as _pgt
                fused_post_conv_prep = _pgt.fused_post_conv_prep
                fused_sigmoid_gating_delta_rule_update = _pgt.fused_sigmoid_gating_delta_rule_update
                # v0.28 causal_conv1d is a Triton kernel that silently returns
                # ZEROS on sm_61 -> replace with the PyTorch implementation.
                causal_conv1d_fn = _pgt.pascal_causal_conv1d
                causal_conv1d_update = _pgt.pascal_causal_conv1d_update
                logger.info_once("[Pascal] GDN torch fallback active (post_conv_prep + sigmoid update + conv1d)")
            ''')
        text = text.replace(anchor, inject, 1)
        changed = True
        print("[Pascal] GDN: fused_post_conv_prep / sigmoid update rebound", flush=True)
    elif "pascal_gdn_torch" in text:
        print("[Pascal] GDN: import rebinding already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate GDN logger anchor")

    # --- 2. add forward_torch to ChunkGatedDeltaRule -----------------------
    fwd_anchor = "    def forward_cuda(\n"
    if fwd_anchor in text and "def forward_torch" not in text:
        method = textwrap.indent(GDN_FORWARD_TORCH, "    ")
        text = text.replace(fwd_anchor, method + fwd_anchor, 1)
        changed = True
        print("[Pascal] GDN: forward_torch added", flush=True)

    # --- 3. force forward_torch selection ----------------------------------
    sel_old = "        else:\n            self._forward_method = self.forward_native\n"
    sel_new = textwrap.indent(GDN_SELECTION, "        ")
    if sel_old in text:
        text = text.replace(sel_old, sel_new, 1)
        changed = True
        print("[Pascal] GDN: forward_torch selected (env-gated)", flush=True)
    elif "self._forward_method = self.forward_torch" in text:
        print("[Pascal] GDN: forward_torch selection already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate forward_native selection")

    if changed:
        GDN.write_text(text)
    return True


GDN_FORWARD_TORCH = textwrap.dedent('''\
def forward_torch(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    core_attn_out: torch.Tensor | None = None,
):
    """Pure-PyTorch chunked delta rule (Pascal)."""
    import os as _os
    o, st = _pgt.PascalChunkGatedDeltaRule().forward(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        core_attn_out=core_attn_out,
    )
    if _os.environ.get("PASCAL_GDN_DEBUG") == "1":
        print(f"[PGDN-OUT] out_abs={float(o.float().abs().mean()):.4f} "
              f"q_abs={float(q.float().abs().mean()):.4f} "
              f"v_abs={float(v.float().abs().mean()):.4f} "
              f"g_min={float(g.float().min()):.3f} "
              f"beta_mean={float(beta.float().mean()):.4f} "
              f"v0={[round(float(x),4) for x in v[0,:4,0,0]]} "
              f"g0={[round(float(x),3) for x in g[0,:4,0]]}", flush=True)
    return o, st
''')

GDN_SELECTION = textwrap.dedent('''\
else:
    if os.environ.get("VLLM_PASCAL_TORCH_GDN", "0") == "1":
        self._forward_method = self.forward_torch
    else:
        self._forward_method = self.forward_native
''')

SDPA_BLOCK = textwrap.dedent('''\
    # --- Pascal SDPA fallback (avoids Triton tl.dot, broken on sm_61) ---
    import os
    import torch.nn.functional as _F
    try:
        from torch.nn.attention.bias import causal_lower_right as _causal_lower_right
    except ImportError:
        _causal_lower_right = None

    _q = query[:num_actual_tokens]       # [total_q, num_heads, head_size]
    # v0.28 logical (B, H, N, 2*hs) cache is transposed to physical
    # (B, N, H, hs) before this point: [num_blocks, block_size, nkv, hs].
    # NB: for hybrid models block_size is LARGE (e.g. 528), not 16.
    _block_size = key_cache.shape[1]
    _num_kv_heads = key_cache.shape[2]
    _head_size = _q.shape[2]
    _num_heads = _q.shape[1]
    _gqa_ratio = _num_heads // _num_kv_heads
    _num_seqs = len(seqused_k)
    _stream_threshold = int(os.environ.get("COPILOT_SDPA_STREAM_THRESHOLD", "0"))
    _q_block = max(16, int(os.environ.get("COPILOT_SDPA_Q_BLOCK", "64")))
    _block_sdpa_q_block = max(_q_block, int(os.environ.get("COPILOT_SDPA_BLOCK_SDPA_Q_BLOCK", "768")))
    _kv_block = max(32, int(os.environ.get("COPILOT_SDPA_KV_BLOCK", "128")))

    for _si in range(_num_seqs):
        _qs = cu_seqlens_q[_si].item()
        _qe = cu_seqlens_q[_si + 1].item()
        _kv_len = seqused_k[_si].item()
        _q_len = _qe - _qs
        if _q_len <= 0:
            continue
        if os.environ.get("PASCAL_GDN_DEBUG") == "1":
            print(f"[PASCAL-SDPA] si={_si} qs={_qs} qe={_qe} q_len={_q_len} kv_len={seqused_k[_si].item()} nseqs={_num_seqs} cu_last={cu_seqlens_q[-1].item()} max_seq={attn_metadata.max_seq_len} nh={_num_heads} nkv={_num_kv_heads} actual={attn_metadata.num_actual_tokens} kvc={tuple(kv_cache.shape)} kc={tuple(key_cache.shape)} hs={self.head_size} attn_type={self.attn_type} layer={type(layer).__name__} kstr={kv_cache.stride()} kcstr={key_cache.stride()} bs={_block_size}", flush=True)
        _nblk = (_kv_len + _block_size - 1) // _block_size
        _blks = block_table[_si, :_nblk]
        _k = key_cache[_blks].reshape(-1, _num_kv_heads, _head_size)[:_kv_len]
        _v = value_cache[_blks].reshape(-1, _num_kv_heads, _head_size)[:_kv_len]
        if os.environ.get("PASCAL_GDN_DEBUG") == "1":
            try:
                _pr = str(getattr(layer, "layer_name", "x")).replace(".", "_")
                if _q.shape[0] == 6:
                    import torch as _t5
                    _t5.save(_q.detach().cpu(), f"/logs/vllm_sdpa_q_tok{_q.shape[0]}_{_pr}.pt")
                    _t5.save(_k.detach().cpu(), f"/logs/vllm_sdpa_k_tok{_k.shape[0]}_{_pr}.pt")
                    _t5.save(_v.detach().cpu(), f"/logs/vllm_sdpa_v_tok{_v.shape[0]}_{_pr}.pt")
            except Exception:
                pass
        if os.environ.get("PASCAL_GDN_DEBUG") == "1":
            print(f"[SDPA-IN] si={_si} q_abs={float(_q.float().abs().mean()):.4f} "
                  f"k_abs={float(_k.float().abs().mean()):.4f} v_abs={float(_v.float().abs().mean()):.4f} "
                  f"qlen={_q_len} kvlen={_kv_len}", flush=True)
        _kv_len = min(_kv_len, _k.shape[0], _v.shape[0])
        if _kv_len <= 0:
            output[_qs:_qe].zero_()
            continue
        _k = _k[:_kv_len]
        _v = _v[:_kv_len]

        if _kv_len > _stream_threshold:
            _q_positions = (
                torch.arange(_q_len, device=_q.device) if _q_len == _kv_len else None
            )
            _used_block_sdpa = False

            if os.environ.get("COPILOT_SDPA_BLOCKWISE_SDPA", "1") != "0":
                _fast_q_block = min(_block_sdpa_q_block, max(16, 7_680_000 // max(1, _kv_len)))
                _kb_base = _k.unsqueeze(0).transpose(1, 2)
                _vb_base = _v.unsqueeze(0).transpose(1, 2)
                while _fast_q_block >= _q_block and not _used_block_sdpa:
                    _attempt_q_block = _fast_q_block
                    try:
                        for _q_start in range(0, _q_len, _attempt_q_block):
                            _q_end = min(_q_start + _attempt_q_block, _q_len)
                            _kv_end = _q_end if _q_positions is not None else _kv_len
                            if _kv_end <= 0:
                                output[_qs + _q_start:_qs + _q_end].zero_()
                                continue
                            _qb = _q[_qs + _q_start:_qs + _q_end].unsqueeze(0).transpose(1, 2)
                            _kb = _kb_base[:, :, :_kv_end]
                            _vb = _vb_base[:, :, :_kv_end]
                            if os.environ.get("PASCAL_GDN_DEBUG") == "1":
                                print(f"[PASCAL-SDPA-BLK] qb={tuple(_qb.shape)} kb={tuple(_kb.shape)} vb={tuple(_vb.shape)} qpos={'Y' if _q_positions is not None else 'N'} qstart={_q_start} qend={_q_end} kvend={_kv_end} gqa={_gqa_ratio}", flush=True)
                            _sdpa_kwargs = {"scale": self.scale, "is_causal": False}
                            if _q_positions is not None:
                                _q_pos = _q_positions[_q_start:_q_end]
                                _k_pos = torch.arange(_kv_end, device=_q.device)
                                _sdpa_kwargs["attn_mask"] = _k_pos.unsqueeze(0) <= _q_pos.unsqueeze(1)
                            # NOTE: torch enable_gqa is BROKEN on sm_61 (silent
                            # wrong results, verified numerically: 1.5x divergence
                            # vs manual expansion). Always expand k/v explicitly.
                            if _gqa_ratio > 1:
                                _kb = _kb.repeat_interleave(_gqa_ratio, dim=1)
                                _vb = _vb.repeat_interleave(_gqa_ratio, dim=1)
                            _ao = _F.scaled_dot_product_attention(_qb, _kb, _vb, **_sdpa_kwargs)
                            output[_qs + _q_start:_qs + _q_end] = _ao.to(_q.dtype).squeeze(0).transpose(0, 1)
                        _used_block_sdpa = True
                    except torch.cuda.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if _attempt_q_block <= _q_block:
                            break
                        _fast_q_block = max(_q_block, _attempt_q_block // 2)
                    except RuntimeError as _block_sdpa_err:
                        if "out of memory" in str(_block_sdpa_err).lower():
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if _attempt_q_block <= _q_block:
                                break
                            _fast_q_block = max(_q_block, _attempt_q_block // 2)
                        else:
                            raise

            if _used_block_sdpa:
                continue

            _q_heads = _q[_qs:_qe].transpose(0, 1)
            _k_heads = _k.transpose(0, 1)
            _v_heads = _v.transpose(0, 1)

            for _q_start in range(0, _q_len, _q_block):
                _q_end = min(_q_start + _q_block, _q_len)
                _chunk_q = _q_end - _q_start
                _q_chunk = _q_heads[:, _q_start:_q_end].reshape(_num_kv_heads, _gqa_ratio, _chunk_q, _head_size).float()
                # (K is expanded per _kv_block below; no full-seq materialization)
                # (V is expanded per _kv_block below; no full-seq materialization)
                _m = torch.full((_num_kv_heads, _gqa_ratio, _chunk_q), float("-inf"), device=_q.device)
                _l = torch.zeros_like(_m)
                _acc = torch.zeros((_num_kv_heads, _gqa_ratio, _chunk_q, _head_size), dtype=torch.float32, device=_q.device)
                _has_kv = False

                for _k_start in range(0, _kv_len, _kv_block):
                    _k_end = min(_k_start + _kv_block, _kv_len)
                    _k_chunk = _k_heads[:, _k_start:_k_end].unsqueeze(1).expand(-1, _gqa_ratio, -1, -1).float()
                    _v_chunk = _v_heads[:, _k_start:_k_end].unsqueeze(1).expand(-1, _gqa_ratio, -1, -1).float()
                    if _k_chunk.shape[2] == 0:
                        continue
                    _has_kv = True
                    _scores = torch.einsum("hgqd,hgkd->hgqk", _q_chunk, _k_chunk) * self.scale
                    if _q_positions is not None:
                        _q_pos = _q_positions[_q_start:_q_end]
                        _k_pos = torch.arange(_k_start, _k_end, device=_q.device)
                        _mask = _k_pos.unsqueeze(0) > _q_pos.unsqueeze(1)
                        _scores = _scores.masked_fill(_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
                    _scores_max = _scores.amax(dim=-1)
                    _new_m = torch.maximum(_m, _scores_max)
                    _old_scale = torch.exp(_m - _new_m)
                    _weights = torch.exp(_scores - _new_m.unsqueeze(-1))
                    _acc = _acc * _old_scale.unsqueeze(-1) + torch.einsum("hgqk,hgkd->hgqd", _weights, _v_chunk)
                    _l = _l * _old_scale + _weights.sum(dim=-1)
                    _m = _new_m

                if not _has_kv:
                    output[_qs + _q_start:_qs + _q_end].zero_()
                    continue
                _chunk_out = (_acc / _l.clamp_min(1e-9).unsqueeze(-1)).reshape(_num_heads, _chunk_q, _head_size)
                output[_qs + _q_start:_qs + _q_end] = _chunk_out.transpose(0, 1).to(_q.dtype)
            continue

        _qb = _q[_qs:_qe].unsqueeze(0).transpose(1, 2)
        _kb = _k.unsqueeze(0).transpose(1, 2)
        _vb = _v.unsqueeze(0).transpose(1, 2)
        _sdpa_kwargs = {"scale": self.scale, "is_causal": _q_len == _kv_len}
        # NOTE: torch enable_gqa is BROKEN on sm_61 (silent wrong results).
        # Always expand k/v explicitly.
        if _gqa_ratio > 1:
            _kb = _kb.repeat_interleave(_gqa_ratio, dim=1)
            _vb = _vb.repeat_interleave(_gqa_ratio, dim=1)
        _ao = _F.scaled_dot_product_attention(_qb, _kb, _vb, **_sdpa_kwargs)
        output[_qs:_qe] = _ao.to(_q.dtype).squeeze(0).transpose(0, 1)
        if os.environ.get("PASCAL_GDN_DEBUG") == "1":
            print(f"[SDPA-OUT] si={_si} ao_abs={float(_ao.float().abs().mean()):.4f}", flush=True)
''').strip("\n")


def patch_triton_attn() -> bool:
    text = TRITON_ATTN.read_text()
    if "Pascal SDPA fallback" in text:
        print("[Pascal] SDPA: fallback already applied", flush=True)
        return True

    old = '''        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
        )'''
    if old not in text:
        raise SystemExit("pascal_runtime_patch: unified_attention call anchor not found in triton_attn.py")
    text = text.replace(old, textwrap.indent(SDPA_BLOCK, "        "), 1)
    TRITON_ATTN.write_text(text)
    print("[Pascal] SDPA: Triton unified_attention replaced with chunked SDPA", flush=True)
    return True


def patch_gguf_adapter() -> bool:
    """GGUF-plugin 0.0.5 normalization bug: HF model_type qwen3_5/qwen3_5_moe
    is not in gguf.MODEL_ARCH_NAMES (the gguf package uses qwen35/qwen35moe).
    Mirror the plugin's own deepseek/cohere normalization."""
    text = GGUF_ADAPTER.read_text()
    if "PASCAL-TILED-STRATEGY" in text:
        print("[Pascal] GGUF adapter: already patched (tiled strategy)", flush=True)
        return False
    if "qwen3_5_moe" in text:
        raise SystemExit(
            "pascal_runtime_patch: gguf adapter carries a stale patch variant; "
            "recreate the container from the image "
            "(docker compose up -d --force-recreate), then retry"
        )
    old = '        if model_type == "gemma3_text":\n            model_type = "gemma3"\n'
    new = old + '        if model_type == "qwen3_5":\n            model_type = "qwen35"\n        if model_type == "qwen3_5_moe":\n            model_type = "qwen35moe"\n'
    if old not in text:
        raise SystemExit("pascal_runtime_patch: failed to locate GGUF adapter gemma3 anchor")
    text = text.replace(old, new, 1)
    changed = True
    print("[Pascal] GGUF adapter: qwen3_5 -> qwen35 normalization added", flush=True)

    # v0.28: Qwen3_5Config always builds a default vision_config, so the
    # plugin's config-based is_multimodal fires for TEXT-ONLY models.
    # Trust vllm's multimodal_config instead (None for language-only models).
    mm_old = '        is_multimodal = (\n            hasattr(config, "vision_config") and config.vision_config is not None\n        )\n'
    mm_new = textwrap.dedent('''\
        mm_cfg = getattr(model_config, "multimodal_config", None)
        is_multimodal = mm_cfg is not None and not mm_cfg.language_model_only
''')
    if mm_old in text:
        text = text.replace(mm_old, textwrap.indent(mm_new, "        "), 1)
        changed = True
        print("[Pascal] GGUF adapter: is_multimodal now trusts vllm multimodal_config", flush=True)
    elif "is_multimodal = mm_cfg is not None" in text:
        print("[Pascal] GGUF adapter: is_multimodal patch already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate GGUF adapter is_multimodal anchor")

    # transformers 5.x Qwen3.5: (a) the causal-LM model build MUTATES the
    # shared hf_config (hidden 2560->4096) when given the multimodal
    # Qwen3_5Config; (b) the Qwen3_5Config top level has no vocab_size, so
    # from_config(whole config) fails. Build the dummy from the TEXT config.
    tc_old = "            dummy_model = AutoModelForCausalLM.from_config(\n                config, trust_remote_code=model_config.trust_remote_code\n            )"
    tc_new = "            dummy_model = AutoModelForCausalLM.from_config(\n                copy.deepcopy(config.get_text_config()),\n                trust_remote_code=model_config.trust_remote_code,\n            )"
    if tc_old in text:
        text = text.replace(tc_old, tc_new, 1)
        if "import copy" not in text:
            text = text.replace("import torch", "import copy\nimport torch", 1)
        changed = True
        print("[Pascal] GGUF adapter: dummy from_config(text_config) patched", flush=True)
    elif "from_config(\n                copy.deepcopy(config.get_text_config())" in text:
        print("[Pascal] GGUF adapter: text_config patch already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate GGUF adapter from_config anchor")

    # GGUF file (llama.cpp qwen35) has no linear_attn.dt_bias tensors; the
    # transformers dummy expects them -> unmapped. Skip the strict check for
    # dt_bias only (model keeps its init value; conversion dropped the tensor).
    db_old = "        if unmapped_params:\n            raise RuntimeError(\n                f\"Failed to map GGUF parameters \"\n                f\"({len(unmapped_params)}): {unmapped_params}\"\n            )\n"
    db_new = (
        "        if unmapped_params:\n"
        "            filtered_unmapped = [\n"
        "                x\n"
        "                for x in unmapped_params\n"
        "                if not re.fullmatch(\n"
        "                    r\"model\\.layers\\.\\d+\\.linear_attn\\.dt_bias\", x\n"
        "                )\n"
        "            ]\n"
        "            if filtered_unmapped:\n"
        "                raise RuntimeError(\n"
        "                    f\"Failed to map GGUF parameters \"\n"
        "                    f\"({len(filtered_unmapped)}): {filtered_unmapped}\"\n"
        "                )\n"
    )
    if db_old in text:
        text = text.replace(db_old, db_new, 1)
        changed = True
        print("[Pascal] GGUF adapter: dt_bias unmapped tolerated", flush=True)
    elif "linear_attn\\.dt_bias" in text:
        print("[Pascal] GGUF adapter: dt_bias skip already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate GGUF adapter unmapped anchor")

    # GGUF data for conv1d arrives as [out, kernel] (2D, GGUF row-major
    # layout); the vllm conv param is [out, 1, kernel] -> unsqueeze only.
    cv_old = "        for hf_name, weight in weights:\n            weight = self.transform_weight(hf_name, weight)\n"
    cv_block = (
        "        for hf_name, weight in weights:\n"
        "            import os as _os\n"
        "            if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\" and (\"A_log\" in hf_name or \"dt_bias\" in hf_name):\n"
        "                print(f\"[MAPW-NAME] {hf_name} shape={tuple(weight.shape)}\", flush=True)\n"
        "            weight = self.transform_weight(hf_name, weight)\n"
        "            if hf_name.endswith(\".linear_attn.conv1d.weight\"):\n"
        "                if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "                    print(f\"[MAPW] conv1d in: {tuple(weight.shape)}\", flush=True)\n"
        "                if weight.ndim == 2:\n"
        "                    weight = weight.unsqueeze(1)\n"
        "                if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "                    print(f\"[MAPW] conv1d out: {tuple(weight.shape)}\", flush=True)\n"
        "            if hf_name.endswith(\".linear_attn.A_log\"):\n"
        "                # GGUF stores ssm_a = -exp(A_log); vllm expects raw A_log.\n"
        "                if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "                    print(f\"[MAPW] A_log in: {[round(float(v),4) for v in weight.flatten()[:3]]}\", flush=True)\n"
        "                weight = torch.log(-weight.to(torch.float32)).to(weight.dtype)\n"
        "                if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "                    print(f\"[MAPW] A_log out: {[round(float(v),4) for v in weight.flatten()[:3]]}\", flush=True)\n"
        "            # llama.cpp stores RMSNorm weights as (1 + w) only for SOME\n"
        "            # architectures (qwen3_5 family); llama-family files store\n"
        "            # raw weights. A blind (w - 1) corrupts every norm and turns\n"
        "            # the output into noise (MiniCPM5 regression, 2026-09-15).\n"
        "            _arch = getattr(getattr(self, \"config\", None), \"model_type\", \"\")\n"
        "            if _arch in (\"qwen3_5\", \"qwen3_5_moe\", \"qwen35\", \"qwen35moe\") and (\n"
        "                    hf_name.endswith(\".input_layernorm.weight\")\n"
        "                    or hf_name.endswith(\".post_attention_layernorm.weight\")\n"
        "                    or hf_name.endswith(\"model.norm.weight\")\n"
        "                    or hf_name.endswith(\".q_norm.weight\")\n"
        "                    or hf_name.endswith(\".k_norm.weight\")\n"
        "            ):\n"
        "                weight = weight - 1\n"
        "            # --- PASCAL-LLAMA-QK-PERM (2026-09-15) --------------------\n"
        "            # llama.cpp-family GGUF files store q/k projections in\n"
        "            # the interleaved (NEOX) per-head layout: per 128-wide\n"
        "            # head rows are [0,64,1,65,...]. vLLM applies the\n"
        "            # HF-style (rotate_half) RoPE on top, so the weights\n"
        "            # must be converted back to the HF row order or every\n"
        "            # q/k rotation is wrong (found on MiniCPM5-2B; proven\n"
        "            # by tensor diff vs HF: V matches, q/k mismatched).\n"
        "            if _arch == \"llama\" and weight.ndim == 2 and (\n"
        "                    hf_name.endswith(\".self_attn.q_proj.qweight\")\n"
        "                    or hf_name.endswith(\".self_attn.k_proj.qweight\")):\n"
        "                _od = weight.shape[0]\n"
        "                if _od % 128 == 0 and _od > 0:\n"
        "                    _r = torch.arange(_od, device=weight.device).reshape(-1, 128)\n"
        "                    _sel = torch.cat([_r[:, 0::2], _r[:, 1::2]], dim=1).reshape(-1)\n"
        "                    weight = weight[_sel]\n"
        "                    if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "                        print(f\"[PASCAL-QK-PERM] {hf_name} out={_od} reshaped\", flush=True)\n"
        "            # --- PASCAL-TILED-STRATEGY (2026-09-14) ------------------\n"
        "            # llama.cpp qwen35 stores ALL V-side weights TILED\n"
        "            # ([G0v0..G15v0, G0v1, ..]) and we load them AS-IS.\n"
        "            # The Pascal runtime compensates via the tiled q/k expansion\n"
        "            # in pascal_gdn_torch._expand_qk_to_vheads (v0.18 fix #7/#8).\n"
        "            # NEVER add grouped un-reordering here: quantized qweights\n"
        "            # (Q4_K/Q5_K blocks span 256 values) cannot be permuted\n"
        "            # column-wise, and out_proj would need exactly that.\n"
    )
    if cv_old not in text:
        raise SystemExit("pascal_runtime_patch: failed to locate GGUF adapter map_weights anchor")
    m2 = "if hf_name.endswith(\".linear_attn.conv1d.weight\") and weight.ndim == 2:"
    m3 = "if hf_name.endswith(\".linear_attn.conv1d.weight\"):"
    if m3 in text:
        print("[Pascal] GGUF adapter: conv1d transform already correct", flush=True)
    elif m2 in text:
        # Replace the old single-line condition + body with the full block.
        import re as _re
        text = _re.sub(
            r"            if hf_name\.endswith\(\"\\.linear_attn\\.conv1d\\.weight\"\) and weight\.ndim == 2:\n"
            r"(?:                weight = weight\\.(?:unsqueeze\\(1\\)|transpose\\(0, 1\\)\\.unsqueeze\\(1\\))\n)+",
            cv_block[44:],
            text,
            count=1,
        )
        changed = True
        print("[Pascal] GGUF adapter: conv1d transform upgraded (2D+3D)", flush=True)
    else:
        text = text.replace(cv_old, cv_block, 1)
        changed = True
        print("[Pascal] GGUF adapter: conv1d transpose+unsqueeze transform added", flush=True)

    # The gguf qwen35 name map has NO entry for linear_attn.dt_bias, so the
    # parameter is left at its init value (ones) while the GGUF file DOES
    # contain blk.N.ssm_dt.bias. Wrong dt_bias -> wrong GDN gating (garbage
    # output). Map it explicitly (mirrors the v0.18 working loader).
    dt_old = "        return gguf_to_hf_name_map\n"
    dt_new = (
        "        for _i in range(text_config.num_hidden_layers):\n"
        "            _k = f\"blk.{_i}.ssm_dt.bias\"\n"
        "            if _k not in gguf_to_hf_name_map:\n"
        "                gguf_to_hf_name_map[_k] = (\n"
        "                    f\"model.layers.{_i}.linear_attn.dt_bias\"\n"
        "                )\n"
        "        for _i in range(text_config.num_hidden_layers):\n"
        "            # suffix-less names get a trailing dot in the key; fix it\n"
        "            gguf_to_hf_name_map.pop(f\"blk.{_i}.ssm_a.\", None)\n"
        "            gguf_to_hf_name_map[f\"blk.{_i}.ssm_a\"] = (\n"
        "                f\"model.layers.{_i}.linear_attn.A_log\"\n"
        "            )\n"
        "        return gguf_to_hf_name_map\n"
    )
    if dt_old in text and "ssm_dt.bias" not in text:
        text = text.replace(dt_old, dt_new, 1)
        changed = True
        print("[Pascal] GGUF adapter: ssm_dt.bias -> dt_bias mapping added", flush=True)
    elif "ssm_dt.bias" in text:
        print("[Pascal] GGUF adapter: dt_bias mapping already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate GGUF adapter return anchor")
    GGUF_ADAPTER.write_text(text)
    return changed


def patch_gguf_quant() -> bool:
    """GGUF-plugin 0.0.5 load-time fixes for the sm_61/4GB build:
    1. qweight_type params need output_dim/input_dim (fused loader narrows).
    2. the type loader must not go through vllm's fused weight_loader_v2
       (0-dim narrow crash) - store directly on the param.
    3. _materialize_gguf_weight_parameter keeps the replaced raw parameter
       alive via its bound-method weight_loader; its data_container still
       references all shard tensors -> unbounded VRAM growth during
       process_weights_after_loading (OOM on 4GB). Clear it.
    """
    changed = False
    lin = Path(SITE / "vllm_gguf_plugin/quantization/linear.py")
    text = lin.read_text()
    od_old = '                "num_elements": len(output_partition_sizes),\n                "ignore_warning": True,\n'
    if od_old in text and '"output_dim": 0' not in text:
        text = text.replace(
            od_old,
            '                "num_elements": len(output_partition_sizes),\n'
            '                "input_dim": 0,\n'
            '                "output_dim": 0,\n'
            '                "ignore_warning": True,\n',
            1,
        )
        changed = True
        print("[Pascal] GGUF quant: qweight_type output_dim added (linear)", flush=True)
    elif '"output_dim": 0' in text:
        print("[Pascal] GGUF quant: output_dim already present", flush=True)
    else:
        raise SystemExit("pascal_runtime_patch: failed to locate linear.py qweight_type attrs anchor")
    lin.write_text(text)

    par = Path(SITE / "vllm_gguf_plugin/quantization/params.py")
    text = par.read_text()
    tl_old = '    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):\n        if loaded_shard_id is None and hasattr(param, "_store"):\n            param._store(loaded_weight)\n            return\n        base_loader(param, loaded_weight, loaded_shard_id)\n'
    tl_new = (
        '    def _gguf_weight_type_loader_v2(param, loaded_weight, loaded_shard_id=None):\n'
        "        # Pascal: tuple (fused) shard ids must not go through vllm's\n"
        "        # _load_fused_module_from_checkpoint (0-dim scalar narrow crash).\n"
        '        if hasattr(param, "_store") and (\n'
        "            loaded_shard_id is None or isinstance(loaded_shard_id, tuple)\n"
        "        ):\n"
        "            param._store(loaded_weight)\n"
        "            return\n"
        "        base_loader(param, loaded_weight, loaded_shard_id)\n"
    )
    if "tuple (fused) shard ids" in text:
        print("[Pascal] GGUF quant: type loader bypass already present", flush=True)
    elif tl_old not in text:
        raise SystemExit("pascal_runtime_patch: failed to locate params.py type loader")
    else:
        text = text.replace(tl_old, tl_new, 1)
        changed = True
        print("[Pascal] GGUF quant: type loader bypasses fused narrow", flush=True)

    mm_old = "    layer.register_parameter(param_name, qweight)\n"
    mm_new = (
        "    layer.register_parameter(param_name, qweight)\n"
        "    # Pascal: the old raw parameter is kept alive by its bound-method\n"
        "    # weight_loader fallback; clear its shard tensors so\n"
        "    # process_weights_after_loading does not double VRAM (OOM 4GB).\n"
        "    if hasattr(raw_param, \"data_container\"):\n"
        "        raw_param.data_container.clear()\n"
        "    if hasattr(raw_param, \"shard_id\"):\n"
        "        raw_param.shard_id.clear()\n"
        "    if hasattr(raw_param, \"shard_id_map\"):\n"
        "        raw_param.shard_id_map.clear()\n"
        "    if raw_param.data.numel() > 0:\n"
        "        try:\n"
        "            raw_param.data = torch.empty(\n"
        "                0, dtype=raw_param.data.dtype, device=raw_param.data.device\n"
        "            )\n"
        "        except Exception:\n"
        "            pass\n"
    )
    if "Pascal: the old raw parameter is kept alive" not in text:
        if mm_old not in text:
            raise SystemExit("pascal_runtime_patch: failed to locate params.py register anchor")
        text = text.replace(mm_old, mm_new, 1)
        changed = True
        print("[Pascal] GGUF quant: raw-param shard tensors released", flush=True)
    else:
        print("[Pascal] GGUF quant: raw-param release already present", flush=True)
    par.write_text(text)
    return changed


def patch_tq_backend() -> bool:
    """TurboQuant backend prefill selects FlashAttention-2 when
    _HAS_FLASH_ATTN is set; FA2 has no sm_61 SASS (Ampere+ only), so the
    custom op 'varlen_fwd' is missing at runtime. Force the flag off on
    <sm_80 so the backend uses its per-request SDPA path (which reads K/V
    from the TQ cache for continuation chunks)."""
    path = SITE / "vllm/v1/attention/backends/turboquant_attn.py"
    text = path.read_text()
    old = "_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()\n"
    marker = "# Pascal: FA2 needs sm_80+; force per-request SDPA path"
    if marker in text:
        print("[Pascal] TQ backend: FA2-disable already present", flush=True)
        return False
    if old not in text:
        raise SystemExit("pascal_runtime_patch: failed to locate turboquant _HAS_FLASH_ATTN")
    new = (
        old
        + "# Pascal: FA2 needs sm_80+; force per-request SDPA path\n"
        + "if _HAS_FLASH_ATTN and torch.cuda.is_available():\n"
        + "    _cc = torch.cuda.get_device_capability()\n"
        + "    if _cc[0] < 8:\n"
        + "        _HAS_FLASH_ATTN = False\n"
    )
    text = text.replace(old, new, 1)
    path.write_text(text)
    print("[Pascal] TQ backend: _HAS_FLASH_ATTN forced False below sm_80", flush=True)
    return True


def patch_tq_prefill_sdpa() -> bool:
    """The TurboQuant backend SDPA fallbacks (active on sm61 since FA2 is
    forced off) call F.scaled_dot_product_attention with enable_gqa and 3D
    tensors in _prefill_attention. This torch build's fused kernels need 4D
    and no GQA, so those calls fall back to the math kernel which
    materializes O(seq^2) intermediates -> CUDA OOM on the 4 GB card above
    ~6k-token prefill. Fix: expand K/V explicitly (repeat_interleave gives
    the HF repeat_kv head order), pass 4D tensors, and run under
    sdpa_kernel(EFFICIENT_ATTENTION) with a RuntimeError fallback."""
    path = SITE / "vllm/v1/attention/backends/turboquant_attn.py"
    text = path.read_text()
    marker = "# Pascal: 4D mem-efficient prefill SDPA"
    if marker in text:
        print("[Pascal] TQ prefill SDPA: already patched", flush=True)
        return False

    old1 = ("                    q_t = q_seq.transpose(0, 1).contiguous()\n"
            "                    k_t = k_seq.transpose(0, 1).contiguous()\n"
            "                    v_t = v_seq.transpose(0, 1).contiguous()\n"
            "                    out = F.scaled_dot_product_attention(\n"
            "                        q_t,\n"
            "                        k_t,\n"
            "                        v_t,\n"
            "                        is_causal=True,\n"
            "                        scale=self.scale,\n"
            "                        enable_gqa=use_gqa,\n"
            "                    ).transpose(0, 1)\n")
    new1 = ("                    # Pascal: 4D mem-efficient prefill SDPA\n"
            "                    q4 = q_seq.transpose(0, 1).unsqueeze(0)  # (1, Hq, L, D)\n"
            "                    k4 = k_seq.transpose(0, 1).unsqueeze(0)  # (1, Hk, L, D)\n"
            "                    v4 = v_seq.transpose(0, 1).unsqueeze(0)\n"
            "                    if use_gqa:\n"
            "                        _rep = q4.shape[1] // k4.shape[1]\n"
            "                        k4 = k4.repeat_interleave(_rep, dim=1)\n"
            "                        v4 = v4.repeat_interleave(_rep, dim=1)\n"
            "                    try:\n"
            "                        with torch.nn.attention.sdpa_kernel(\n"
            "                            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION\n"
            "                        ):\n"
            "                            out = F.scaled_dot_product_attention(\n"
            "                                q4, k4, v4, is_causal=True, scale=self.scale\n"
            "                            )\n"
            "                    except RuntimeError:\n"
            "                        out = F.scaled_dot_product_attention(\n"
            "                            q4, k4, v4, is_causal=True, scale=self.scale\n"
            "                        )\n"
            "                    out = out.squeeze(0).transpose(0, 1)\n")

    old2 = ("            # SDPA fallback: expand KV for GQA, build causal mask\n"
            "            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)\n"
            "            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
            "            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
            "            # Build causal mask: query position p can attend to K position j\n"
            "            # where j <= cached_len + p (p is 0-indexed within chunk)\n"
            "            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len\n"
            "            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)\n"
            "            mask = k_pos <= q_pos  # (q_len, seq_len)\n"
            "            out = F.scaled_dot_product_attention(\n"
            "                q_t,\n"
            "                k_t,\n"
            "                v_t,\n"
            "                attn_mask=mask,\n"
            "                scale=self.scale,\n"
            "                enable_gqa=(Hk < Hq),\n"
            "            )  # (1, Hq, q_len, D)\n")
    new2 = ("            # Pascal: 4D mem-efficient prefill SDPA (continuation)\n"
            "            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)\n"
            "            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
            "            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
            "            if Hk < Hq:\n"
            "                _rep = Hq // Hk\n"
            "                k_t = k_t.repeat_interleave(_rep, dim=1)\n"
            "                v_t = v_t.repeat_interleave(_rep, dim=1)\n"
            "            # Build causal mask: query position p can attend to K position j\n"
            "            # where j <= cached_len + p (p is 0-indexed within chunk)\n"
            "            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len\n"
            "            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)\n"
            "            mask = k_pos <= q_pos  # (q_len, seq_len)\n"
            "            try:\n"
            "                with torch.nn.attention.sdpa_kernel(\n"
            "                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION\n"
            "                ):\n"
            "                    out = F.scaled_dot_product_attention(\n"
            "                        q_t, k_t, v_t, attn_mask=mask, scale=self.scale\n"
            "                    )\n"
            "            except RuntimeError:\n"
            "                out = F.scaled_dot_product_attention(\n"
            "                    q_t, k_t, v_t, attn_mask=mask, scale=self.scale\n"
            "                )  # (1, Hq, q_len, D)\n")

    for name, old in (("prefill", old1), ("continuation", old2)):
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                f"pascal_runtime_patch: tq_prefill_sdpa anchor '{name}' count={n}"
            )
    text = text.replace(old1, new1).replace(old2, new2)
    path.write_text(text)
    print("[Pascal] TQ prefill SDPA: 4D mem-efficient paths installed", flush=True)
    return True


def patch_tq_continuation_groups() -> bool:
    """Replace the continuation-prefill wholesale K/V repeat (materialized
    [1, Hq, seq, D] x2, ~4x the compact size) with a per-KV-group loop that
    feeds expanded (stride-0) K/V views straight into SDPA. No seq_len-sized
    buffers are allocated; long continuations (>50k) previously OOMed the
    4 GB card on those buffers. Results are numerically identical (same
    grouped head order)."""
    path = SITE / "vllm/v1/attention/backends/turboquant_attn.py"
    text = path.read_text()
    marker = "# Pascal: 4D mem-efficient prefill SDPA (continuation, grouped)"
    if marker in text:
        print("[Pascal] TQ continuation groups: already patched", flush=True)
        return False

    old = ("            # Pascal: 4D mem-efficient prefill SDPA (continuation)\n"
           "            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)\n"
           "            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
           "            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
           "            if Hk < Hq:\n"
           "                _rep = Hq // Hk\n"
           "                k_t = k_t.repeat_interleave(_rep, dim=1)\n"
           "                v_t = v_t.repeat_interleave(_rep, dim=1)\n"
           "            # Build causal mask: query position p can attend to K position j\n"
           "            # where j <= cached_len + p (p is 0-indexed within chunk)\n"
           "            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len\n"
           "            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)\n"
           "            mask = k_pos <= q_pos  # (q_len, seq_len)\n"
           "            try:\n"
           "                with torch.nn.attention.sdpa_kernel(\n"
           "                    torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION\n"
           "                ):\n"
           "                    out = F.scaled_dot_product_attention(\n"
           "                        q_t, k_t, v_t, attn_mask=mask, scale=self.scale\n"
           "                    )\n"
           "            except RuntimeError:\n"
           "                out = F.scaled_dot_product_attention(\n"
           "                    q_t, k_t, v_t, attn_mask=mask, scale=self.scale\n"
           "                )  # (1, Hq, q_len, D)\n")
    new = ("            # Pascal: 4D mem-efficient prefill SDPA (continuation, grouped)\n"
           "            q_t = query.transpose(0, 1).unsqueeze(0)  # (1, Hq, q_len, D)\n"
           "            k_t = k_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
           "            v_t = v_full.transpose(0, 1).unsqueeze(0)  # (1, Hk, seq_len, D)\n"
           "            # Build causal mask: query position p can attend to K position j\n"
           "            # where j <= cached_len + p (p is 0-indexed within chunk)\n"
           "            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len\n"
           "            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)\n"
           "            mask = k_pos <= q_pos  # (q_len, seq_len)\n"
           "            if Hk < Hq:\n"
           "                # Pascal: per-KV-group attention; avoids materializing\n"
           "                # [1, Hq, seq, D] K/V copies (repeat x4) that OOMed long\n"
           "                # continuation chunks on the 4 GB card. Expanded into\n"
           "                # small per-group buffers, reused across groups.\n"
           "                _rep = Hq // Hk\n"
           "                out = torch.empty(1, Hq, q_len, D, device=q_t.device, dtype=q_t.dtype)\n"
           "                # Pascal sm_61 fix 2026-09-15: do not materialize per-group\n"
           "                # K/V buffers (they scale with seq_len and OOMed long\n"
           "                # continuations >50k on the 4 GB card); pass expanded\n"
           "                # (stride-0) views directly into SDPA instead.\n"
           "                for _j in range(Hk):\n"
           "                    _kj = k_t[:, _j:_j + 1].expand(1, _rep, seq_len, D)\n"
           "                    _vj = v_t[:, _j:_j + 1].expand(1, _rep, seq_len, D)\n"
           "                    _qj = q_t[:, _j * _rep:(_j + 1) * _rep]\n"
           "                    try:\n"
           "                        with torch.nn.attention.sdpa_kernel(\n"
           "                            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION\n"
           "                        ):\n"
           "                            out[:, _j * _rep:(_j + 1) * _rep] = F.scaled_dot_product_attention(\n"
           "                                _qj, _kj, _vj, attn_mask=mask, scale=self.scale\n"
           "                            )\n"
           "                    except RuntimeError:\n"
           "                        out[:, _j * _rep:(_j + 1) * _rep] = F.scaled_dot_product_attention(\n"
           "                            _qj, _kj, _vj, attn_mask=mask, scale=self.scale\n"
           "                        )\n"
           "            else:\n"
           "                try:\n"
           "                    with torch.nn.attention.sdpa_kernel(\n"
           "                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION\n"
           "                    ):\n"
           "                        out = F.scaled_dot_product_attention(\n"
           "                            q_t, k_t, v_t, attn_mask=mask, scale=self.scale\n"
           "                        )\n"
           "                except RuntimeError:\n"
           "                    out = F.scaled_dot_product_attention(\n"
           "                        q_t, k_t, v_t, attn_mask=mask, scale=self.scale\n"
           "                    )  # (1, Hq, q_len, D)\n")

    n = text.count(old)
    if n != 1:
        raise SystemExit(
            f"pascal_runtime_patch: tq_continuation_groups anchor count={n}"
        )
    text = text.replace(old, new)
    path.write_text(text)
    print("[Pascal] TQ continuation groups: grouped K/V attention installed", flush=True)
    return True


PENALTIES = SITE / "vllm/v1/worker/gpu/sample/penalties.py"


def patch_penalties_bincount() -> bool:
    """Replace the Triton _bincount_kernel call with a PyTorch equivalent.

    The Triton kernel uses atomic_or/atomic_add which the bundled Triton
    lowers to .acq_rel/.relaxed PTX that requires sm_70+ (fails on sm_61).
    The fallback is numerically identical and only runs for requests that
    actually use repetition/presence/frequency penalties.
    """
    if not PENALTIES.exists():
        print("[Pascal] penalties: file not found, skip", flush=True)
        return False
    text = PENALTIES.read_text()
    if "_pascal_bincount_torch" in text:
        print("[Pascal] penalties: torch fallback already applied", flush=True)
        return False
    old = (
        "    num_tokens = expanded_idx_mapping.shape[0]\n"
        "    BLOCK_SIZE = 1024\n"
        "    num_blocks = triton.cdiv(max_prefill_len, BLOCK_SIZE)\n"
        "    _bincount_kernel[(num_tokens, num_blocks)](\n"
        "        expanded_idx_mapping,\n"
        "        all_token_ids,\n"
        "        all_token_ids.stride(0),\n"
        "        prompt_len,\n"
        "        prefill_len,\n"
        "        prompt_bin_mask,\n"
        "        prompt_bin_mask.stride(0),\n"
        "        output_bin_counts,\n"
        "        output_bin_counts.stride(0),\n"
        "        BLOCK_SIZE=BLOCK_SIZE,\n"
        "    )\n"
    )
    if old not in text:
        raise SystemExit(
            "pascal_runtime_patch: failed to locate _bincount_kernel call"
        )
    new = (
        "    # [Pascal sm_61] Triton _bincount_kernel uses atomic ops that need\n"
        "    # sm_70+; use the PyTorch equivalent instead (numerically identical).\n"
        "    _pascal_bincount_torch(\n"
        "        expanded_idx_mapping,\n"
        "        all_token_ids,\n"
        "        prompt_len,\n"
        "        prefill_len,\n"
        "        prompt_bin_mask,\n"
        "        output_bin_counts,\n"
        "    )\n"
    )
    text = text.replace(old, new, 1)
    text = text.rstrip() + "\n\n\n" + _PASCAL_BINCOUNT_TORCH
    PENALTIES.write_text(text)
    print("[Pascal] penalties: bincount torch fallback applied", flush=True)
    return True


_PASCAL_BINCOUNT_TORCH = textwrap.dedent('''\
def _pascal_bincount_torch(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
) -> None:
    """PyTorch equivalent of the Triton _bincount_kernel (Pascal sm_61).

    Sets bit (token % 32) of word (token // 32) in prompt_bin_mask for the
    prompt tokens of each request, and counts output tokens (from prompt_len
    to prefill_len) in output_bin_counts. Both outputs are int32 (num_reqs,
    dim) tensors. The per-word bit assignments are unique per token, so
    summing single bits is equivalent to bitwise OR.
    """
    num_tokens = expanded_idx_mapping.shape[0]
    for i in range(num_tokens):
        r = int(expanded_idx_mapping[i].item())
        pl = int(prompt_len[r].item())
        fl = int(prefill_len[r].item())
        if pl > 0:
            t = all_token_ids[r, :pl].to(torch.int64)
            uniq = torch.unique(t)
            words = uniq >> 5
            bits = (torch.ones_like(uniq) << (uniq & 31)).to(
                prompt_bin_mask.dtype
            )
            prompt_bin_mask[r].index_add_(0, words, bits)
        if fl > pl:
            t2 = all_token_ids[r, pl:fl].to(torch.int64)
            output_bin_counts[r].index_add_(
                0, t2, torch.ones_like(t2, dtype=output_bin_counts.dtype)
            )
''')


LLAMA_MODEL = SITE / "vllm/model_executor/models/llama.py"


def patch_llama_layer_probe() -> bool:
    """TEMP PROBE: per-layer hidden-state stats for llama-arch models
    (HF cross-check for the MiniCPM5 pilot). Enabled via PASCAL_GDN_DEBUG=1."""
    path = LLAMA_MODEL
    if not path.exists():
        print("[Pascal] llama probe: file not found, skip", flush=True)
        return False
    text = path.read_text()
    old = (
        "            hidden_states, residual = layer(\n"
        "                positions, hidden_states, residual, **extra_layer_kwargs\n"
        "            )\n"
        "            self._maybe_add_hidden_state(\n"
    )
    if "[LAYER-L]" in text:
        print("[Pascal] llama probe: already applied", flush=True)
        return False
    if old not in text:
        raise SystemExit("pascal_runtime_patch: failed to locate llama layer loop")
    new = (
        "            hidden_states, residual = layer(\n"
        "                positions, hidden_states, residual, **extra_layer_kwargs\n"
        "            )\n"
        "            import os as _os\n"
        "            if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "                _hs = hidden_states\n"
        "                _full = _hs.float() + (0.0 if residual is None else residual.float())\n"
        "                print(f\"[LAYER-L] n={idx} mean={float(_hs.float().abs().mean()):.6f} \"\n"
        "                      f\"sum_mean={float(_full.abs().mean()):.6f} \"\n"
        "                      f\"last6={[round(float(v), 4) for v in _hs[-1, :6]]} \"\n"
        "                      f\"full6={[round(float(v), 4) for v in _full[-1, :6]]}\", flush=True)\n"
        "                try:\n"
        "                    import torch as _t3\n"
        "                    _t3.save(_full.detach().cpu(),\n"
        "                             f\"/logs/vllm_full_tok{_full.shape[0]}_L{idx}.pt\")\n"
        "                except Exception:\n"
        "                    pass\n"
        "            self._maybe_add_hidden_state(\n"
    )
    text = text.replace(old, new, 1)
    # third probe: layer-0 internals (norm-in and attention-out) via DecoderLayer
    if "_pascal_dbg" not in text:
        old3 = (
            "        if residual is None:\n"
            "            residual = hidden_states\n"
            "            hidden_states = self.input_layernorm(hidden_states)\n"
            "        else:\n"
            "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n"
            "        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)\n"
        )
        new3 = (
            "        _pascal_dbg = None\n"
            "        if residual is None:\n"
            "            residual = hidden_states\n"
            "            hidden_states = self.input_layernorm(hidden_states)\n"
            "            _pascal_dbg = \"0\"\n"
            "        else:\n"
            "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n"
            "        import os as _os3\n"
            "        if _pascal_dbg == \"0\" and _os3.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
            "            try:\n"
            "                import torch as _t4\n"
            "                _t4.save(hidden_states.detach().cpu(),\n"
            "                         f\"/logs/vllm_normin_tok{hidden_states.shape[0]}.pt\")\n"
            "            except Exception:\n"
            "                pass\n"
            "        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)\n"
            "        if _pascal_dbg == \"0\" and _os3.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
            "            try:\n"
            "                import torch as _t4\n"
            "                _t4.save(hidden_states.detach().cpu(),\n"
            "                         f\"/logs/vllm_attnout_tok{hidden_states.shape[0]}.pt\")\n"
            "            except Exception:\n"
            "                pass\n"
        )
        if old3 in text:
            text = text.replace(old3, new3, 1)
            print("[Pascal] llama probe: layer0 internals save added", flush=True)
        else:
            print("[Pascal] llama probe: DecoderLayer anchor missing (skip internals)", flush=True)
    if "[FINAL-L]" not in text:
        new2 = (
            "        hidden_states, _ = self.norm(hidden_states, residual)\n"
            "        import os as _os2\n"
            "        if _os2.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
            "            print(f\"[FINAL-L] mean={float(hidden_states.float().abs().mean()):.6f} \"\n"
            "                  f\"last6={[round(float(v), 4) for v in hidden_states[-1, :6]]}\", flush=True)\n"
            "            try:\n"
            "                import torch as _t2\n"
            "                _t2.save(hidden_states.detach().cpu(),\n"
            "                         f\"/logs/final_hidden_tok{hidden_states.shape[0]}.pt\")\n"
            "            except Exception:\n"
            "                pass\n"
        )
        old2 = "        hidden_states, _ = self.norm(hidden_states, residual)\n"
        if old2 in text:
            text = text.replace(old2, new2, 1)
        else:
            print("[Pascal] llama probe: final-norm anchor missing (skip)", flush=True)
    else:
        print("[Pascal] llama probe: final-norm print already present", flush=True)
    path.write_text(text)
    print("[Pascal] llama probe: layer-wise hidden print added", flush=True)
    return True


def patch_qwen_embeds() -> bool:
    """The GGUF file quantizes token embeddings (Q6_K); the vllm qwen3_5 /
    qwen3_next models create VocabParallelEmbedding WITHOUT the quant_config,
    so loading 'embed_tokens.qweight_type' fails. Pass the quant config so the
    GGUFEmbeddingMethod materializes a quantized embedding."""
    changed = False
    for fname in ("qwen3_5.py", "qwen3_next.py"):
        path = SITE / f"vllm/model_executor/models/{fname}"
        text = path.read_text()
        if "quant_config=getattr(self, \"quant_config\", None)" in text:
            print(f"[Pascal] embed quant_config already present in {fname}", flush=True)
            continue
        old = "        self.embed_tokens = VocabParallelEmbedding(\n            self.vocab_size,\n            config.hidden_size,\n        )"
        if old not in text:
            # variant with org_num_embeddings / quant_config=None
            old = "        self.embed_tokens = VocabParallelEmbedding(\n            self.vocab_size,\n            config.hidden_size,\n            org_num_embeddings=self.vocab_size,\n            quant_config=None,\n        )"
        if old not in text:
            raise SystemExit(f"pascal_runtime_patch: failed to locate embed anchor in {fname}")
        new = old.replace(
            "            config.hidden_size,\n        )",
            "            config.hidden_size,\n            quant_config=getattr(self, \"quant_config\", None),\n        )",
        )
        if new == old:
            new = old.replace(
                "            quant_config=None,\n        )",
                "            quant_config=getattr(self, \"quant_config\", None),\n        )",
            )
        text = text.replace(old, new, 1)
        path.write_text(text)
        changed = True
        print(f"[Pascal] embed quant_config patched in {fname}", flush=True)

    # Debug: print the first token's embedding values (compares against the
    # gguf-package dequantize reference) when PASCAL_GDN_DEBUG is set.
    qn = SITE / "vllm/model_executor/models/qwen3_next.py"
    qn_text = qn.read_text()
    e_old = "        return self.embed_tokens(input_ids)"
    e_new = (
        "        import os as _os\n"
        "        _hs = self.embed_tokens(input_ids)\n"
        "        if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
        "            print(f\"[EMBED] ids={input_ids.flatten()[:6].tolist()} \"\n"
        "                  f\"vals={[round(float(v), 4) for v in _hs[0, :6].flatten()]}\", flush=True)\n"
        "        return _hs"
    )
    if e_old in qn_text and "self.embed_tokens(input_ids)" not in qn_text.replace(e_old, "", 1):
        qn_text = qn_text.replace(e_old, e_new, 1)
        qn.write_text(qn_text)
        print("[Pascal] embed debug print added (qwen3_next)", flush=True)
    return changed


# NOTE (2026-09-14, v0.28 migration): the temporary RoPE-disable probe was
# REMOVED from this patch module. It must never be re-enabled:
# full-attention layers REQUIRE RoPE, and a stale container had silently
# disabled rotary embeddings via "PASCAL PROBE: RoPE disabled".
# Verification: grep for "rotary_emb(positions, q, k)" in qwen3_next.py.


def patch_qwen_layer_probe() -> bool:
    """TEMP PROBE: print per-layer hidden-state stats for HF cross-check."""
    path = SITE / "vllm/model_executor/models/qwen3_next.py"
    text = path.read_text()
    old = (
        "            self._maybe_add_hidden_state(\n"
        "                aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
        "            )\n"
    )
    if old in text and "[LAYER-N]" not in text:
        new = (
            "            import os as _os\n"
            "            if _os.environ.get(\"PASCAL_GDN_DEBUG\") == \"1\":\n"
            "                _h0 = hidden_states[0, :6].float()\n"
            "                _full = hidden_states.float() + (0.0 if residual is None else residual.float())\n"
            "                _f6 = _full[0, :6]\n"
            "                print(f\"[LAYER-N] n={layer_idx} out_mean={float(hidden_states.float().abs().mean()):.6f} \"\n"
            "                      f\"sum_mean={float(_full.abs().mean()):.6f} \"\n"
            "                      f\"first6={[round(float(v), 4) for v in _h0]} \"\n"
            "                      f\"full6={[round(float(v), 4) for v in _f6]}\", flush=True)\n"
            "            self._maybe_add_hidden_state(\n"
            "                aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
            "            )\n"
        )
        text = text.replace(old, new, 1)
        path.write_text(text)
        print("[Pascal] PROBE: layer-wise hidden print added", flush=True)
        return True
    return False


def main() -> None:
    applied = []
    if patch_gdn_torch():
        applied.append("gdn_torch")
    if patch_triton_attn():
        applied.append("sdpa")
    if patch_gguf_adapter():
        applied.append("gguf_adapter")
    if patch_gguf_quant():
        applied.append("gguf_quant")
    if patch_tq_backend():
        applied.append("tq_backend")
    if patch_tq_prefill_sdpa():
        applied.append("tq_prefill_sdpa")
    if patch_tq_continuation_groups():
        applied.append("tq_continuation_groups")
    if patch_penalties_bincount():
        applied.append("penalties_bincount")
    if patch_llama_layer_probe():
        applied.append("llama_probe")
    if patch_qwen_embeds():
        applied.append("qwen_embeds")
    if patch_qwen_layer_probe():
        applied.append("layer_probe")
    print(f"[Pascal] runtime patches applied: {applied}", flush=True)


if __name__ == "__main__":
    main()

"""
TurboQuant integration hook for vLLM Pascal (sm_61).

This module is installed as a runtime hot-patch by start-vllm.sh.
It monkey-patches the vLLM Attention layer to intercept KV cache writes
on full-attention layers (8 out of 32 in Qwen3.5-4B), compressing them
via TurboQuant's rotation + Lloyd-Max + QJL pipeline.

Modes:
    - off:          No TQ activity — passthrough.
  - capture_only: Capture KV into compressed store alongside normal paged cache;
                  always use the standard paged-cache attention output.
                  Reports VRAM savings in logs. No quality impact.
  - hybrid:       Use compressed historical KV + exact recent ring buffer for
                  decode-phase attention on full-attention layers.
                  Runs both original and TQ attention; returns TQ output.
  - full:         Replace paged KV cache attention entirely on decode.
                  Prefill uses original attention + captures to TQ store.
                  Decode computes attention purely from TQ compressed store
                  + exact ring buffer, skipping the original forward.
                  Eliminates float32 intermediate memory bottleneck.

Environment variables:
    TURBOQUANT_MODE       off | capture_only | hybrid | full  (default: full)
  TURBOQUANT_KEY_BITS   Key quantization bits (default: 3)
  TURBOQUANT_VAL_BITS   Value quantization bits (default: 2)
  TURBOQUANT_RING_CAP   Ring buffer capacity in tokens (default: 64)
"""

from __future__ import annotations

import logging
import math
import os
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("turboquant.pascal_hook")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODE_OFF = "off"
MODE_CAPTURE_ONLY = "capture_only"
MODE_HYBRID = "hybrid"
MODE_FULL = "full"

_GLOBAL_MODE = os.environ.get("TURBOQUANT_MODE", MODE_FULL)
_KEY_BITS = int(os.environ.get("TURBOQUANT_KEY_BITS", "3"))
_VAL_BITS = int(os.environ.get("TURBOQUANT_VAL_BITS", "2"))
_RING_CAP = int(os.environ.get("TURBOQUANT_RING_CAP", "64"))

# Registry: layer_name -> TQLayerState
_TQ_STATES: dict[str, "TQLayerState"] = {}


# ---------------------------------------------------------------------------
# Per-layer state
# ---------------------------------------------------------------------------
@dataclass
class TQLayerState:
    """Per full-attention-layer TurboQuant state."""
    layer_idx: int
    head_dim: int
    num_kv_heads: int
    num_query_heads: int
    key_bits: int
    value_bits: int
    ring_capacity: int
    device: torch.device

    # Lazy-initialized on first capture
    engine: object = field(default=None, repr=False)
    store: object = field(default=None, repr=False)
    _initialized: bool = False

    def ensure_initialized(self):
        """Lazy init: create TQ store + engine on first use."""
        if self._initialized:
            return
        from turboquant.store import CompressedKVStore
        from turboquant.capture import KVCaptureEngine

        self.store = CompressedKVStore(
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            key_bits=self.key_bits,
            value_bits=self.value_bits,
            value_group_size=min(32, self.head_dim),
            device=self.device,
            layer_idx=self.layer_idx,
        )
        self.engine = KVCaptureEngine(
            store=self.store,
            ring_capacity=self.ring_capacity,
            device=self.device,
            dtype=torch.float16,  # Pascal SM_61 doesn't support bfloat16
        )
        self._initialized = True
        logger.info(
            "[TurboQuant] Initialized layer %d: head_dim=%d, kv_heads=%d, "
            "key_bits=%d, val_bits=%d, ring=%d",
            self.layer_idx, self.head_dim, self.num_kv_heads,
            self.key_bits, self.value_bits, self.ring_capacity,
        )


def _create_tq_state(layer_idx, head_dim, num_kv_heads, num_query_heads, device):
    return TQLayerState(
        layer_idx=layer_idx,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_query_heads=num_query_heads,
        key_bits=_KEY_BITS,
        value_bits=_VAL_BITS,
        ring_capacity=_RING_CAP,
        device=device,
    )


# ---------------------------------------------------------------------------
# Hook: KV cache capture
# ---------------------------------------------------------------------------
def _capture_kv(state: TQLayerState, key: torch.Tensor, value: torch.Tensor,
                num_tokens: int):
    """Capture KV tensors into TQ compressed store."""
    state.ensure_initialized()
    if num_tokens <= 1:
        state.engine.ingest_decode(key[:num_tokens], value[:num_tokens], num_tokens)
    else:
        state.engine.ingest_prefill(key[:num_tokens], value[:num_tokens], num_tokens)


# ---------------------------------------------------------------------------
# Patching: Attention.forward wrapper
# ---------------------------------------------------------------------------
_ORIG_ATTENTION_FORWARD = None
_CAPTURE_CALL_COUNT = 0
_CAPTURE_OK_COUNT = 0
_CAPTURE_ERR_COUNT = 0
_HYBRID_CALL_COUNT = 0
_HYBRID_OK_COUNT = 0
_HYBRID_ERR_COUNT = 0
_HYBRID_REL_ERR_SUM = 0.0
_FULL_CALL_COUNT = 0
_FULL_OK_COUNT = 0
_FULL_ERR_COUNT = 0


def _patched_attention_forward(self_attn, query, key, value, output_shape=None):
    """Wrapped Attention.forward that adds TQ capture and optional hybrid decode."""
    global _GLOBAL_MODE, _CAPTURE_CALL_COUNT, _CAPTURE_OK_COUNT, _CAPTURE_ERR_COUNT

    layer_name = getattr(self_attn, "layer_name", None)
    tq_state = _TQ_STATES.get(layer_name) if layer_name else None

    # No TQ state for this layer (GDN layers, or TQ disabled)
    if tq_state is None or _GLOBAL_MODE == MODE_OFF:
        return _ORIG_ATTENTION_FORWARD(self_attn, query, key, value, output_shape)

    # Reset TQ store on prefill (new request) to prevent cross-request
    # KV contamination.  With max_num_seqs=1, every prefill is a fresh request.
    if query.shape[0] > 1 and tq_state._initialized:
        tq_state.engine.reset()
        tq_state._dq_store_ver = -1

    # --- Capture mode: always run original, also capture KV ---
    if _GLOBAL_MODE == MODE_CAPTURE_ONLY:
        result = _ORIG_ATTENTION_FORWARD(self_attn, query, key, value, output_shape)
        # Capture K and V after original forward (they are the raw projections)
        if key is not None and value is not None:
            _CAPTURE_CALL_COUNT += 1
            num_tokens = key.shape[0]
            k_shaped = key.view(-1, self_attn.num_kv_heads, self_attn.head_size)
            v_shaped = value.view(-1, self_attn.num_kv_heads, self_attn.head_size)
            try:
                _capture_kv(tq_state, k_shaped, v_shaped, num_tokens)
                _CAPTURE_OK_COUNT += 1
                # Log stats periodically (every 100 captures on layer 0)
                if tq_state.layer_idx == 0 and _CAPTURE_OK_COUNT % 100 == 1:
                    logger.warning(
                        "[TurboQuant] Capture active: calls=%d, ok=%d, err=%d, "
                        "tokens_stored=%d",
                        _CAPTURE_CALL_COUNT, _CAPTURE_OK_COUNT,
                        _CAPTURE_ERR_COUNT, tq_state.store.num_tokens,
                    )
            except Exception as e:
                _CAPTURE_ERR_COUNT += 1
                if tq_state.layer_idx == 0 and _CAPTURE_ERR_COUNT <= 3:
                    logger.warning("[TurboQuant] Capture error on layer 0: %s", e)
        return result

    # --- Hybrid mode: run original, then replace decode output with TQ ---
    if _GLOBAL_MODE == MODE_HYBRID:
        global _HYBRID_CALL_COUNT, _HYBRID_OK_COUNT, _HYBRID_ERR_COUNT
        global _HYBRID_REL_ERR_SUM

        # Always run original forward (keeps paged KV cache consistent)
        orig_result = _ORIG_ATTENTION_FORWARD(
            self_attn, query, key, value, output_shape
        )

        # Also capture KV to TQ
        if key is not None and value is not None:
            _CAPTURE_CALL_COUNT += 1
            num_tokens = key.shape[0]
            k_shaped = key.view(-1, self_attn.num_kv_heads, self_attn.head_size)
            v_shaped = value.view(-1, self_attn.num_kv_heads, self_attn.head_size)
            try:
                _capture_kv(tq_state, k_shaped, v_shaped, num_tokens)
                _CAPTURE_OK_COUNT += 1
            except Exception:
                _CAPTURE_ERR_COUNT += 1

        # For decode (single token), compute hybrid attention from TQ store
        is_decode = query.shape[0] == 1
        if is_decode and tq_state._initialized:
            try:
                from turboquant.score import compute_hybrid_attention

                scale = 1.0 / math.sqrt(tq_state.head_dim)
                q_shaped = query.view(
                    -1, tq_state.num_query_heads, tq_state.head_dim
                )

                recent = tq_state.engine.ring.peek()
                recent_k = recent[0] if recent else None
                recent_v = recent[1] if recent else None

                hybrid_out = compute_hybrid_attention(
                    query=q_shaped,
                    store=tq_state.store,
                    recent_k=recent_k,
                    recent_v=recent_v,
                    num_query_heads=tq_state.num_query_heads,
                    scale=scale,
                )
                if hybrid_out is not None:
                    flat_out = hybrid_out.reshape(orig_result.shape).to(
                        orig_result.dtype
                    )
                    _HYBRID_OK_COUNT += 1

                    # Quality tracking: relative error on layer 0
                    if tq_state.layer_idx == 0:
                        _HYBRID_CALL_COUNT += 1
                        diff = (flat_out - orig_result).float().norm()
                        orig_norm = orig_result.float().norm().clamp(min=1e-8)
                        _HYBRID_REL_ERR_SUM += (diff / orig_norm).item()
                        if _HYBRID_CALL_COUNT % 100 == 1:
                            avg_err = _HYBRID_REL_ERR_SUM / _HYBRID_CALL_COUNT
                            logger.warning(
                                "[TurboQuant] Hybrid active: decode_calls=%d, "
                                "avg_rel_err=%.6f, capture ok/err=%d/%d",
                                _HYBRID_CALL_COUNT, avg_err,
                                _CAPTURE_OK_COUNT, _CAPTURE_ERR_COUNT,
                            )
                    return flat_out
            except Exception as e:
                _HYBRID_ERR_COUNT += 1
                if _HYBRID_ERR_COUNT <= 5:
                    logger.warning(
                        "[TurboQuant] Hybrid decode error (layer %d): %s",
                        tq_state.layer_idx, e,
                    )

        return orig_result

    # --- Full mode: replace paged KV entirely on decode ---
    if _GLOBAL_MODE == MODE_FULL:
        global _FULL_CALL_COUNT, _FULL_OK_COUNT, _FULL_ERR_COUNT

        is_decode = query.shape[0] == 1

        if not is_decode:
            # Prefill: run original attention + capture KV to TQ store
            result = _ORIG_ATTENTION_FORWARD(
                self_attn, query, key, value, output_shape
            )
            if key is not None and value is not None:
                _CAPTURE_CALL_COUNT += 1
                num_tokens = key.shape[0]
                k_shaped = key.view(-1, self_attn.num_kv_heads, self_attn.head_size)
                v_shaped = value.view(-1, self_attn.num_kv_heads, self_attn.head_size)
                try:
                    _capture_kv(tq_state, k_shaped, v_shaped, num_tokens)
                    _CAPTURE_OK_COUNT += 1
                except Exception as e:
                    _CAPTURE_ERR_COUNT += 1
                    if _CAPTURE_ERR_COUNT <= 3:
                        logger.warning(
                            "[TurboQuant] Full-mode prefill capture error "
                            "(layer %d): %s", tq_state.layer_idx, e,
                        )
            # Invalidate dequant cache after prefill changes the store
            tq_state._dq_store_ver = -1
            return result

        # Decode: capture KV, then compute attention from TQ store
        if key is not None and value is not None:
            _CAPTURE_CALL_COUNT += 1
            num_tokens = key.shape[0]
            k_shaped = key.view(-1, self_attn.num_kv_heads, self_attn.head_size)
            v_shaped = value.view(-1, self_attn.num_kv_heads, self_attn.head_size)
            try:
                _capture_kv(tq_state, k_shaped, v_shaped, num_tokens)
                _CAPTURE_OK_COUNT += 1
            except Exception:
                _CAPTURE_ERR_COUNT += 1

        if tq_state._initialized:
            try:
                flat = tq_state.store.get_flat_cache()
                store_tokens = flat.num_tokens if flat is not None else 0
                cached_ver = getattr(tq_state, '_dq_store_ver', -1)

                # Re-dequantize only when compressed store has changed
                # (ring buffer flushes every RING_CAP tokens)
                if store_tokens != cached_ver:
                    if flat is not None and store_tokens >= 16:
                        from turboquant.kv_cache import dequantize_values
                        # Cache in fp16 to halve memory bandwidth
                        tq_state._dq_k = tq_state.store.quantizer.dequantize(
                            flat.prod_q
                        ).half()  # (H_kv, N_hist, D)
                        tq_state._dq_v = dequantize_values(
                            flat.value_q, 32
                        ).half()  # (H_kv, N_hist, D)
                    else:
                        tq_state._dq_k = None
                        tq_state._dq_v = None
                    tq_state._dq_store_ver = store_tokens

                has_hist = getattr(tq_state, '_dq_k', None) is not None
                recent = tq_state.engine.ring.peek()
                has_recent = recent is not None

                if has_hist or has_recent:
                    H_kv = tq_state.num_kv_heads
                    D = tq_state.head_dim
                    gqa = tq_state.num_query_heads // H_kv
                    scale = 1.0 / math.sqrt(D)

                    # Build KV — fp16 throughout to save bandwidth
                    if has_hist and has_recent:
                        rk = recent[0].transpose(0, 1).half()
                        rv = recent[1].transpose(0, 1).half()
                        k_all = torch.cat([tq_state._dq_k, rk], dim=1)
                        v_all = torch.cat([tq_state._dq_v, rv], dim=1)
                    elif has_hist:
                        k_all = tq_state._dq_k
                        v_all = tq_state._dq_v
                    else:
                        k_all = recent[0].transpose(0, 1).half()
                        v_all = recent[1].transpose(0, 1).half()

                    # Attention in fp16, softmax in fp32 for stability
                    q = query.half().view(1, H_kv, gqa, D).permute(
                        1, 2, 0, 3
                    )  # (H_kv, G, 1, D) fp16
                    k = k_all.unsqueeze(1)  # (H_kv, 1, N, D) fp16
                    v = v_all.unsqueeze(1)  # (H_kv, 1, N, D) fp16

                    scores = torch.matmul(
                        q, k.transpose(-2, -1)
                    ) * scale  # (H_kv, G, 1, N) fp16
                    weights = F.softmax(scores.float(), dim=-1).half()
                    out = torch.matmul(weights, v)  # (H_kv, G, 1, D) fp16

                    flat_out = out.permute(2, 0, 1, 3).reshape(
                        output_shape if output_shape is not None
                        else query.shape
                    ).to(query.dtype)
                    _FULL_OK_COUNT += 1
                    _FULL_CALL_COUNT += 1
                    if tq_state.layer_idx == 0 and _FULL_CALL_COUNT % 200 == 1:
                        logger.warning(
                            "[TurboQuant] Full mode: decode=%d, "
                            "ok=%d, err=%d, store=%d, dq_hit=%s",
                            _FULL_CALL_COUNT, _FULL_OK_COUNT,
                            _FULL_ERR_COUNT, store_tokens,
                            store_tokens == cached_ver,
                        )
                    return flat_out
            except Exception as e:
                _FULL_ERR_COUNT += 1
                if _FULL_ERR_COUNT <= 5:
                    logger.warning(
                        "[TurboQuant] Full-mode decode error (layer %d): %s",
                        tq_state.layer_idx, e,
                    )

        # Fallback: if TQ store isn't initialized yet, use original
        return _ORIG_ATTENTION_FORWARD(
            self_attn, query, key, value, output_shape
        )

    return _ORIG_ATTENTION_FORWARD(self_attn, query, key, value, output_shape)


# ---------------------------------------------------------------------------
# Public API: install hooks after model load
# ---------------------------------------------------------------------------
def install_hooks_on_model(model, device=None):
    """Discover full-attention layers in the loaded model and register TQ states.

    Call this AFTER the model is loaded but BEFORE inference starts.
    """
    global _ORIG_ATTENTION_FORWARD

    if _GLOBAL_MODE == MODE_OFF:
        logger.info("[TurboQuant] Mode is OFF, skipping hook installation")
        return

    if device is None:
        device = torch.device("cuda:0")

    from vllm.model_executor.layers.attention.attention import Attention

    # Install forward wrapper (once)
    if _ORIG_ATTENTION_FORWARD is None:
        _ORIG_ATTENTION_FORWARD = Attention.forward
        Attention.forward = _patched_attention_forward
        logger.warning("[TurboQuant] Installed Attention.forward wrapper")

    # Walk model to find attention layers that expose an inner vLLM Attention
    next_layer_idx = len(_TQ_STATES)
    full_attn_count = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        attn = getattr(module, "attn", None)
        if cls_name not in ("Qwen3NextAttention", "DFlashQwen3Attention"):
            continue
        if not isinstance(attn, Attention):
            continue

        layer_name = getattr(attn, "layer_name", None)
        if layer_name and layer_name not in _TQ_STATES:
            head_dim = attn.head_size
            num_kv_heads = attn.num_kv_heads
            num_q_heads = attn.num_heads
            state = _create_tq_state(
                layer_idx=next_layer_idx,
                head_dim=head_dim,
                num_kv_heads=num_kv_heads,
                num_query_heads=num_q_heads,
                device=device,
            )
            _TQ_STATES[layer_name] = state
            next_layer_idx += 1
            full_attn_count += 1

    logger.warning(
        "[TurboQuant] Registered %d attention layers (mode=%s, "
        "key_bits=%d, val_bits=%d, ring=%d)",
        full_attn_count, _GLOBAL_MODE, _KEY_BITS, _VAL_BITS, _RING_CAP,
    )


def get_stats() -> dict:
    """Return TQ compression statistics for logging."""
    stats = {
        "mode": _GLOBAL_MODE,
        "num_layers": len(_TQ_STATES),
        "layers": {},
    }
    total_compressed = 0
    total_uncompressed = 0
    for name, state in _TQ_STATES.items():
        if not state._initialized:
            continue
        mem = state.store.memory_bytes()
        # Estimate uncompressed equivalent
        n_tokens = 0
        flat = state.store.get_flat_cache()
        if flat is not None:
            n_tokens = flat.num_tokens
        uncomp = n_tokens * state.num_kv_heads * state.head_dim * 2 * 2  # K+V, fp16
        total_compressed += mem
        total_uncompressed += uncomp
        stats["layers"][name] = {
            "tokens": n_tokens,
            "compressed_mb": mem / 1e6,
            "uncompressed_mb": uncomp / 1e6,
            "ratio": uncomp / max(mem, 1),
        }
    stats["total_compressed_mb"] = total_compressed / 1e6
    stats["total_uncompressed_mb"] = total_uncompressed / 1e6
    stats["compression_ratio"] = total_uncompressed / max(total_compressed, 1)
    stats["capture"] = {
        "calls": _CAPTURE_CALL_COUNT,
        "ok": _CAPTURE_OK_COUNT,
        "err": _CAPTURE_ERR_COUNT,
    }
    stats["hybrid"] = {
        "calls": _HYBRID_CALL_COUNT,
        "ok": _HYBRID_OK_COUNT,
        "err": _HYBRID_ERR_COUNT,
    }
    stats["full"] = {
        "calls": _FULL_CALL_COUNT,
        "ok": _FULL_OK_COUNT,
        "err": _FULL_ERR_COUNT,
    }
    return stats

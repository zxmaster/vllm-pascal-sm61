#!/usr/bin/env python3
"""
pascal_gdn_torch.py — pure-PyTorch implementations of the GDN (Gated Delta Net)
kernels for Pascal (sm_61), where the upstream Triton FLA kernels produce
incorrect results (see MEMORIES.md §6/§13).

Replaces, in vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn:
  - fused_post_conv_prep
  - fused_sigmoid_gating_delta_rule_update
  - ChunkGatedDeltaRule (prefill) -> forward_torch

Math mirrors the upstream Triton kernels exactly:
  g      = -exp(A_log) * softplus(a + dt_bias)
  beta   = sigmoid(b)
  H      = H * exp(g)
  u      = (v - H @ k) * beta
  H      = H + outer(u, k)
  o      = H @ (q * scale)

Installed at runtime by start-vllm-v028.sh when VLLM_PASCAL_TORCH_GDN=1.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def pascal_causal_conv1d(
    x: torch.Tensor,          # [dim, total_tokens] (vllm layout)
    weight: torch.Tensor,     # [dim, width]
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,  # [num_lines, dim, width-1]
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    **kwargs,
):
    """Pure-PyTorch causal conv1d (prefill) — replaces the v0.28 Triton
    kernel which silently produces ZEROS on sm_61.

    Returns the activated conv output [dim, total_tokens]; updates
    conv_states in place for the trailing (width-1) inputs of each seq.
    """
    dim, total = x.shape
    width = weight.shape[1]
    dev = x.device
    dt = x.dtype
    out = torch.empty_like(x)
    for seq in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[seq].item())
        end = int(query_start_loc[seq + 1].item())
        if end <= start:
            continue
        cidx = int(cache_indices[seq].item()) if cache_indices is not None else seq
        has_init = bool(has_initial_state[seq].item()) if has_initial_state is not None else False
        if conv_states is not None and has_init:
            state = conv_states[cidx].clone().to(dt)  # [dim, width-1]
        else:
            state = torch.zeros(dim, width - 1, dtype=dt, device=dev)
        for t in range(start, end):
            # HF convention (transformers qwen3_5 causal_conv1d_fn with
            # padding=width-1): out[t] = w[0]*x[t-(w-1)] + ... + w[last]*x[t].
            # state holds the trailing inputs, oldest first: [x_{t-2}, x_{t-1}, x_t]
            # after the update below, so state[j] pairs with weight[:, j].
            acc = x[:, t] * weight[:, width - 1]
            if bias is not None:
                acc = acc + bias
            acc = acc + (state * weight[:, : width - 1]).sum(dim=1)
            y = F.silu(acc)
            out[:, t] = y
            # shift state left, append x[:, t]
            state = torch.cat([state[:, 1:], x[:, t : t + 1]], dim=1)
        if conv_states is not None:
            conv_states[cidx] = state.to(conv_states.dtype)
    return out


def pascal_causal_conv1d_update(
    x: torch.Tensor,          # [num_tokens, dim] (continuous batching)
    conv_states: torch.Tensor,  # [num_lines, dim, width-1]
    weight: torch.Tensor,     # [dim, width]
    bias: torch.Tensor | None,
    activation: str | None = "silu",
    conv_state_indices: torch.Tensor | None = None,
    **kwargs,
):
    """Pure-PyTorch causal conv1d single-step (decode)."""
    n, dim = x.shape
    width = weight.shape[1]
    dev = x.device
    dt = x.dtype
    out = torch.empty_like(x)
    for i in range(n):
        cidx = int(conv_state_indices[i].item()) if conv_state_indices is not None else 0
        state = conv_states[cidx].clone().to(dt)
        # same convention as pascal_causal_conv1d (w[last] pairs with x[t])
        acc = x[i] * weight[:, width - 1]
        if bias is not None:
            acc = acc + bias
        acc = acc + (state * weight[:, : width - 1]).sum(dim=1)
        out[i] = F.silu(acc)
        conv_states[cidx] = torch.cat(
            [state[:, 1:], x[i : i + 1].t()], dim=1
        ).to(conv_states.dtype)
    return out


SOFTPLUS_THRESHOLD = 20.0
L2NORM_EPS = 1e-6


def _softplus_stable(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Stable softplus matching the Triton kernels: where(beta*x<=thr, ...)."""
    bx = beta * x
    sp = torch.where(bx > 0, bx + torch.log1p(torch.exp(-bx)), torch.log1p(torch.exp(bx)))
    sp = torch.where(bx <= SOFTPLUS_THRESHOLD, sp, bx)
    return sp / beta if beta != 1.0 else sp


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """Per-head L2 normalization: x / sqrt(sum(x^2) + eps)."""
    inv = torch.rsqrt(x.square().sum(dim=-1, keepdim=True) + L2NORM_EPS)
    return x * inv


def fused_post_conv_prep(
    conv_output: torch.Tensor,  # [L, qkv_dim]
    a: torch.Tensor,            # [L, HV]
    b: torch.Tensor,            # [L, HV]
    A_log: torch.Tensor,        # [HV]
    dt_bias: torch.Tensor,      # [HV]
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
):
    L = conv_output.shape[0]
    qkv_dim = conv_output.shape[1]
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    assert qkv_dim == 2 * H * K + HV * V, (
        f"qkv_dim={qkv_dim} != 2*H*K + HV*V = {2 * H * K + HV * V}"
    )

    q = conv_output[:, : H * K].view(L, H, K)
    k = conv_output[:, H * K : 2 * H * K].view(L, H, K)
    v = conv_output[:, 2 * H * K :].view(L, HV, V)

    if apply_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)

    x = a + dt_bias  # [L, HV]
    g = -torch.exp(A_log) * _softplus_stable(x)  # [L, HV]
    if output_g_exp:
        g = torch.exp(g)
    beta = torch.sigmoid(b)
    if os.environ.get("PASCAL_GDN_DEBUG") == "1":
        print(f"[GDN-GATE] A_log0={[round(float(v),3) for v in A_log[:4]]} "
              f"dt0={[round(float(v),3) for v in dt_bias[:4]]} "
              f"a0={[round(float(v),3) for v in a[0,:4]]} "
              f"x0={[round(float(v),3) for v in x[0,:4]]}", flush=True)

    return q, k, v, g.float(), beta.float()


def _expand_qk_to_vheads(qk: torch.Tensor, hv: int) -> torch.Tensor:
    """Expand q/k [..., H, D] to [..., HV, D] in GGUF TILED V-head order.

    llama.cpp qwen35 stores V-heads tiled: [G0v0, G1v0, ..., G15v0, G0v1, ...]
    and every V-side weight (alpha/beta, z, out_proj, qkv-V, conv1d-V, A_log,
    dt_bias) is loaded AS-IS in that tiled order (pascal_runtime_patch.py,
    PASCAL-TILED-STRATEGY). Tiled V-head i therefore pairs with k-head (i % H):
    a BLOCK repeat [G0..G{nk-1}, G0..G{nk-1}], NOT a grouped repeat_interleave.

    (v0.18 fixes #7/#8: head_map = arange(H) % Hg; decode i_h = i_hv % H.)
    """
    h = qk.shape[-2]
    if hv == h:
        return qk
    reps = hv // h
    return qk.repeat(*([1] * (qk.ndim - 2)), reps, 1)

def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
):
    """Torch equivalent of fused_sigmoid_gating_delta_rule_update.

    q/k: [B, T, H, K]; v: [B, T, HV, V]; a/b: [B, T, HV] (flat varlen allowed
    via cu_seqlens). Returns (o, final_state) with o = [1, *v.shape].
    """
    if is_kda:
        raise NotImplementedError("pascal_gdn_torch: is_kda not supported")
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    if os.environ.get("PASCAL_GDN_DEBUG") == "1":
        print(
            f"[PGDN-dec] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
            f"a={tuple(a.shape)} b={tuple(b.shape)} A_log={tuple(A_log.shape)} "
            f"dt_bias={tuple(dt_bias.shape)} "
            f"init={None if initial_state is None else tuple(initial_state.shape)} "
            f"cu={None if cu_seqlens is None else tuple(cu_seqlens.shape)} "
            f"ssm_idx={None if ssm_state_indices is None else tuple(ssm_state_indices.shape)} "
            f"use_qk={use_qk_l2norm_in_kernel}",
            flush=True,
        )
    device = k.device

    # v0.28 decode shapes: q/k/v = [B, T, H|HV, K|V] (T=1 per decode step),
    # a/b flat [B*T, HV], cu_seqlens over the flat token axis, o = [1, B*T, HV, V].
    # Normalize everything to the flat [num_tokens, ...] layout.
    if scale is None:
        scale = K ** -0.5
    num_tokens = B * T
    q_hv = _expand_qk_to_vheads(q, HV)
    k_hv = _expand_qk_to_vheads(k, HV)
    if use_qk_l2norm_in_kernel:
        q_hv = _l2norm(q_hv) * scale
        k_hv = _l2norm(k_hv)
    else:
        q_hv = q_hv * scale  # [B, T, HV, K]
    qf = q_hv.reshape(num_tokens, HV, K).float()  # [NT, HV, K]
    kf = k_hv.reshape(num_tokens, HV, K).float()
    vf = v.reshape(num_tokens, HV, V).float()
    af = torch.as_tensor(a, device=device).reshape(num_tokens, HV).float()
    bf = torch.as_tensor(b, device=device).reshape(num_tokens, HV).float()
    dt_bias_f = dt_bias.float()
    A_log_f = A_log.float()

    N = len(cu_seqlens) - 1 if cu_seqlens is not None else num_tokens
    o = torch.empty((1, num_tokens, HV, V), dtype=v.dtype, device=device)

    with torch.no_grad():
        state = initial_state  # [num_slots, HV, V, K]

        def state_idx(n: int, t: int) -> int | None:
            if ssm_state_indices is None:
                return None
            if ssm_state_indices.ndim == 2:
                return int(ssm_state_indices[n, t].item())
            flat = n * T + t
            if ssm_state_indices.numel() > flat:
                return int(ssm_state_indices.reshape(-1)[flat].item())
            return int(ssm_state_indices[n].item())

        for n in range(N):
            if cu_seqlens is not None:
                bos, eos = int(cu_seqlens[n].item()), int(cu_seqlens[n + 1].item())
            else:
                bos, eos = n * T, n * T + T
            if eos <= bos:
                continue
            seq_len = eos - bos

            # initial state for this sequence
            idx0 = state_idx(n, 0) if ssm_state_indices is not None else None
            if state is None:
                h = torch.zeros(HV, V, K, dtype=torch.float32, device=device)
            elif idx0 is None:
                h = state[bos].float().clone()  # state flat [T, HV, V, K]
            elif idx0 <= 0:
                h = torch.zeros(HV, V, K, dtype=torch.float32, device=device)
            else:
                h = state[idx0].float().clone()

            for t in range(bos, eos):
                x = af[t] + dt_bias_f
                g_t = -torch.exp(A_log_f) * _softplus_stable(x, beta)  # [HV]
                h = h * torch.exp(g_t.float())[:, None, None]
                u = vf[t] - torch.einsum("hvk,hk->hv", h, kf[t])  # [HV, V]
                u = u * torch.sigmoid(bf[t])[:, None]
                h = h + u[:, :, None] * kf[t][:, None, :]
                o[0, t] = torch.einsum("hvk,hk->hv", h, qf[t]).to(v.dtype)

                if inplace_final_state and state is not None:
                    idx = state_idx(n, t) if ssm_state_indices is not None else None
                    if idx is None:
                        state[bos + t] = h.to(state.dtype)
                    elif idx > 0:
                        state[idx] = h.to(state.dtype)

    return o, state


class PascalChunkGatedDeltaRule:
    """Duck-typed replacement for ChunkGatedDeltaRule on Pascal.

    forward(q, k, v, g, beta, initial_state, output_final_state, cu_seqlens,
    chunk_indices, chunk_offsets, use_qk_l2norm_in_kernel, core_attn_out)
    -> (o, final_state), o = [1, L, HV, V].
    """

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool = True,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
    ):
        # q/k: [1, L, H, K]; v: [1, L, HV, V]; g/beta: [1, L, HV]
        q = q[0]
        k = k[0]
        v = v[0]
        g = g[0]
        beta = beta[0]
        L, H, K = k.shape
        HV, V = v.shape[1], v.shape[2]
        if os.environ.get("PASCAL_GDN_DEBUG") == "1":
            print(
                f"[PGDN] q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
                f"g={tuple(g.shape)} beta={tuple(beta.shape)} "
                f"init={None if initial_state is None else tuple(initial_state.shape)} "
                f"cu={None if cu_seqlens is None else tuple(cu_seqlens.shape)} "
                f"chunk_off={None if chunk_offsets is None else tuple(chunk_offsets.shape)}",
                flush=True,
            )
        device = k.device
        scale = K ** -0.5

        q_hv = _expand_qk_to_vheads(q, HV)
        k_hv = _expand_qk_to_vheads(k, HV)
        if use_qk_l2norm_in_kernel:
            q_hv = _l2norm(q_hv) * scale
            k_hv = _l2norm(k_hv)
        else:
            q_hv = q_hv * scale

        N = len(cu_seqlens) - 1 if cu_seqlens is not None else 1
        o = torch.empty((1, L, HV, V), dtype=v.dtype, device=device)
        state_dtype = initial_state.dtype if initial_state is not None else torch.float32
        final_state = torch.empty(
            (N, HV, V, K), dtype=state_dtype, device=device
        ) if output_final_state else None

        with torch.no_grad():
            # compute in fp32 (model runs fp16); cast once outside the loop
            vf = v.float()
            kf = k_hv.float()
            qf = q_hv.float()
            for n in range(N):
                if cu_seqlens is not None:
                    bos, eos = int(cu_seqlens[n].item()), int(cu_seqlens[n + 1].item())
                else:
                    bos, eos = 0, L
                # NOTE: initial_state is [N, HV, V, K] here (per-sequence slots,
                # pre-indexed by the caller via prefill_state_indices).
                h = initial_state[n].float().clone() if initial_state is not None else \
                    torch.zeros(HV, V, K, dtype=torch.float32, device=device)

                for t in range(bos, eos):
                    h = h * torch.exp(g[t].float())[:, None, None]
                    u = vf[t] - torch.einsum("hvk,hk->hv", h, kf[t])  # [HV, V]
                    u = u * beta[t].float()[:, None]
                    h = h + u[:, :, None] * kf[t][:, None, :]
                    o[0, t] = torch.einsum("hvk,hk->hv", h, qf[t]).to(v.dtype)

                if output_final_state:
                    final_state[n] = h.to(initial_state.dtype)

        if core_attn_out is not None:
            o_flat = o.squeeze(0).reshape(-1)
            co_flat = core_attn_out.reshape(-1)
            co_flat[: o_flat.numel()].copy_(o_flat)

        return o, final_state

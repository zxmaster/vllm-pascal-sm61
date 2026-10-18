#!/usr/bin/env python3
"""
Verify the PyTorch fallback recurrence against an independent CPU implementation.
Loads the exact tensors dumped from the GPU run and compares.
"""
import torch

print("Loading tensors...")
inputs = torch.load('/tmp/recurrence_inputs.pt', weights_only=True)
outputs = torch.load('/tmp/recurrence_outputs.pt', weights_only=True)

q = inputs['q']
k = inputs['k']
v = inputs['v']
g = inputs['g']
beta = inputs['beta']
initial_state = inputs['initial_state']
cu_seqlens = inputs['cu_seqlens']
scale = inputs['scale']
head_map = inputs['head_map']

print(f"q: {q.shape} {q.dtype}")
print(f"k: {k.shape} {k.dtype}")
print(f"v: {v.shape} {v.dtype}")
print(f"g: {g.shape} {g.dtype}")
print(f"beta: {beta.shape} {beta.dtype}")
print(f"initial_state: {initial_state.shape if initial_state is not None else None}")
print(f"cu_seqlens: {cu_seqlens}")
print(f"scale: {scale}")
print(f"head_map: {head_map}")

gpu_output = outputs['output']
gpu_final_state = outputs['final_state']

print(f"\nGPU output: {gpu_output.shape} {gpu_output.dtype}")
print(f"GPU output stats: mean={gpu_output.float().mean():.6f} std={gpu_output.float().std():.6f}")
print(f"GPU final_state: {gpu_final_state.shape if gpu_final_state is not None else None}")

# --- Independent CPU recurrence ---
print("\n=== Running independent CPU recurrence ===")
B, T_total, Hg, K = q.shape
H = v.shape[2]
V = v.shape[3]

N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
o_cpu = torch.zeros(B, T_total, H, V, dtype=q.dtype)
final_state_cpu = initial_state.clone() if initial_state is not None else None

for seq_idx in range(N):
    if cu_seqlens is not None:
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
    else:
        start = 0
        end = T_total

    if initial_state is not None:
        h = initial_state[seq_idx].float()
    else:
        h = torch.zeros(H, V, K, dtype=torch.float32)

    for t in range(start, end):
        q_t = q[0, t].float()
        k_t = k[0, t].float()
        v_t = v[0, t].float()
        g_t = g[0, t].float()
        beta_t = beta[0, t].float()

        k_exp = k_t[head_map]
        q_exp = q_t[head_map]

        h = h * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
        delta = v_t - torch.bmm(h, k_exp.unsqueeze(-1)).squeeze(-1)
        delta = delta * beta_t.unsqueeze(-1)
        h = h + torch.bmm(delta.unsqueeze(-1), k_exp.unsqueeze(-2))
        o_cpu[0, t] = (
            torch.bmm(h, q_exp.unsqueeze(-1)).squeeze(-1) * scale
        ).to(q.dtype)

        # Print per-timestep stats
        h_stats = f"h: mean={h.mean():.6f} std={h.std():.6f} max={h.abs().max():.6f}"
        o_stats = f"o: mean={o_cpu[0,t].float().mean():.6f} std={o_cpu[0,t].float().std():.6f}"
        print(f"  t={t}: {h_stats} | {o_stats}")

    if final_state_cpu is not None:
        final_state_cpu[seq_idx] = h.to(initial_state.dtype)

print(f"\nCPU output stats: mean={o_cpu.float().mean():.6f} std={o_cpu.float().std():.6f}")

# --- Compare ---
diff = (gpu_output.float() - o_cpu.float()).abs()
print(f"\n=== Comparison ===")
print(f"Max abs diff: {diff.max():.8f}")
print(f"Mean abs diff: {diff.mean():.8f}")
print(f"Relative diff: {(diff / (gpu_output.float().abs() + 1e-8)).mean():.8f}")

# Check per-token comparison
for t in range(T_total):
    d = (gpu_output[0, t].float() - o_cpu[0, t].float()).abs()
    print(f"  Token {t}: max_diff={d.max():.8f}, mean_diff={d.mean():.8f}, "
          f"gpu_mean={gpu_output[0,t].float().mean():.6f}, cpu_mean={o_cpu[0,t].float().mean():.6f}")

# Final state comparison
if gpu_final_state is not None and final_state_cpu is not None:
    fs_diff = (gpu_final_state.float() - final_state_cpu.float()).abs()
    print(f"\nFinal state: max_diff={fs_diff.max():.8f}, mean_diff={fs_diff.mean():.8f}")

# --- Also try with FP32 throughout (no fp16 cast) ---
print("\n=== FP32-only CPU recurrence (no dtype cast) ===")
o_f32 = torch.zeros(B, T_total, H, V, dtype=torch.float32)
for seq_idx in range(N):
    if cu_seqlens is not None:
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
    else:
        start = 0
        end = T_total

    if initial_state is not None:
        h = initial_state[seq_idx].float()
    else:
        h = torch.zeros(H, V, K, dtype=torch.float32)

    for t in range(start, end):
        q_t = q[0, t].float()
        k_t = k[0, t].float()
        v_t = v[0, t].float()
        g_t = g[0, t].float()
        beta_t = beta[0, t].float()

        k_exp = k_t[head_map]
        q_exp = q_t[head_map]

        h = h * torch.exp(g_t).unsqueeze(-1).unsqueeze(-1)
        delta = v_t - torch.bmm(h, k_exp.unsqueeze(-1)).squeeze(-1)
        delta = delta * beta_t.unsqueeze(-1)
        h = h + torch.bmm(delta.unsqueeze(-1), k_exp.unsqueeze(-2))
        o_f32[0, t] = torch.bmm(h, q_exp.unsqueeze(-1)).squeeze(-1) * scale

diff_f32 = (gpu_output.float() - o_f32).abs()
print(f"FP32 vs GPU: max_diff={diff_f32.max():.8f}, mean_diff={diff_f32.mean():.8f}")

# Show actual tensor values for the last token
T = T_total
print(f"\n=== Last token (t={T-1}) output comparison ===")
print(f"GPU  [head0, :8]: {gpu_output[0, T-1, 0, :8].tolist()}")
print(f"CPU  [head0, :8]: {o_cpu[0, T-1, 0, :8].tolist()}")
print(f"F32  [head0, :8]: {o_f32[0, T-1, 0, :8].tolist()}")
print(f"GPU  [head31,:8]: {gpu_output[0, T-1, 31, :8].tolist()}")
print(f"CPU  [head31,:8]: {o_cpu[0, T-1, 31, :8].tolist()}")
print(f"F32  [head31,:8]: {o_f32[0, T-1, 31, :8].tolist()}")

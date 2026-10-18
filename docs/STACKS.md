# Stack map — what is what

This document removes the naming confusion around the project. There are **two
repositories** and **three engine-version containers**; they answer different
questions.

## 1. Two repositories

| Repository | Local path | Role |
|---|---|---|
| `zxmaster/vllm-pascal` (private) | `~/Projects/vllm-pascal` | The working stack: every build patch, runtime patch, model profile, validation script and local tuning we actually run. This is the "combine" that ingests any model supported by the GGUF plugin. |
| `zxmaster/vllm-pascal-sm61` (public) | `~/Projects/vllm-pascal-sm61` | The community-facing glue layer: Dockerfiles, build/runtime patches, compose files, docs. No private model profiles, no secrets, no local paths. Synced from the private repo with `scripts/sync-public.sh`. |

The public repository is **not a runtime**. It is a clean export of the reusable
layer; nothing in it is required to run the private stack.

## 2. Engine versions vs. model profiles

Container names encode the **vLLM engine version**, not the model:

| Container | Image | Engine | Status |
|---|---|---|---|
| `vllm-openai-v031` | `local/vllm-openai:sm61-v031` | vLLM 0.31.1.dev0 | **Current / primary** |
| `vllm-openai-v029` | `local/vllm-openai:sm61-v029` | vLLM 0.29.1.dev0 | Reserve / rollback |
| `vllm-openai-v028` | `local/vllm-openai:sm61-v028` | vLLM 0.28.x | Deep reserve |

A container is just a slot. The **model** is chosen by the profile file passed as
`VLLM_ENV_FILE`:

| Profile | Model | Notes |
|---|---|---|
| `.env.v028`, `.env.v029`, `.env.v031` | `Qwen3.5-4B-IQ4_XS` (served as `Qwen3.5-4B-GGUF`) | Same model in all three; only the engine differs. |
| `.env.qwen38-27b` | `Qwen3.8-27B-UD-IQ2_M` | **Not working yet** — CUDA OOM on 4 GB, CPU offload does not engage. Tracked in `PLAN.md`. |
| `.env.minicpm5` | `MiniCPM5-2B` | Separate small-model slot. |

So "`vllm-qwen38` pointing at `v028`" was never a model binding: `v028` runs
whatever profile it was started with, and it was started with the 4B profile.

## 3. One GPU, one active container

The GTX 1050 Mobile has 4 GB of VRAM. Only one vLLM container may hold the GPU at a
time. The gateway stops the active backend before starting another (swap), and
Sablier groups (`vllm28`, `vllm29`, `vllm31`) provide scale-to-zero.

## 4. How the gateway sees it

The AI gateway keeps a single model list in `config.yaml`:

- `vllm-qwen35` → `vllm-openai-v031` → `Qwen3.5-4B-GGUF` (current).
- The former `vllm-qwen38` entry is **disabled**: the 27B profile does not start,
  and the old container it pointed at was running the 4B profile. Re-enable it only
  after `PLAN.md` (27B on 4 GB) is closed.

## 5. Why GGUF, and why not every quantization

vLLM does **not** require FP16 weights: it loads quantized GGUF through
`vllm-gguf-plugin`. The plugin supports the IQ1/IQ2/IQ3/IQ4 families and Q2_K–Q8_1.
Upstream does **not** implement `Q1_0` / `TQ1_0` / `TQ2_0` — the quantizations used by
Bonsai-27B. We added `Q1_0` ourselves: see `docs/Q1_0-BONSAI.md` (Triton dequant +
fused GEMM, bit-exact against the llama.cpp reference). The remaining limit is
hardware: Bonsai's 3.54 GiB of 1-bit weights do not fit into a 4 GB card next to the
CUDA context — a 6 GB GPU or larger is required.

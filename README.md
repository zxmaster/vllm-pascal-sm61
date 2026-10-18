# vllm-pascal

**Run vLLM on Pascal-generation GPUs (compute capability 6.1, e.g. GTX 1050/1060/1080)
with GGUF quantization and TurboQuant KV cache.**

This repository contains the **glue layer** — build patches, runtime patches,
Dockerfiles, compose files and validation scripts — that makes modern vLLM work on
`sm_61` hardware with 4 GB of VRAM. It is not a fork of vLLM; upstream sources are
fetched at build time and patched in a controlled, fail-fast way.

> **Status:** experimental, tuned for a single 4 GB GTX 1050 Mobile. It works and is
> validated (battery 12/12, 28k-token prefill), but it is not a general-purpose
> vLLM distribution. Expect to read the docs before running it.

## What is included

| Path | Purpose |
|---|---|
| `docker/sm61/` | Dockerfiles (source cache + image), build-time patches, entrypoint |
| `scripts/` | build scripts, start script, runtime patches, validation/battery tools |
| `compose.sm61-v031.yaml` | current primary stack (vLLM 0.31, TurboQuant KV, GGUF) |
| `compose.sm61-v029.yaml` | previous generation, kept as reserve/rollback |
| `compose.sm61-v028.yaml` | older generation, deep reserve |
| `docs/UPSTREAM.md` | how upstream syncs work, anchors, hot spots, upgrade procedure |
| `docs/FIXES.md` | registry of every fix and the reason it exists |
| `docs/MIGRATION.md` | how to deploy on a fresh machine without rebuilding |

## Why patches are needed

Pascal (sm_61) is below the minimum compute capability supported by current
PyTorch/vLLM kernels. The build and runtime patches address, among others:

- FlashAttention extension disabled; SDPA fallbacks for attention;
- Triton kernels with `sm_70+` atomics replaced with torch/CUDA-CAS equivalents;
- GatedDeltaNet (Qwen3.5) fused kernels replaced with a pure-torch path;
- GGUF weight loading fixes (qwen3.5 normalization, conv1d layout, llama Q/K permutation);
- TurboQuant prefill/continuation memory fixes so long contexts fit in 4 GB.

Each patch fails loudly (`SystemExit: failed to locate anchor`) if the upstream code
changed, instead of silently patching the wrong thing.

## Requirements

- NVIDIA GPU with compute capability 6.1 and ~4 GB VRAM (developed on GTX 1050 Mobile);
- Docker with NVIDIA container runtime;
- ~50 GB free disk for the image and build cache (the image itself is ~46 GB);
- A GGUF model (the reference setup uses Qwen3.5-4B IQ4_XS).

### Torch (built from source)

Current PyTorch wheels do not support `sm_61`, so the build compiles a matching torch
from source. **This is the longest part of the build** (hours on a laptop); it is
cached in the source-cache image, so subsequent builds reuse it.

The torch artifact is intentionally **not** shipped in this repository. If you already
have a compatible `sm_61` torch package, place it under
`artifacts/torch/site-packages/current/` (containing `torch/`, `torchgen/`,
`functorch/` and `torch-*.dist-info/`) and set `SM61_USE_TORCH_ARTIFACT=1` to skip the
torch build. The default `.env.example` keeps it at `0` (build from source).

## Build and run (short version)

```bash
# 1. Fetch sources and build the image (hours on a laptop; torch is reused via artifact)
SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh
SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh

# 2. Start the stack
VLLM_ENV_FILE=.env.v031 docker compose -f compose.sm61-v031.yaml up -d

# 3. First request loads the model (lazy, ~3 min on this hardware)
curl -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/v1/models
```

See `docs/MIGRATION.md` for the full checklist, volume layout, and rollback steps.
`docs/UPSTREAM.md` explains how to move to a newer vLLM tag safely.

## Validation

The repository ships the same checks used to validate releases:

- `scripts/v028/battery_full.py` — 12 short prompts (expected 12/12);
- `scripts/minicpm5/longctx.py` — needle-in-a-haystack long-prefill test;
- `scripts/smoke-test.sh` — endpoint smoke test.

Current reference numbers (v0.31.0, Qwen3.5-4B IQ4_XS, 32k context, TurboQuant):
KV cache pool **45,251 tokens**; battery **12/12**; long prefill **28,453 tokens in
334 s** (previous generation: 630 s).

## License and upstream

Our code is released under the **Apache License 2.0** (see `LICENSE`). It patches
vLLM at build time; vLLM itself is not redistributed here — the build downloads the
upstream tag (`SM61_VLLM_REF`) from the official repository. vLLM is developed by the
[vLLM project](https://github.com/vllm-project/vllm) under Apache-2.0; see `NOTICE`.

TurboQuant is used natively from vLLM 0.28+ — no external GPL submodule is required
or included. Model weights are not included; download them under their own license.

This project is not affiliated with or endorsed by the vLLM project. Pascal GPUs are
below vLLM's supported compute capability; use at your own risk.

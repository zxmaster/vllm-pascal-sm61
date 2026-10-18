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
| `docs/RELEASE-TORCH.md` | what the prebuilt `sm_61` torch wheel is, how it was built and verified |

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

### Torch (prebuilt wheel available)

Current PyTorch wheels do not support `sm_61`: they import fine and then fail with
`CUDA error: no kernel image is available for execution on the device`. The build
therefore needs a torch compiled with `TORCH_CUDA_ARCH_LIST=6.1` — **the longest step of
the build** (hours on a laptop).

A prebuilt wheel for exactly this case is published in
[Releases](https://github.com/zxmaster/vllm-pascal-sm61/releases) — PyTorch 2.13.0,
CUDA 12.9, `sm_61`, CPython 3.12:

```bash
./scripts/fetch-torch-artifact.sh     # downloads, verifies SHA-256, unpacks
```

It unpacks into `artifacts/torch/site-packages/current/` (containing `torch/`, `torchgen/`,
`functorch/`, `torch-*.dist-info/`); `build-sm61-image.sh` detects that tree and sets
`SM61_USE_TORCH_ARTIFACT=1` by itself, so the torch build is skipped. The same wheel is
pip-installable — from a local file or straight from the release URL:

```bash
pip install --no-deps https://github.com/zxmaster/vllm-pascal-sm61/releases/download/torch-sm61-v2.13.0-r1/torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl
```

Provenance, contents and verification steps: [`docs/RELEASE-TORCH.md`](docs/RELEASE-TORCH.md).
If you prefer to build torch yourself, nothing else is needed — leave
`artifacts/torch/site-packages/current/` empty and the build compiles torch from source
(`SM61_USE_TORCH_ARTIFACT=0`), which is also how a new artifact is produced:
`scripts/extract-sm61-torch-artifact.sh` → `scripts/package-torch-wheel.py`.

## From a fresh clone to the first token

Prerequisites: x86-64 Linux, NVIDIA GPU with compute capability **6.1**, driver, Docker with
the `nvidia` runtime (nvidia-container-toolkit), ~150 GB free disk. `prepare-host.sh`
checks all of that and creates the directory layout, the external `intranet_net` network and
`.env.v031` for you.

```bash
git clone https://github.com/zxmaster/vllm-pascal-sm61 && cd vllm-pascal-sm61

./scripts/prepare-host.sh            # checks GPU/driver/docker/disk, makes dirs + net + .env.v031
./scripts/fetch-torch-artifact.sh    # prebuilt torch wheel: download, verify, unpack
./scripts/prepare-model.sh           # GGUF weights + tokenizer + config.json (not redistributed here)

SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh   # clones upstream torch + vLLM
SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh          # compiles vLLM for sm_61

VLLM_ENV_FILE=.env.v031 docker compose -f compose.sm61-v031.yaml up -d

# the backend loads the model lazily on the first request (~3 min on a GTX 1050)
curl -s -H 'Authorization: Bearer EMPTY' http://127.0.0.1:8000/v1/models
```

Typical wall-clock on a 4-core laptop (i7-7700HQ, MAX_JOBS=4), reference run:

| Step | Time | Notes |
|---|---|---|
| `prepare-host.sh` | seconds | idempotent |
| `fetch-torch-artifact.sh` | ~2 min | 288 MB download + unpack |
| `prepare-model.sh` | 2–40 min | 2.3 GiB GGUF + tokenizer; network-bound (37 min at 1.3 MB/s) |
| `build-sm61-source-cache.sh` | 10–30 min | git clones of torch + vLLM (submodules) |
| `build-sm61-image.sh` | **~40 min** | vLLM compiled for `sm_61`; torch build skipped |
| first request | ~3 min | lazy load, KV pool printed in the log |

The torch compile — the step that used to dominate — is gone: the wheel replaces several
hours of CPU time. If you skip `fetch-torch-artifact.sh`, the same build compiles torch from
source instead.

`prepare-model.sh` also pins `architectures` in `config.json` to `Qwen3_5ForCausalLM`: the
stacks run the text-only path (`--language-model-only`), while the upstream file declares the
multimodal wrapper. Everything else in that file stays upstream's (`--keep-architectures`
leaves it untouched).

Expected log markers from a healthy start (`docker logs vllm-openai-v031`):

```
[Pascal] runtime patches applied: ['sdpa', 'gguf_adapter', 'gguf_quant', 'tq_backend', ...]
GPU KV cache size: 45,251 tokens          # MAX_MODEL_LEN=32768, GPU_MEMORY_UTILIZATION=0.85
Application startup complete.
```

Say hello:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Authorization: Bearer EMPTY' -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-4B-GGUF","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":32}'
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

The prebuilt `sm_61` torch wheel published in Releases is a **modified build of PyTorch**,
redistributed under PyTorch's own licenses (the wheel carries the complete license set in
`torch-*.dist-info/licenses/`, including every bundled third-party license). It contains no
NVIDIA runtime libraries and no GPL code. See `THIRD-PARTY-NOTICES.md`.

TurboQuant is used natively from vLLM 0.28+ — no external GPL submodule is required
or included. Model weights are not included; download them under their own license.

This project is not affiliated with or endorsed by the vLLM project. Pascal GPUs are
below vLLM's supported compute capability; use at your own risk.

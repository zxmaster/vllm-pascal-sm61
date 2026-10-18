# Third-party notices

This repository contains only our own "glue" layer (build patches, runtime patches,
Dockerfiles, compose files, scripts, docs), licensed under **Apache-2.0** — see `LICENSE`.

Nothing third-party is vendored into git. Everything else is either downloaded at build
time from its upstream repository or published as a binary artifact under its own license,
as described below.

## vLLM — Apache-2.0

Served by build time (`SM61_VLLM_REF`), neither vendored nor redistributed here.
Upstream: <https://github.com/vllm-project/vllm>. The build applies patches from
`docker/sm61/patch-vllm-sm61-*.py`; those modifications are visible in this repository.

## TurboQuant — GPL-3.0 (not included)

The historical custom TurboQuant hook lived in an `external/turboquant` submodule
(GPL-3.0) and belonged to the v0.18 era only. It is **not** part of this repository and is
not used by the current stacks: TurboQuant KV cache is used natively from vLLM 0.28+
(Apache-2.0).

## Prebuilt PyTorch wheel (release assets)

`torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl` is a **modified, rebuilt** PyTorch:

- upstream: `pytorch/pytorch`, tag `v2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb`;
- modifications: the three small build-time patches in `docker/sm61/patch-torch-fabric.py`
  (NVML fabric path disabled for older `nvml.h`; upstream `__CUDA_NO_HALF_OPERATORS__` fix
  mirrored for `symm_mem` sources) and `TORCH_CUDA_ARCH_LIST=6.1`;
- license: PyTorch's own license set —
  `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`,
  declared in the wheel's metadata. The complete text for PyTorch and for **every bundled
  third-party component** (MKL, oneDNN, NNPACK, FXdiv, sleef, protobuf, pybind11, LLVM
  OpenMP, CUTLASS, flash-attention, gloo, kineto, XNNPACK, …) ships inside the wheel at
  `torch-*.dist-info/licenses/`. Redistributing the wheel therefore keeps all required
  notices with it;
- **not** included: NVIDIA CUDA runtime/redistributable libraries. The wheel links against
  a CUDA 12.x installation that you provide yourself (driver + `nvidia-*` packages or the
  CUDA toolkit), exactly like the official PyTorch wheels.

The wheel is a convenience build for hardware that official wheels no longer cover
(`sm_61`). It is provided as-is, without warranty, and is not affiliated with or endorsed by
the PyTorch or vLLM projects.

## Model weights

Not included. Download models (e.g. GGUF quantizations of Qwen3.5-4B, MiniCPM5-2B) from
their publishers and follow their licenses.

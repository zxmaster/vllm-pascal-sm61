# Prebuilt PyTorch for Pascal (compute capability 6.1)

This release ships a **ready-to-use PyTorch wheel built for `sm_61`** (GTX 10-series /
Pascal), so you do not have to compile torch yourself.

Official PyTorch wheels contain SASS for `sm_75` and newer only. On Pascal they import
fine and then fail on the first GPU kernel:

```
CUDA error: no kernel image is available for execution on the device
```

Building torch from source with `TORCH_CUDA_ARCH_LIST=6.1` fixes that, but takes **hours**
on a laptop — and it is the single longest step of the vLLM build. This wheel is the
result of that build, packaged so that it can simply be downloaded.

## Artifacts

| File | What it is |
|---|---|
| `torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl` | PyTorch 2.13.0 (upstream tag `v2.13.0`), CUDA 12.9, `sm_61` only, CPython 3.12 |
| `SHA256SUMS` | checksum of the wheel |
| `build-info.json` | what it was built from, how, and how it was verified |

## Use it — option A: Docker build of vLLM (the fast path)

```bash
git clone https://github.com/zxmaster/vllm-pascal-sm61 && cd vllm-pascal-sm61

./scripts/prepare-host.sh              # GPU/docker/disk checks, dirs, intranet_net, .env.v031
./scripts/fetch-torch-artifact.sh      # downloads, verifies sha256, unpacks
./scripts/prepare-model.sh             # GGUF + tokenizer + config.json
SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh
SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh   # torch build skipped
```

`fetch-torch-artifact.sh` unpacks the wheel into
`artifacts/torch/site-packages/current/`, which is the layout the Dockerfile expects;
`build-sm61-image.sh` then detects it and sets `SM61_USE_TORCH_ARTIFACT=1` by itself.

## Use it — option B: plain Python

```bash
# straight from this release (verified: pip accepts the release URL directly)
pip install --no-deps https://github.com/zxmaster/vllm-pascal-sm61/releases/download/torch-sm61-v2.13.0-r1/torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl
# or from a local file
pip install --no-deps torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl

python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
# 2.13.0a0+gitcf30153 ['sm_61']
```

The wheel carries no CUDA runtime of its own: you need an NVIDIA driver and the CUDA 12.x
userspace libraries (`nvidia-*` pip packages or a CUDA 12.9 toolkit), exactly as with the
official wheels. It is a drop-in replacement in a CUDA 12.9 Python 3.12 environment.

## What is inside

- `torch/`, `torchgen/`, `functorch/` and the `torch-2.13.0a0+gitcf30153.dist-info/`
  metadata, with the `torchrun` console script;
- the full upstream license set (`dist-info/licenses/`, incl. every bundled third-party
  license) — the same set the official wheel ships;
- **no** NVIDIA runtime libraries (they are not redistributed here), and no GPL code.

Dropped from the repacked tree: `__pycache__`/`*.pyc` produced while the image was built,
and pip's `direct_url.json` / `REQUESTED` / `INSTALLER`.

## Why a wheel and not a tarball

A wheel is a zip with a `.dist-info/` directory — it installs with `pip`, unpacks with any
unzip tool, has a standard name for pinning in requirements files, and keeps the license
files in their canonical location. One artifact serves both audiences: `pip install` for a
venv, and unpacking into the Docker build context.

`RECORD` is regenerated for the repacked file, so `pip uninstall` / `pip check` stay honest.
Indexes contain no `torch==2.13.0a0+gitcf30153`, so pip resolves the local wheel by path —
this is a prebuilt, self-contained artifact, not a published package on PyPI.

## How it was verified

- `pip install --no-deps --force-reinstall` in a clean `vllm/vllm-openai:v0.29.0-cu129`
  container: installs, imports, `torch.cuda.get_arch_list() == ['sm_61']`, `torchrun` link
  created, CPU kernels run, NCCL 2.29.7 reported;
- `torch/`, `torchgen/`, `functorch/` compare **byte-identical** to the tree inside the
  running `sm61-v031` image, i.e. to the torch that produced the reference numbers below;
- that image, on a 4 GB GTX 1050 Mobile, validated vLLM 0.31 with this exact torch:
  battery **12/12**, KV pool **45,251 tokens**, **28,453-token** prefill in 334 s.

## Built how (reproducibility)

| Item | Value |
|---|---|
| Upstream | `pytorch/pytorch` tag `v2.13.0`, commit `cf30153c4c131c8164ee7798e5022d810682e2cb` |
| Arch | `TORCH_CUDA_ARCH_LIST=6.1`, `CMAKE_CUDA_ARCHITECTURES=61` |
| CUDA | toolkit 12.9, NCCL 2.29.7 |
| Python / glibc | CPython 3.12, glibc 2.39 |
| Base | `vllm/vllm-openai:v0.29.0-cu129` |
| Local patches | `docker/sm61/patch-torch-fabric.py` (NVML fabric path disabled for older `nvml.h`; upstream `__CUDA_NO_HALF_OPERATORS__` fix mirrored for `symm_mem`) |
| vLLM in the same image | `v0.31.0` |

To rebuild it yourself (hours of CPU):

```bash
SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh
SM61_USE_TORCH_ARTIFACT=0 SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh
sh ./scripts/extract-sm61-torch-artifact.sh         # -> artifacts/torch/site-packages/
python3 ./scripts/package-torch-wheel.py             # -> dist/*.whl
```

## License

This wheel is a **modified build of PyTorch**, redistributed under PyTorch's own licenses
(see `torch-…dist-info/licenses/` inside the wheel). The modifications are the small
build-time patches listed above; the patch sources live in this repository. PyTorch is
developed by the PyTorch project (https://pytorch.org). This project is not affiliated with
or endorsed by the PyTorch or vLLM projects.

## Limitations

- `sm_61` only — this wheel will **not** run on newer GPUs, and provides no forward
  compatibility (no PTX JIT): use official wheels for `sm_75+`;
- CPython 3.12 on Linux x86-64, CUDA 12.x;
- built for the CUDA userspace that ships in the vLLM images; other CUDA 12 minor versions
  may require matching `nvidia-*` packages.

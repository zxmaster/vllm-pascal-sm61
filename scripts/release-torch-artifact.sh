#!/bin/sh
# release-torch-artifact.sh — publish the prebuilt sm_61 torch wheel to GitHub Releases.
#
# The public repo (zxmaster/vllm-pascal-sm61) ships the build layer but not the
# torch binary: current PyTorch wheels carry SASS for sm_75+ only, so Pascal users
# must otherwise spend hours compiling torch. This script packages the extracted
# torch tree from a working image into a pip-installable wheel and publishes it as
# a GitHub Release asset, next to a checksum file and a machine-readable build
# record (where it came from, how it was built, how it was verified).
#
# Usage:
#   sh scripts/release-torch-artifact.sh                 # package + publish
#   sh scripts/release-torch-artifact.sh --dry-run       # package only, no upload
#   sh scripts/release-torch-artifact.sh --rebuild       # re-package an existing wheel
#   TAG=torch-sm61-v2.13.0-r2 sh scripts/release-torch-artifact.sh
#
# Env:
#   REPO         public repository            (default zxmaster/vllm-pascal-sm61)
#   TAG          release tag                  (default torch-sm61-v2.13.0-r1)
#   SRC          extracted torch tree         (default artifacts/torch/site-packages/current)
#   OUT          artifact directory           (default dist)
#   SOURCE_IMAGE image the artifact came from (default local/vllm-openai:sm61-v031)
#   VERIFY_IMAGE image used for the pip smoke test (default vllm/vllm-openai:v0.29.0-cu129)
set -eu

REPO=${REPO:-zxmaster/vllm-pascal-sm61}
TAG=${TAG:-torch-sm61-v2.13.0-r1}
SRC=${SRC:-artifacts/torch/site-packages/current}
OUT=${OUT:-dist}
SOURCE_IMAGE=${SOURCE_IMAGE:-local/vllm-openai:sm61-v031}
VERIFY_IMAGE=${VERIFY_IMAGE:-vllm/vllm-openai:v0.29.0-cu129}
VLLM_REF=${VLLM_REF:-v0.31.0}

REBUILD=0
DRY=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."
mkdir -p "$OUT"

wheel_name() {
  ls "$OUT"/torch-*-cp312-cp312-linux_x86_64.whl 2>/dev/null | head -1 || true
}

wheel=$(wheel_name)
if [ -z "$wheel" ] || [ "$REBUILD" = "1" ]; then
  echo "== packaging wheel from $SRC"
  python3 scripts/package-torch-wheel.py --src "$SRC" --out "$OUT"
  wheel=$(wheel_name)
fi
[ -n "$wheel" ] || { echo "wheel not found in $OUT" >&2; exit 1; }
wheel_base=$(basename "$wheel")
wheel_size=$(wc -c <"$wheel")
wheel_sha=$(sha256sum "$wheel" | cut -d' ' -f1)

echo "== wheel: $wheel_base ($wheel_size bytes)"
echo "== sha256: $wheel_sha"

# --- build record ----------------------------------------------------------
torch_version=$(python3 - "$wheel" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as zf:
    meta = zf.read(next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))).decode()
for line in meta.splitlines():
    if line.startswith("Version: "):
        print(line[len("Version: "):].strip())
        break
PY
)

git_field() { # git_field <image> <repo-path> <format>
  docker run --rm --entrypoint git "$1" -C "$2" log -1 --format="$3" 2>/dev/null | tr -d '\r' || true
}
TORCH_COMMIT=${TORCH_COMMIT:-$(git_field local/vllm-source-cache:sm61-v031 /opt/src/pytorch '%H')}
TORCH_COMMIT_DATE=${TORCH_COMMIT_DATE:-$(git_field local/vllm-source-cache:sm61-v031 /opt/src/pytorch '%ci')}
TORCH_UPSTREAM_TAG=${TORCH_UPSTREAM_TAG:-$(docker run --rm --entrypoint git local/vllm-source-cache:sm61-v031 -C /opt/src/pytorch describe --tags 2>/dev/null | tr -d '\r' || true)}

python3 - "$OUT/build-info.json" "$wheel_base" "$wheel_sha" "$wheel_size" \
         "$torch_version" "${TORCH_UPSTREAM_TAG:-v2.13.0}" "${TORCH_COMMIT:-unknown}" \
         "${TORCH_COMMIT_DATE:-unknown}" "$SOURCE_IMAGE" "$VLLM_REF" "$TAG" <<'PY'
import json, sys, datetime, pathlib
out, wheel, sha, size, version, tag, commit, commit_date, image, vllm_ref, release = sys.argv[1:12]
info = {
    "schema": 1,
    "artifact": "PyTorch wheel built for compute capability 6.1 (Pascal)",
    "release_tag": release,
    "wheel": wheel,
    "sha256": sha,
    "size_bytes": int(size),
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "python": {"wheel_tag": "cp312-cp312-linux_x86_64", "abi": "CPython 3.12", "glibc": "2.39"},
    "torch": {
        "version": version,
        "upstream_tag": tag,
        "commit": commit,
        "commit_date": commit_date,
        "license": "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT",
    },
    "cuda": {"toolkit": "12.9", "arch_list": ["sm_61"], "nccl": "2.29.7"},
    "build": {
        "base_image": "vllm/vllm-openai:v0.29.0-cu129",
        "artifact_source_image": "local/vllm-openai:sm61-v028",
        "verified_against_image": image,
        "vllm_ref": vllm_ref,
        "env": {"TORCH_CUDA_ARCH_LIST": "6.1", "CMAKE_CUDA_ARCHITECTURES": "61"},
        "patches": [
            {
                "file": "docker/sm61/patch-torch-fabric.py",
                "changes": [
                    "c10/cuda/driver_api.h: disable the NVML fabric member declaration",
                    "c10/cuda/PeerToPeerAccess.cpp: stub get_fabric_access for images whose nvml.h predates the fabric API",
                    "caffe2/CMakeLists.txt: mirror upstream -U__CUDA_NO_HALF_OPERATORS__ for symm_mem sources",
                ],
            }
        ],
    },
    "usage": {
        "pip": "pip install --no-deps <wheel>",
        "docker_build": "scripts/fetch-torch-artifact.sh -> artifacts/torch/site-packages/current (SM61_USE_TORCH_ARTIFACT=1)",
    },
    "verification": {
        "pip_install": "clean base image, pip install --no-deps --force-reinstall",
        "arch_list": ["sm_61"],
        "torchrun_entry_point": "created",
        "binary_identity": "torch/, torchgen/, functorch/ compare byte-identical to the running sm61-v031 image",
        "gpu_validation": "battery 12/12 + 28,453-token prefill on GTX 1050 Mobile with this exact tree",
    },
}
pathlib.Path(out).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
print(f"== build-info: {out}")
PY

# --- checksums -------------------------------------------------------------
(
  cd "$OUT"
  sha256sum "$wheel_base" >SHA256SUMS
)
echo "== checksums: $OUT/SHA256SUMS"

if [ "$DRY" = "1" ]; then
  echo "(dry-run: release not published)"
  cat "$OUT/SHA256SUMS"
  exit 0
fi

# --- publish ---------------------------------------------------------------
notes_file=docs/RELEASE-TORCH.md
[ -f "$notes_file" ] || { echo "missing release notes: $notes_file" >&2; exit 1; }

if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "== release $TAG exists -> uploading assets"
  gh release upload "$TAG" --repo "$REPO" --clobber \
    "$wheel" "$OUT/SHA256SUMS" "$OUT/build-info.json"
  gh release edit "$TAG" --repo "$REPO" --notes-file "$notes_file"
else
  echo "== creating release $TAG"
  gh release create "$TAG" --repo "$REPO" \
    --title "Prebuilt sm_61 torch $torch_version (CUDA 12.9, Python 3.12)" \
    --notes-file "$notes_file" \
    "$wheel" "$OUT/SHA256SUMS" "$OUT/build-info.json"
fi

echo "== published: https://github.com/$REPO/releases/tag/$TAG"

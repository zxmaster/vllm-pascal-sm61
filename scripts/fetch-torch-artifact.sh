#!/bin/sh
# fetch-torch-artifact.sh — download the prebuilt sm_61 torch and put it where the build expects it.
#
# Official PyTorch wheels ship SASS for sm_75+ only, so on Pascal (compute capability
# 6.1) the build has to compile torch from source — hours of CPU. This script downloads
# the wheel we built for that case (5 min on a laptop, seconds on a server), verifies
# its SHA-256, and unpacks it into the layout the Dockerfile consumes:
#
#   artifacts/torch/site-packages/current/{torch,torchgen,functorch,torch-*.dist-info}
#
# build-sm61-image.sh then detects that tree and sets SM61_USE_TORCH_ARTIFACT=1.
#
# Usage:
#   ./scripts/fetch-torch-artifact.sh              # download + verify + unpack
#   ./scripts/fetch-torch-artifact.sh --pip        # ... and pip install --no-deps it
#   ./scripts/fetch-torch-artifact.sh --verify-only
#   ./scripts/fetch-torch-artifact.sh --force      # re-download even if cached
#   SM61_TORCH_URL=https://mirror/… ./scripts/fetch-torch-artifact.sh
set -eu

REPO=${SM61_TORCH_REPO:-zxmaster/vllm-pascal-sm61}
TAG=${SM61_TORCH_TAG:-torch-sm61-v2.13.0-r1}
WHEEL=${SM61_TORCH_WHEEL:-torch-2.13.0a0+gitcf30153-cp312-cp312-linux_x86_64.whl}
SHA256=${SM61_TORCH_SHA256:-cf11579353c563e13a4910f0a753ba2291a9983417b0883fae5d59b1c85450fd}
URL=${SM61_TORCH_URL:-https://github.com/$REPO/releases/download/$TAG/$WHEEL}
DEST=${SM61_TORCH_DEST:-artifacts/torch/site-packages/current}
CACHE=${SM61_TORCH_CACHE:-dist}

PIP=0
VERIFY_ONLY=0
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --pip) PIP=1 ;;
    --verify-only) VERIFY_ONLY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

mkdir -p "$CACHE"
wheel_path="$CACHE/$WHEEL"

if [ "$FORCE" = "1" ]; then
  rm -f "$wheel_path"
fi

if [ -f "$wheel_path" ] && [ "$(sha256_of "$wheel_path")" = "$SHA256" ]; then
  echo "== cached and verified: $wheel_path"
else
  [ "$VERIFY_ONLY" = "1" ] && { echo "not downloaded: $wheel_path" >&2; exit 1; }
  echo "== downloading $WHEEL ($TAG)"
  echo "   $URL"
  curl -fL --retry 5 --retry-delay 5 --retry-all-errors --connect-timeout 20 \
       --progress-bar -o "$wheel_path.part" "$URL"
  mv "$wheel_path.part" "$wheel_path"
  actual=$(sha256_of "$wheel_path")
  if [ "$actual" != "$SHA256" ]; then
    echo "SHA-256 mismatch — refusing to use this file:" >&2
    echo "  expected $SHA256" >&2
    echo "  actual   $actual" >&2
    rm -f "$wheel_path"
    exit 1
  fi
  echo "== sha256 ok: $SHA256"
fi

if [ "$VERIFY_ONLY" = "1" ]; then
  echo "== artifact verified, nothing else to do"
  exit 0
fi

if [ "$PIP" = "1" ]; then
  echo "== pip install --no-deps $wheel_path"
  python3 -m pip install --no-deps "$wheel_path"
fi

# Unpack as a wheel (zip) into the build-context layout the Dockerfile reads.
echo "== unpacking into $DEST"
rm -rf "$DEST"
mkdir -p "$DEST"
python3 - "$wheel_path" "$DEST" <<'PY'
import sys, zipfile
wheel, dest = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(wheel) as zf:
    bad = zf.testzip()
    if bad is not None:
        raise SystemExit(f"corrupt member in wheel: {bad}")
    zf.extractall(dest)
    print(f"   extracted {len(zf.namelist())} members")
PY

for pkg in torch torchgen functorch; do
  [ -d "$DEST/$pkg" ] || { echo "unpacked tree lacks $pkg/" >&2; exit 1; }
done
ls -d "$DEST"/torch-*.dist-info >/dev/null 2>&1 || { echo "unpacked tree lacks torch-*.dist-info" >&2; exit 1; }

size=$(du -sh "$DEST" | cut -f1)
echo "== ready: $DEST ($size)"
echo "   next: SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh"
echo "         SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh   # torch build is skipped"

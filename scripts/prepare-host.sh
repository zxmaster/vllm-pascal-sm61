#!/bin/sh
# prepare-host.sh — bring a fresh host to the point where the image can be built.
#
# Checks the prerequisites, creates the directory layout the compose files mount,
# makes sure the external docker network exists, and seeds .env.v031 from
# .env.example. Idempotent: safe to re-run.
#
# Usage:
#   ./scripts/prepare-host.sh            # check + create what is missing
#   ./scripts/prepare-host.sh --check    # check only, change nothing
set -eu

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

cd "$(dirname "$0")/.."

fails=0
warns=0
fail() { echo "FAIL  $*" >&2; fails=$((fails + 1)); }
warn() { echo "WARN  $*" >&2; warns=$((warns + 1)); }
ok()   { echo "ok    $*"; }

# --- 1. platform -----------------------------------------------------------
case "$(uname -m)" in
  x86_64|amd64) ok "architecture x86_64" ;;
  *) fail "architecture $(uname -m): the image is built for x86_64 only" ;;
esac
case "$(uname -s)" in
  Linux) ok "kernel $(uname -r)" ;;
  *) fail "operating system $(uname -s): only Linux is supported" ;;
esac

# --- 2. GPU ----------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found — install the NVIDIA driver"
else
  gpu_line=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
  if [ -z "$gpu_line" ]; then
    fail "nvidia-smi reports no GPU"
  else
    gpu_name=$(printf '%s' "$gpu_line" | cut -d, -f1 | sed 's/^ //;s/ $//')
    compute_cap=$(printf '%s' "$gpu_line" | cut -d, -f2 | sed 's/^ //;s/ $//')
    vram=$(printf '%s' "$gpu_line" | cut -d, -f3 | sed 's/^ //;s/ $//')
    if [ "$compute_cap" = "6.1" ]; then
      ok "GPU $gpu_name (compute $compute_cap, $vram)"
    else
      warn "GPU $gpu_name has compute capability $compute_cap — this image is built for sm_61 (6.1) only; use official wheels instead"
    fi
  fi
fi

# --- 3. docker + nvidia runtime -------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  fail "docker not found"
else
  ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo '(daemon not reachable)')"
  runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null || echo '{}')
  case "$runtimes" in
    *'"nvidia"'*) ok "docker runtime 'nvidia' available" ;;
    *) fail "docker runtime 'nvidia' missing — install nvidia-container-toolkit and run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker" ;;
  esac
fi

# --- 4. disk ---------------------------------------------------------------
avail_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1024/1024}')
if [ "$avail_gb" -lt 60 ]; then
  fail "only ${avail_gb} GiB free here — the image is ~46 GB and the build needs ~150 GB in total"
elif [ "$avail_gb" -lt 150 ]; then
  warn "${avail_gb} GiB free — enough for the image, but the full build (source cache + image + model) wants ~150 GB"
else
  ok "${avail_gb} GiB free"
fi

# --- 5. directory layout ---------------------------------------------------
for d in volumes/models volumes/hf-cache volumes/hf-configs volumes/vllm-cache logs; do
  if [ -d "$d" ]; then
    ok "dir $d"
  elif [ "$CHECK_ONLY" = "1" ]; then
    warn "dir $d missing (would be created)"
  else
    mkdir -p "$d"
    ok "dir $d (created)"
  fi
done

# --- 6. external network ---------------------------------------------------
if ! command -v docker >/dev/null 2>&1; then
  :
elif docker network inspect intranet_net >/dev/null 2>&1; then
  subnet=$(docker network inspect intranet_net --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || true)
  case "$subnet" in
    172.18.0.0/16|172.18.*) ok "network intranet_net ($subnet)" ;;
    *) warn "network intranet_net has subnet '$subnet' but the compose files pin the container IP to 172.18.100.9 — either recreate the network (docker network rm intranet_net && docker network create --subnet 172.18.0.0/16 intranet_net) or drop the ipv4_address/networks block from the compose file" ;;
  esac
elif [ "$CHECK_ONLY" = "1" ]; then
  warn "network intranet_net missing (would be created)"
else
  if docker network create --subnet 172.18.0.0/16 --gateway 172.18.0.1 intranet_net >/dev/null 2>&1; then
    ok "network intranet_net (created, 172.18.0.0/16)"
  elif docker network create intranet_net >/dev/null 2>&1; then
    warn "network intranet_net created with a docker-assigned subnet — the pinned IP 172.18.100.9 may not be inside it; fix the subnet or the compose ipv4_address"
  else
    fail "could not create network intranet_net"
  fi
fi

# --- 7. env profile --------------------------------------------------------
if [ -f .env.v031 ]; then
  ok ".env.v031 present"
elif [ "$CHECK_ONLY" = "1" ]; then
  warn ".env.v031 missing (would be copied from .env.example)"
else
  cp .env.example .env.v031
  ok ".env.v031 created from .env.example"
fi

# --- 8. torch artifact -----------------------------------------------------
if [ -d artifacts/torch/site-packages/current/torch ]; then
  ok "torch artifact present (build will reuse it, SM61_USE_TORCH_ARTIFACT=1)"
else
  warn "no torch artifact — run ./scripts/fetch-torch-artifact.sh to skip the multi-hour torch build (otherwise the build compiles torch from source)"
fi

echo
if [ "$fails" -gt 0 ]; then
  echo "== $fails blocking problem(s), $warns warning(s) — fix the FAIL lines above"
  exit 1
fi
echo "== host looks ready ($warns warning(s))"
echo "   next: ./scripts/fetch-torch-artifact.sh          # prebuilt torch (recommended)"
echo "         ./scripts/prepare-model.sh                 # GGUF + tokenizer + config"
echo "         SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh"
echo "         SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh"

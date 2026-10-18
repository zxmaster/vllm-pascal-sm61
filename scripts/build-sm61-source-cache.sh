#!/bin/sh
set -eu

# Build the vLLM source-cache image (git clones used by the sm61 build).
# v0.28 variant: clones only vLLM (PyTorch is not built from source anymore).
# Usage: SM61_ENV_FILE=.env.v028 ./scripts/build-sm61-source-cache.sh

if [ -f "${SM61_ENV_FILE:-./.env}" ]; then
  while IFS= read -r env_line || [ -n "$env_line" ]; do
    case "$env_line" in
      ''|'#'*)
        continue
        ;;
      SM61_*'='*)
        env_key=${env_line%%=*}
        env_value=${env_line#*=}
        export "$env_key=$env_value"
        ;;
    esac
  done < "${SM61_ENV_FILE:-./.env}"
fi

IMAGE=${IMAGE:-${SM61_SOURCE_CACHE_IMAGE:-local/vllm-source-cache:sm61-v029}}
DOCKERFILE=${DOCKERFILE:-${SM61_SOURCE_DOCKERFILE:-docker/sm61/Dockerfile.sources}}
TORCH_REF=${TORCH_REF:-${SM61_TORCH_REF:-v2.13.0}}
VLLM_REF=${VLLM_REF:-${SM61_VLLM_REF:-v0.28.0}}
GIT_SUBMODULE_JOBS=${GIT_SUBMODULE_JOBS:-${SM61_GIT_SUBMODULE_JOBS:-8}}

docker build \
  -t "$IMAGE" \
  -f "$DOCKERFILE" \
  --build-arg TORCH_REF="$TORCH_REF" \
  --build-arg VLLM_REF="$VLLM_REF" \
  --build-arg GIT_SUBMODULE_JOBS="$GIT_SUBMODULE_JOBS" \
  .

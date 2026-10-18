#!/bin/sh
set -eu

if [ -f "${SM61_ENV_FILE:-./.env}" ]; then
  while IFS= read -r env_line || [ -n "$env_line" ]; do
    case "$env_line" in
      ''|'#'*)
        continue
        ;;
      SM61_*'='*|IMAGE=''*|DOCKERFILE=''*|LOG_ROOT=''* )
        env_key=${env_line%%=*}
        env_value=${env_line#*=}
        export "$env_key=$env_value"
        ;;
    esac
  done < "${SM61_ENV_FILE:-./.env}"
fi

IMAGE=${IMAGE:-local/vllm-openai:sm61-v029}
DOCKERFILE=${DOCKERFILE:-${SM61_DOCKERFILE:-docker/sm61/Dockerfile}}
LOG_ROOT=${LOG_ROOT:-logs/build}
SOURCE_CACHE_IMAGE=${SM61_SOURCE_CACHE_IMAGE:-local/vllm-source-cache:sm61-v029}
TORCH_REF=${SM61_TORCH_REF:-v2.10.0}
VLLM_REF=${SM61_VLLM_REF:-v0.18.0}
MAX_JOBS=${SM61_MAX_JOBS:-8}
USE_TORCH_ARTIFACT=${SM61_USE_TORCH_ARTIFACT:-}
TORCH_ARTIFACT_SUBDIR=${SM61_TORCH_ARTIFACT_SUBDIR:-current}
BUILDX_BUILDER=${SM61_BUILDX_BUILDER:-}
BUILDX_PROGRESS=${SM61_BUILDX_PROGRESS:-plain}

if [ -z "$USE_TORCH_ARTIFACT" ]; then
  if [ -d "artifacts/torch/site-packages/$TORCH_ARTIFACT_SUBDIR/torch" ]; then
    USE_TORCH_ARTIFACT=1
  else
    USE_TORCH_ARTIFACT=0
  fi
fi

mkdir -p "$LOG_ROOT"

timestamp=$(date -u +%Y%m%dT%H%M%SZ)
log_file="$LOG_ROOT/sm61-build-${timestamp}.log"
latest_link="$LOG_ROOT/latest.log"
fifo_path=$(mktemp -u /tmp/sm61-build-log-fifo.XXXXXX)

mkfifo "$fifo_path"
ln -sfn "$(basename "$log_file")" "$latest_link"
tee -a "$log_file" <"$fifo_path" &
tee_pid=$!

printf 'Writing build log to %s\n' "$log_file"
printf 'Torch artifact reuse: %s (%s)\n' "$USE_TORCH_ARTIFACT" "$TORCH_ARTIFACT_SUBDIR"

if docker buildx version >/dev/null 2>&1; then
  printf 'Build command: docker buildx build --load\n'
  if [ -n "$BUILDX_BUILDER" ]; then
    printf 'Buildx builder: %s\n' "$BUILDX_BUILDER"
    if docker buildx build \
      --builder "$BUILDX_BUILDER" \
      --load \
      --progress "$BUILDX_PROGRESS" \
      --build-arg SOURCE_CACHE_IMAGE="$SOURCE_CACHE_IMAGE" \
      --build-arg TORCH_REF="$TORCH_REF" \
      --build-arg VLLM_REF="$VLLM_REF" \
      --build-arg MAX_JOBS="$MAX_JOBS" \
      --build-arg USE_TORCH_ARTIFACT="$USE_TORCH_ARTIFACT" \
      --build-arg TORCH_ARTIFACT_SUBDIR="$TORCH_ARTIFACT_SUBDIR" \
      -t "$IMAGE" \
      -f "$DOCKERFILE" \
      . >"$fifo_path" 2>&1; then
      status=0
    else
      status=$?
    fi
  else
    if docker buildx build \
      --load \
      --progress "$BUILDX_PROGRESS" \
      --build-arg SOURCE_CACHE_IMAGE="$SOURCE_CACHE_IMAGE" \
      --build-arg TORCH_REF="$TORCH_REF" \
      --build-arg VLLM_REF="$VLLM_REF" \
      --build-arg MAX_JOBS="$MAX_JOBS" \
      --build-arg USE_TORCH_ARTIFACT="$USE_TORCH_ARTIFACT" \
      --build-arg TORCH_ARTIFACT_SUBDIR="$TORCH_ARTIFACT_SUBDIR" \
      -t "$IMAGE" \
      -f "$DOCKERFILE" \
      . >"$fifo_path" 2>&1; then
      status=0
    else
      status=$?
    fi
  fi
else
  printf 'Build command: docker build\n'
  if docker build \
    --build-arg SOURCE_CACHE_IMAGE="$SOURCE_CACHE_IMAGE" \
    --build-arg TORCH_REF="$TORCH_REF" \
    --build-arg VLLM_REF="$VLLM_REF" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    --build-arg USE_TORCH_ARTIFACT="$USE_TORCH_ARTIFACT" \
    --build-arg TORCH_ARTIFACT_SUBDIR="$TORCH_ARTIFACT_SUBDIR" \
    -t "$IMAGE" \
    -f "$DOCKERFILE" \
    . >"$fifo_path" 2>&1; then
    status=0
  else
    status=$?
  fi
fi

wait "$tee_pid" 2>/dev/null || true
rm -f "$fifo_path"

exit "$status"

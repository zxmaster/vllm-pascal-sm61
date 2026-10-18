#!/bin/sh
set -eu

BASE_ENV_FILE=${BASE_ENV_FILE:-.env}
COMPOSE_FILES=${COMPOSE_FILES:-${COMPOSE_FILE:-compose.yaml:compose.sm61.yaml}}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B-GGUF}
OUTPUT_DIR=${OUTPUT_DIR:-/logs/bench}
RUN_PREFIX=${RUN_PREFIX:-dflash-bench}
MODES=${MODES:-baseline-tq,baseline-no-tq,dflash-only,dflash-tq}
SKIP_QUALITY_BENCH=${SKIP_QUALITY_BENCH:-0}
CONTAINER_NAME=${CONTAINER_NAME:-vllm-openai}
CONTAINER_BASE_URL=${CONTAINER_BASE_URL:-http://127.0.0.1:8000}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --modes)
      MODES=${2:?--modes requires a value}
      shift 2
      ;;
    --skip-quality-bench)
      SKIP_QUALITY_BENCH=1
      shift
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

tmp_dir=$(mktemp -d)

cleanup() {
  rm -rf "$tmp_dir"
}

trap cleanup EXIT HUP INT TERM

prepare_dflash_draft() {
  env_file=$1
  if ! awk -F= 'BEGIN { enabled = 0 } $1 == "DFLASH_ENABLE" { enabled = ($2 == "1") } END { exit(enabled ? 0 : 1) }' "$env_file"; then
    return 0
  fi

  sh ./scripts/prepare-dflash-draft.sh --env-file "$env_file"
}

set_env_var() {
  key=$1
  value=$2
  file=$3
  tmp_file=$tmp_dir/$(basename "$file").tmp

  awk -v key="$key" -v value="$value" '
    BEGIN { found = 0 }
    $0 ~ "^" key "=" {
      print key "=" value
      found = 1
      next
    }
    { print }
    END {
      if (!found) {
        print key "=" value
      }
    }
  ' "$file" > "$tmp_file"

  mv "$tmp_file" "$file"
}

compose_up_vllm() {
  env_file=$1
  old_ifs=$IFS

  set -- docker compose
  IFS=:
  for compose_file in $COMPOSE_FILES; do
    set -- "$@" -f "$compose_file"
  done
  IFS=$old_ifs

  set -- "$@" up -d --force-recreate vllm
  VLLM_ENV_FILE="$env_file" "$@"
}

container_smoke_test() {
  docker exec \
    -e API_KEY="$API_KEY" \
    -e MODEL_NAME="$MODEL_NAME" \
    "$CONTAINER_NAME" \
    sh /workspace/scripts/smoke-test.sh "$CONTAINER_BASE_URL"
}

container_quality_bench() {
  mode=$1
  docker exec \
    -e API_KEY="$API_KEY" \
    -e MODEL_NAME="$MODEL_NAME" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    "$CONTAINER_NAME" \
    sh /workspace/scripts/quality-bench.sh "$CONTAINER_BASE_URL" "" "$RUN_PREFIX-$mode"
}

run_mode() {
  mode=$1
  env_file=$tmp_dir/$mode.env
  cp "$BASE_ENV_FILE" "$env_file"

  case "$mode" in
    baseline-tq)
      set_env_var DFLASH_ENABLE 0 "$env_file"
      set_env_var TURBOQUANT_ENABLE 1 "$env_file"
      ;;
    baseline-no-tq)
      set_env_var DFLASH_ENABLE 0 "$env_file"
      set_env_var TURBOQUANT_ENABLE 0 "$env_file"
      ;;
    dflash-only)
      set_env_var DFLASH_ENABLE 1 "$env_file"
      set_env_var TURBOQUANT_ENABLE 0 "$env_file"
      ;;
    dflash-tq)
      set_env_var DFLASH_ENABLE 1 "$env_file"
      set_env_var TURBOQUANT_ENABLE 1 "$env_file"
      ;;
    *)
      printf 'Unsupported mode: %s\n' "$mode" >&2
      exit 2
      ;;
  esac

  set_env_var IDLE_UNLOAD_TIMEOUT_SECONDS 0 "$env_file"

  printf '=== %s ===\n' "$mode"
  prepare_dflash_draft "$env_file"
  compose_up_vllm "$env_file"
  container_smoke_test

  if [ "$SKIP_QUALITY_BENCH" = "1" ]; then
    return 0
  fi

  container_quality_bench "$mode"
}

for mode in $(printf '%s' "$MODES" | tr ',' ' '); do
  run_mode "$mode"
done

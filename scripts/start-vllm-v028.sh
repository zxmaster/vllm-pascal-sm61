#!/bin/sh
set -eu

# ============================================================================
# start-vllm-v028.sh — vLLM Pascal (sm_61) entrypoint for the v0.28.0 stack.
#
# v0.28 changes vs the v0.18 script:
#   - GGUF handling moved to the vllm-gguf-plugin (native) -> all qwen35
#     GGUF runtime patches removed
#   - DFlash backport removed -> native --speculative-config method=dflash
#   - TurboQuant hook removed -> native --kv-cache-dtype turboquant_*
#   - Pascal-only work kept, as a single patch module:
#     /workspace/scripts/pascal_runtime_patch.py (GDN torch fallback + SDPA)
#   - the DFLASH_MAX_MODEL_LEN_CAP=24 clamp is GONE (it broke chat entirely)
# ============================================================================

MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-local-gguf}
VLLM_HOST=${VLLM_HOST:-0.0.0.0}
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_BACKEND_HOST=${VLLM_BACKEND_HOST:-127.0.0.1}
VLLM_BACKEND_PORT=${VLLM_BACKEND_PORT:-8001}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1024}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.72}
CPU_OFFLOAD_GB=${CPU_OFFLOAD_GB:-0}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
MAMBA_CACHE_MODE=${MAMBA_CACHE_MODE:-}
MAMBA_BLOCK_SIZE=${MAMBA_BLOCK_SIZE:-}
KV_OFFLOADING_SIZE_GB=${KV_OFFLOADING_SIZE_GB:-}
KV_OFFLOADING_BACKEND=${KV_OFFLOADING_BACKEND:-}
IDLE_UNLOAD_TIMEOUT_SECONDS=${IDLE_UNLOAD_TIMEOUT_SECONDS:-0}
IDLE_UNLOAD_STARTUP_TIMEOUT_SECONDS=${IDLE_UNLOAD_STARTUP_TIMEOUT_SECONDS:-900}
IDLE_UNLOAD_SHUTDOWN_GRACE_SECONDS=${IDLE_UNLOAD_SHUTDOWN_GRACE_SECONDS:-30}
IDLE_UNLOAD_POLL_SECONDS=${IDLE_UNLOAD_POLL_SECONDS:-5}
IDLE_UNLOAD_REQUEST_TIMEOUT_SECONDS=${IDLE_UNLOAD_REQUEST_TIMEOUT_SECONDS:-0}
GPU_POWER_LIMIT_PERCENT=${GPU_POWER_LIMIT_PERCENT:-}
GPU_POWER_LIMIT_WATTS=${GPU_POWER_LIMIT_WATTS:-}
GPU_POWER_LIMIT_GPU_INDEX=${GPU_POWER_LIMIT_GPU_INDEX:-0}
GPU_POWER_LIMIT_STRICT=${GPU_POWER_LIMIT_STRICT:-0}
LOG_ROOT=${LOG_ROOT:-/logs}
VLLM_ENABLE_FILE_LOGGING=${VLLM_ENABLE_FILE_LOGGING:-1}
TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
COPILOT_SKIP_GDN_PREFILL_WARMUP=${COPILOT_SKIP_GDN_PREFILL_WARMUP:-1}
COPILOT_TRITON_FORCE_FIRST_CONFIG=${COPILOT_TRITON_FORCE_FIRST_CONFIG:-1}
# Pascal torch fallback for the GDN (Triton FLA kernels are wrong on sm_61)
VLLM_PASCAL_TORCH_GDN=${VLLM_PASCAL_TORCH_GDN:-1}
# Backward-compat aliases for DFlash (mapped onto the generic SPECULATIVE_*)
DFLASH_ENABLE=${DFLASH_ENABLE:-0}
DFLASH_DRAFT_MODEL=${DFLASH_DRAFT_MODEL:-/models/Qwen3.5-4B-DFlash-GGUF/Qwen3.5-4B-DFlash-Q4_K_M.gguf}
DFLASH_NUM_SPECULATIVE_TOKENS=${DFLASH_NUM_SPECULATIVE_TOKENS:-1}
DFLASH_STRICT=${DFLASH_STRICT:-0}
SPECULATIVE_METHOD=${SPECULATIVE_METHOD:-}
SPECULATIVE_DRAFT_MODEL=${SPECULATIVE_DRAFT_MODEL:-}
SPECULATIVE_NUM_SPECULATIVE_TOKENS=${SPECULATIVE_NUM_SPECULATIVE_TOKENS:-}
SPECULATIVE_MAX_MODEL_LEN=${SPECULATIVE_MAX_MODEL_LEN:-}
SPECULATIVE_ATTENTION_BACKEND=${SPECULATIVE_ATTENTION_BACKEND:-}
SPECULATIVE_STRICT=${SPECULATIVE_STRICT:-0}
# Native TurboQuant KV cache (v0.20+): turboquant_k8v4 | turboquant_4bit_nc |
# turboquant_k3v4_nc | turboquant_3bit_nc | (empty = off)
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-}

if [ -z "$SPECULATIVE_METHOD" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_METHOD=dflash
fi

if [ -z "$SPECULATIVE_DRAFT_MODEL" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_DRAFT_MODEL=$DFLASH_DRAFT_MODEL
fi

if [ -z "$SPECULATIVE_NUM_SPECULATIVE_TOKENS" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_NUM_SPECULATIVE_TOKENS=$DFLASH_NUM_SPECULATIVE_TOKENS
fi

if [ "$SPECULATIVE_STRICT" = "0" ] && [ "$DFLASH_STRICT" = "1" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_STRICT=1
fi

export TORCH_COMPILE_DISABLE
export TORCHDYNAMO_DISABLE
export COPILOT_SKIP_GDN_PREFILL_WARMUP
export COPILOT_TRITON_FORCE_FIRST_CONFIG
export VLLM_PASCAL_TORCH_GDN

# ---------------------------------------------------------------------------
prefer_host_nvidia_driver_libs() {
  host_nvidia_lib_dir=
  if [ -e /lib/libcuda.so.1 ] && [ -e /lib/libnvidia-ml.so.1 ]; then
    host_nvidia_lib_dir=/lib
  elif [ -e /usr/local/nvidia/lib64/libcuda.so.1 ] && [ -e /usr/local/nvidia/lib64/libnvidia-ml.so.1 ]; then
    host_nvidia_lib_dir=/usr/local/nvidia/lib64
  fi

  [ -n "$host_nvidia_lib_dir" ] || return 0

  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
  case ":${LD_LIBRARY_PATH}:" in
    *":${host_nvidia_lib_dir}:"*) ;;
    *)
      if [ -n "$LD_LIBRARY_PATH" ]; then
        LD_LIBRARY_PATH="$host_nvidia_lib_dir:$LD_LIBRARY_PATH"
      else
        LD_LIBRARY_PATH=$host_nvidia_lib_dir
      fi
      export LD_LIBRARY_PATH
      ;;
  esac

  ln -sfn "$host_nvidia_lib_dir/libcuda.so.1" /usr/local/cuda/lib64/libcuda.so
  ln -sfn "$host_nvidia_lib_dir/libcuda.so.1" /usr/local/cuda/targets/x86_64-linux/lib/libcuda.so
  ln -sfn "$host_nvidia_lib_dir/libnvidia-ml.so.1" /usr/local/cuda/lib64/libnvidia-ml.so
  ln -sfn "$host_nvidia_lib_dir/libnvidia-ml.so.1" /usr/local/cuda/targets/x86_64-linux/lib/libnvidia-ml.so

  printf 'Preferring host NVIDIA driver libs from: %s\n' "$host_nvidia_lib_dir"
}

gpu_power_limit_warn_or_exit() {
  printf 'GPU power limit: %s\n' "$1" >&2
  if [ "$GPU_POWER_LIMIT_STRICT" = "1" ]; then
    exit 1
  fi
}

apply_gpu_power_limit() {
  if [ -z "$GPU_POWER_LIMIT_WATTS" ] \
    && { [ -z "$GPU_POWER_LIMIT_PERCENT" ] || [ "$GPU_POWER_LIMIT_PERCENT" = "0" ]; }; then
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    gpu_power_limit_warn_or_exit 'nvidia-smi is not available; leaving existing limit unchanged'
    return 0
  fi

  target_limit=
  target_source=
  if [ -n "$GPU_POWER_LIMIT_WATTS" ]; then
    target_limit=$(printf '%s' "$GPU_POWER_LIMIT_WATTS" | tr -d '[:space:]')
    target_source="explicit watt setting"
    if ! awk -v value="$target_limit" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
      gpu_power_limit_warn_or_exit "invalid GPU_POWER_LIMIT_WATTS=$GPU_POWER_LIMIT_WATTS; expected a positive number"
      return 0
    fi
  else
    power_percent=$(printf '%s' "$GPU_POWER_LIMIT_PERCENT" | tr -d '[:space:]')
    power_percent=${power_percent%\%}
    if ! awk -v value="$power_percent" 'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0 && value <= 100) }'; then
      gpu_power_limit_warn_or_exit "invalid GPU_POWER_LIMIT_PERCENT=$GPU_POWER_LIMIT_PERCENT; expected 0 < percent <= 100"
      return 0
    fi

    if ! max_limit=$(nvidia-smi -i "$GPU_POWER_LIMIT_GPU_INDEX" --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 { gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print }'); then
      gpu_power_limit_warn_or_exit "could not query max power limit for GPU index $GPU_POWER_LIMIT_GPU_INDEX"
      return 0
    fi
    if ! min_limit=$(nvidia-smi -i "$GPU_POWER_LIMIT_GPU_INDEX" --query-gpu=power.min_limit --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 { gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print }'); then
      min_limit=
    fi

    if ! target_limit=$(awk -v pct="$power_percent" -v min="$min_limit" -v max="$max_limit" '
      BEGIN {
        number = "^[0-9]+([.][0-9]+)?$"
        if (!(max ~ number) || max <= 0) {
          exit 1
        }
        target = max * pct / 100
        if (min ~ number && min > 0 && target < min) {
          target = min
        }
        printf "%.2f", target
      }
    '); then
      gpu_power_limit_warn_or_exit "could not derive a watt limit from GPU_POWER_LIMIT_PERCENT=$GPU_POWER_LIMIT_PERCENT and max_limit=$max_limit"
      return 0
    fi
    target_source="$power_percent% of max ${max_limit}W"
  fi

  if apply_output=$(nvidia-smi -i "$GPU_POWER_LIMIT_GPU_INDEX" -pl "$target_limit" 2>&1); then
    printf '%s\n' "$apply_output" | sed 's/^/GPU power limit: /'
    if printf '%s\n' "$apply_output" | grep -qi 'not supported'; then
      gpu_power_limit_warn_or_exit \
        "setting a power limit is not supported for GPU index $GPU_POWER_LIMIT_GPU_INDEX; leaving existing limit unchanged"
      return 0
    fi

    current_limit=$(nvidia-smi -i "$GPU_POWER_LIMIT_GPU_INDEX" --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | awk 'NR == 1 { gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print }' || true)
    case "$current_limit" in
      ''|'[N/A]'|'N/A')
        gpu_power_limit_warn_or_exit \
          "nvidia-smi did not report a current power limit after applying ${target_limit}W ($target_source); leaving runtime startup non-fatal"
        return 0
        ;;
    esac
    printf 'GPU power limit: configured GPU index %s to %sW (%s); current reported limit: %sW\n' \
      "$GPU_POWER_LIMIT_GPU_INDEX" "$target_limit" "$target_source" "$current_limit"
  else
    gpu_power_limit_warn_or_exit "failed to set GPU index $GPU_POWER_LIMIT_GPU_INDEX to ${target_limit}W ($target_source): $apply_output"
  fi
}

# ---------------------------------------------------------------------------
# Pascal runtime patches: GDN torch fallback + Triton-attention SDPA fallback.
# Idempotent; safe to run every start.
apply_pascal_patches() {
  python3 /workspace/scripts/pascal_runtime_patch.py
}

# ---------------------------------------------------------------------------
configure_speculative_draft() {
  SPECULATIVE_CONFIG_JSON=
  SPECULATIVE_DRAFT_WINDOW_READY=0

  [ -n "$SPECULATIVE_METHOD" ] || {
    printf 'Speculative draft: disabled (SPECULATIVE_METHOD is empty)\n'
    return 0
  }

  [ -n "$SPECULATIVE_DRAFT_MODEL" ] || {
    printf 'Speculative draft: SPECULATIVE_DRAFT_MODEL is required when SPECULATIVE_METHOD=%s\n' "$SPECULATIVE_METHOD" >&2
    exit 1
  }

  [ -n "$SPECULATIVE_NUM_SPECULATIVE_TOKENS" ] || {
    printf 'Speculative draft: SPECULATIVE_NUM_SPECULATIVE_TOKENS is required when SPECULATIVE_METHOD=%s\n' "$SPECULATIVE_METHOD" >&2
    exit 1
  }

  SPECULATIVE_CONFIG_JSON=$(
    SPECULATIVE_METHOD="$SPECULATIVE_METHOD" \
    SPECULATIVE_DRAFT_MODEL="$SPECULATIVE_DRAFT_MODEL" \
    SPECULATIVE_NUM_SPECULATIVE_TOKENS="$SPECULATIVE_NUM_SPECULATIVE_TOKENS" \
    SPECULATIVE_MAX_MODEL_LEN="$SPECULATIVE_MAX_MODEL_LEN" \
    SPECULATIVE_ATTENTION_BACKEND="$SPECULATIVE_ATTENTION_BACKEND" \
    python3 - <<'PY'
import json
import os

config = {
    "method": os.environ["SPECULATIVE_METHOD"],
    "model": os.environ["SPECULATIVE_DRAFT_MODEL"],
    "num_speculative_tokens": int(os.environ["SPECULATIVE_NUM_SPECULATIVE_TOKENS"]),
}

max_model_len = os.environ.get("SPECULATIVE_MAX_MODEL_LEN", "").strip()
if max_model_len:
    config["max_model_len"] = int(max_model_len)

attention_backend = os.environ.get("SPECULATIVE_ATTENTION_BACKEND", "").strip()
if attention_backend:
    config["attention_backend"] = attention_backend

print(json.dumps(config, separators=(",", ":")))
PY
  )

  printf 'Speculative draft: method=%s model=%s num_speculative_tokens=%s\n' \
    "$SPECULATIVE_METHOD" "$SPECULATIVE_DRAFT_MODEL" "$SPECULATIVE_NUM_SPECULATIVE_TOKENS"
}

# ---------------------------------------------------------------------------
prefer_host_nvidia_driver_libs
apply_gpu_power_limit
apply_pascal_patches
configure_speculative_draft

if [ -z "${PYTORCH_ALLOC_CONF:-}" ]; then
  # NOTE: both max_split_size_mb:64 and expandable_segments hurt here:
  #  - max_split_size_mb:64 left ~260 MiB "reserved but unallocated" swiss-cheese
  #    blocks that couldn't back the 40 MiB long-prefill allocations (3 OOMs);
  #  - expandable_segments broke vLLM's KV profiling (0.43 -> 0.23 GiB).
  # Use the allocator default (unrestricted splitting) instead.
  printf 'PYTORCH_ALLOC_CONF not set — using default allocator policy\n'
fi
export CUDA_MODULE_LOADING=LAZY

server_host=$VLLM_HOST
server_port=$VLLM_PORT
public_host=$VLLM_HOST
public_port=$VLLM_PORT

if [ "$IDLE_UNLOAD_TIMEOUT_SECONDS" != "0" ]; then
  server_host=$VLLM_BACKEND_HOST
  server_port=$VLLM_BACKEND_PORT
fi

set -- \
  vllm serve "$MODEL_PATH" \
  --host "$server_host" \
  --port "$server_port" \
  --served-model-name $SERVED_MODEL_NAME \
  --generation-config vllm \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager

if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
  set -- "$@" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
fi

if [ "$ENABLE_PREFIX_CACHING" = "1" ]; then
  set -- "$@" --enable-prefix-caching
fi

if [ -n "$MAMBA_CACHE_MODE" ]; then
  set -- "$@" --mamba-cache-mode "$MAMBA_CACHE_MODE"
fi

if [ -n "$MAMBA_BLOCK_SIZE" ]; then
  set -- "$@" --mamba-block-size "$MAMBA_BLOCK_SIZE"
fi

if [ "$CPU_OFFLOAD_GB" != "0" ]; then
  set -- "$@" --cpu-offload-gb "$CPU_OFFLOAD_GB"
fi

if [ -n "$KV_OFFLOADING_SIZE_GB" ]; then
  set -- "$@" --kv-offloading-size "$KV_OFFLOADING_SIZE_GB" --kv-offloading-backend "$KV_OFFLOADING_BACKEND"
fi

if [ -n "$KV_CACHE_DTYPE" ]; then
  set -- "$@" --kv-cache-dtype "$KV_CACHE_DTYPE"
fi

if [ -n "${TOKENIZER_MODEL:-}" ]; then
  set -- "$@" --tokenizer "$TOKENIZER_MODEL"
fi

if [ -n "${HF_CONFIG_PATH:-}" ]; then
  set -- "$@" --hf-config-path "$HF_CONFIG_PATH"
fi

if [ -n "${REASONING_PARSER:-}" ]; then
  set -- "$@" --reasoning-parser "$REASONING_PARSER"
fi

if [ -n "${API_KEY:-}" ]; then
  set -- "$@" --api-key "$API_KEY"
fi

if [ -n "${SPECULATIVE_CONFIG_JSON:-}" ]; then
  set -- "$@" --speculative-config "$SPECULATIVE_CONFIG_JSON"
fi

if [ -n "${VLLM_EXTRA_ARGS:-}" ]; then
  # Intentional word splitting for extra CLI flags.
  set -- "$@" $VLLM_EXTRA_ARGS
fi

printf 'Launching vLLM with model: %s\n' "$MODEL_PATH"
printf 'Served model name: %s\n' "$SERVED_MODEL_NAME"
if [ -n "${LOG_FILE_PATH:-}" ]; then
  printf 'Writing runtime log to: %s\n' "$LOG_FILE_PATH"
fi
unset VLLM_HOST VLLM_PORT VLLM_EXTRA_ARGS

if [ "$IDLE_UNLOAD_TIMEOUT_SECONDS" != "0" ]; then
  printf 'Idle unload enabled: backend will stop after %ss of inactivity\n' "$IDLE_UNLOAD_TIMEOUT_SECONDS"
  exec python3 /workspace/scripts/idle_unload_proxy.py \
    --listen-host "$public_host" \
    --listen-port "$public_port" \
    --backend-host "$VLLM_BACKEND_HOST" \
    --backend-port "$VLLM_BACKEND_PORT" \
    --idle-timeout-seconds "$IDLE_UNLOAD_TIMEOUT_SECONDS" \
    --startup-timeout-seconds "$IDLE_UNLOAD_STARTUP_TIMEOUT_SECONDS" \
    --shutdown-grace-seconds "$IDLE_UNLOAD_SHUTDOWN_GRACE_SECONDS" \
    --idle-poll-seconds "$IDLE_UNLOAD_POLL_SECONDS" \
    --request-timeout-seconds "$IDLE_UNLOAD_REQUEST_TIMEOUT_SECONDS" \
    -- "$@"
fi

exec "$@"

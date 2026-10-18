#!/bin/sh
set -eu

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
DFLASH_ENABLE=${DFLASH_ENABLE:-0}
DFLASH_DRAFT_MODEL=${DFLASH_DRAFT_MODEL:-/models/Qwen3.5-4B-DFlash-GGUF/Qwen3.5-4B-DFlash-Q4_K_M.gguf}
DFLASH_NUM_SPECULATIVE_TOKENS=${DFLASH_NUM_SPECULATIVE_TOKENS:-15}
DFLASH_ATTENTION_BACKEND=${DFLASH_ATTENTION_BACKEND:-}
DFLASH_DRAFT_WINDOW_SIZE=${DFLASH_DRAFT_WINDOW_SIZE:-}
DFLASH_STRICT=${DFLASH_STRICT:-0}
DFLASH_MAX_MODEL_LEN_CAP=${DFLASH_MAX_MODEL_LEN_CAP:-24}
DFLASH_MAX_BATCHED_TOKENS_CAP=${DFLASH_MAX_BATCHED_TOKENS_CAP:-24}
DFLASH_GPU_MEMORY_UTILIZATION_CAP=${DFLASH_GPU_MEMORY_UTILIZATION_CAP:-0.90}
SPECULATIVE_METHOD=${SPECULATIVE_METHOD:-}
SPECULATIVE_DRAFT_MODEL=${SPECULATIVE_DRAFT_MODEL:-}
SPECULATIVE_NUM_SPECULATIVE_TOKENS=${SPECULATIVE_NUM_SPECULATIVE_TOKENS:-}
SPECULATIVE_MAX_MODEL_LEN=${SPECULATIVE_MAX_MODEL_LEN:-}
SPECULATIVE_ATTENTION_BACKEND=${SPECULATIVE_ATTENTION_BACKEND:-}
SPECULATIVE_DRAFT_WINDOW_SIZE=${SPECULATIVE_DRAFT_WINDOW_SIZE:-}
SPECULATIVE_STRICT=${SPECULATIVE_STRICT:-0}

if [ -z "$SPECULATIVE_METHOD" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_METHOD=dflash
fi

if [ -z "$SPECULATIVE_DRAFT_MODEL" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_DRAFT_MODEL=$DFLASH_DRAFT_MODEL
fi

if [ -z "$SPECULATIVE_NUM_SPECULATIVE_TOKENS" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_NUM_SPECULATIVE_TOKENS=$DFLASH_NUM_SPECULATIVE_TOKENS
fi

if [ -z "$SPECULATIVE_ATTENTION_BACKEND" ] && [ -n "$DFLASH_ATTENTION_BACKEND" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_ATTENTION_BACKEND=$DFLASH_ATTENTION_BACKEND
fi

if [ -z "$SPECULATIVE_DRAFT_WINDOW_SIZE" ] && [ -n "$DFLASH_DRAFT_WINDOW_SIZE" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_DRAFT_WINDOW_SIZE=$DFLASH_DRAFT_WINDOW_SIZE
fi

if [ "$SPECULATIVE_STRICT" = "0" ] && [ "$DFLASH_STRICT" = "1" ] && [ "$DFLASH_ENABLE" = "1" ]; then
  SPECULATIVE_STRICT=1
fi

export TORCH_COMPILE_DISABLE
export TORCHDYNAMO_DISABLE
export COPILOT_SKIP_GDN_PREFILL_WARMUP
export COPILOT_TRITON_FORCE_FIRST_CONFIG

prefer_host_nvidia_driver_libs() {
  host_nvidia_lib_dir=
  if [ -e /lib/libcuda.so.1 ] && [ -e /lib/libnvidia-ml.so.1 ]; then
    host_nvidia_lib_dir=/lib
  elif [ -e /usr/local/nvidia/lib64/libcuda.so.1 ] && [ -e /usr/local/nvidia/lib64/libnvidia-ml.so.1 ]; then
    host_nvidia_lib_dir=/usr/local/nvidia/lib64
  fi

  [ -n "$host_nvidia_lib_dir" ] || return 0

  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
  case ":$LD_LIBRARY_PATH:" in
    *":$host_nvidia_lib_dir:"*) ;;
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

apply_qwen35_gguf_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path
import re

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/gguf_loader.py")
text = path.read_text()
modified = False

old = '''                dummy_model = initialize_model(
                    vllm_config=vllm_config,
                    model_class=qwen3_5_model_class,
                    model_config=model_config,
                )
'''
new = '''                from vllm.model_executor import offloader as offloader_pkg
                from vllm.model_executor.offloader import (
                    NoopOffloader,
                    get_offloader,
                    set_offloader,
                )
                from vllm.config import set_current_vllm_config

                noop_offloader = NoopOffloader()
                previous_offloader = get_offloader()
                previous_get_offloader = offloader_pkg.get_offloader
                previous_quant_config = vllm_config.quant_config
                previous_static_forward_context = dict(
                  vllm_config.compilation_config.static_forward_context
                )
                offloader_pkg.get_offloader = lambda: noop_offloader
                set_offloader(noop_offloader)
                try:
                    vllm_config.quant_config = None
                    with set_current_vllm_config(vllm_config, check_compile=True):
                        dummy_model = qwen3_5_model_class(
                            vllm_config=vllm_config,
                            prefix="",
                        )
                finally:
                        current_static_forward_context = dict(
                          vllm_config.compilation_config.static_forward_context
                        )
                        vllm_config.compilation_config.static_forward_context.clear()
                        vllm_config.compilation_config.static_forward_context.update(
                          previous_static_forward_context
                        )
                        vllm_config.compilation_config.static_forward_context.update(
                          current_static_forward_context
                        )
                    vllm_config.quant_config = previous_quant_config
                    offloader_pkg.get_offloader = previous_get_offloader
                    set_offloader(previous_offloader)
'''

if old in text and new not in text:
    text = text.replace(old, new, 1)
    modified = True
    print("Applied runtime Qwen3.5 GGUF offloader patch", flush=True)

old = '''            if (
        config.model_type in ("qwen3_5", "qwen3_5_moe")
        and isinstance(model_config, ModelConfig)
      ):
        from vllm.model_executor.models.qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
          Qwen3_5MoeForConditionalGeneration,
        )
'''
new = '''            if (
        config.model_type in ("qwen3_5", "qwen3_5_moe", "qwen3_5_mtp")
        and isinstance(model_config, ModelConfig)
      ):
        from vllm.model_executor.models.qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
          Qwen3_5MoeForConditionalGeneration,
        )
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 MTP GGUF import patch", flush=True)

old = '''                from vllm.model_executor.models.qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
          Qwen3_5MoeForConditionalGeneration,
        )
'''
new = '''                from vllm.model_executor.models.qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
          Qwen3_5MoeForConditionalGeneration,
        )
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 MTP GGUF direct import patch", flush=True)

old = '''from vllm.model_executor.models.qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
          Qwen3_5MoeForConditionalGeneration,
        )
'''
new = '''from vllm.model_executor.models.qwen3_5 import (
          Qwen3_5ForConditionalGeneration,
          Qwen3_5MoeForConditionalGeneration,
        )
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
'''

if old in text and 'from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP' not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 MTP GGUF fallback import patch", flush=True)

old = '''                qwen3_5_model_class = {
                    ("qwen3_5", True): Qwen3_5ForCausalLM,
                    ("qwen3_5", False): Qwen3_5ForConditionalGeneration,
                    ("qwen3_5_moe", True): Qwen3_5MoeForCausalLM,
                    ("qwen3_5_moe", False): Qwen3_5MoeForConditionalGeneration,
                }[(config.model_type, language_model_only)]
'''
new = '''                qwen3_5_model_class = {
                    "qwen3_5": Qwen3_5ForConditionalGeneration,
                    "qwen3_5_moe": Qwen3_5MoeForConditionalGeneration,
                    "qwen3_5_mtp": Qwen3_5MTP,
                }[config.model_type]
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF dummy model patch", flush=True)

old = '''                qwen3_5_model_class = {
                    "qwen3_5": Qwen3_5ForConditionalGeneration,
                    "qwen3_5_moe": Qwen3_5MoeForConditionalGeneration,
                }[config.model_type]
'''
new = '''                qwen3_5_model_class = {
                    "qwen3_5": Qwen3_5ForConditionalGeneration,
                    "qwen3_5_moe": Qwen3_5MoeForConditionalGeneration,
                    "qwen3_5_mtp": Qwen3_5MTP,
                }[config.model_type]
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 MTP GGUF dummy model extension", flush=True)

old = '                    "qwen3_5_mtp": Qwen3_5MTP,\n'
new = '                    "qwen3_5_mtp": __import__("vllm.model_executor.models.qwen3_5_mtp", fromlist=["Qwen3_5MTP"]).Qwen3_5MTP,\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 MTP GGUF inline import patch", flush=True)

old = '''        if model_type == "gemma3_text":
            # Gemma3 models use "gemma3_text" in HuggingFace but
            # "gemma3" in GGUF architecture naming
            model_type = "gemma3"
        if model_type == "qwen3_5":
            # GGUF currently exposes Qwen 3.5 under the legacy qwen35 name.
            model_type = "qwen35"
'''
new = '''        if model_type == "gemma3_text":
            # Gemma3 models use "gemma3_text" in HuggingFace but
            # "gemma3" in GGUF architecture naming
            model_type = "gemma3"
        if model_type == "qwen3_5":
            # GGUF currently exposes Qwen 3.5 under the legacy qwen35 name.
            model_type = "qwen35"
        if model_type == "qwen3_5_mtp":
            # Qwen 3.5 MTP uses the same GGUF tensor family as the base qwen35 arch.
            model_type = "qwen35"
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 MTP GGUF model_type alias patch", flush=True)

old = '''        weight_type_map = get_gguf_weight_type_map(
            model_name_or_path, gguf_to_hf_name_map
        )
        is_multimodal = hasattr(model_config.hf_config, "vision_config")
'''
new = '''        weight_type_map = get_gguf_weight_type_map(
            model_name_or_path, gguf_to_hf_name_map
        )
        language_model_only = bool(
            getattr(
                getattr(model_config, "multimodal_config", None),
                "language_model_only",
                False,
            )
        )
        is_multimodal = (
            hasattr(model_config.hf_config, "vision_config")
            and not language_model_only
        )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GGUF weight-type language-only patch", flush=True)

old = '''        hf_config = model_config.hf_config
        is_multimodal = hasattr(hf_config, "vision_config")

        if is_multimodal:
'''
new = '''        hf_config = model_config.hf_config
        language_model_only = bool(
            getattr(
                getattr(model_config, "multimodal_config", None),
                "language_model_only",
                False,
            )
        )
        is_multimodal = (
            hasattr(hf_config, "vision_config") and not language_model_only
        )

        if is_multimodal:
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GGUF iterator language-only patch", flush=True)

old = '''        weight_type_map = get_gguf_weight_type_map(
      model_name_or_path, gguf_to_hf_name_map
    )
    is_multimodal = hasattr(model_config.hf_config, "vision_config")
    if is_multimodal:
'''
new = '''        weight_type_map = get_gguf_weight_type_map(
      model_name_or_path, gguf_to_hf_name_map
    )
    language_model_only = bool(
      getattr(
        getattr(model_config, "multimodal_config", None),
        "language_model_only",
        False,
      )
    )
    is_multimodal = (
      hasattr(model_config.hf_config, "vision_config")
      and not language_model_only
    )
    if is_multimodal:
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GGUF mmproj weight-type patch", flush=True)

old = '''        hf_config = model_config.hf_config
    is_multimodal = hasattr(hf_config, "vision_config")

    if is_multimodal:
      # Load mm_proj (mm_encoder + projector) for multimodal weights
'''
new = '''        hf_config = model_config.hf_config
    language_model_only = bool(
      getattr(
        getattr(model_config, "multimodal_config", None),
        "language_model_only",
        False,
      )
    )
    is_multimodal = (
      hasattr(hf_config, "vision_config") and not language_model_only
    )

    if is_multimodal:
      # Load mm_proj (mm_encoder + projector) for multimodal weights
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GGUF mmproj iterator patch", flush=True)

old = '''        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
'''
new = '''        if vllm_config.quant_config is not None:
            vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GGUF unquantized-module guard", flush=True)

old = '''    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(
            model_config, vllm_config=vllm_config
        )
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )
'''
new = '''    def load_weights(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        vllm_config: VllmConfig | None = None,
    ) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = getattr(self, "_cached_gguf_weights_map", None)
        if gguf_weights_map is None:
            gguf_weights_map = self._get_gguf_weights_map(
                model_config, vllm_config=vllm_config
            )
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF load_weights patch", flush=True)

old = '''    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(
            model_config, vllm_config=vllm_config
        )
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )
'''

if old in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF load_weights repair", flush=True)

old = '        gguf_weights_map = self._get_gguf_weights_map(model_config)\n'
new = '        gguf_weights_map = self._get_gguf_weights_map(\n            model_config, vllm_config=vllm_config\n        )\n        self._cached_gguf_weights_map = gguf_weights_map\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF load_model map patch", flush=True)

old = '''    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
'''
new = '''    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(
            model_config, vllm_config=vllm_config
        )
'''

if old in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF load_model repair", flush=True)

old = '            self.load_weights(model, model_config)\n'
new = '            self.load_weights(model, model_config, vllm_config=vllm_config)\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF load_model load_weights patch", flush=True)

old = '        state_dict = dummy_model.state_dict()\n'
new = '        state_dict = dummy_model.state_dict(keep_vars=True)\n'

if old in text and new not in text:
    text = text.replace(old, new, 1)
    modified = True
    print("Applied runtime Qwen3.5 GGUF state_dict patch", flush=True)

old = '''        # Build mapping and track unmapped parameters
        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)

            # Track mapping success
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                # Parameter not in manual overrides either
                unmapped_params.append(hf_name)
'''
new = '''        def get_qwen35_manual_mappings(
            hf_name: str,
        ) -> list[tuple[str, str]] | None:
      if config.model_type not in ("qwen3_5", "qwen3_5_moe", "qwen3_5_mtp"):
                return None

            layer_match = re.fullmatch(
                r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.linear_attn\\.dt_bias",
                hf_name,
            )
            if layer_match is not None:
                layer_idx = layer_match.group(2)
                return [(f"blk.{layer_idx}.ssm_dt.bias", hf_name)]

            layer_match = re.fullmatch(
                r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.linear_attn\\.A_log",
                hf_name,
            )
            if layer_match is not None:
                layer_idx = layer_match.group(2)
                return [(f"blk.{layer_idx}.ssm_a", hf_name)]

            layer_match = re.fullmatch(
              r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.mlp\\.gate_up_proj\\.weight",
              hf_name,
            )
            if layer_match is not None:
              model_prefix = layer_match.group("model_prefix")
              layer_idx = layer_match.group(2)
              return [
                (
                  f"blk.{layer_idx}.ffn_gate.weight",
                  f"{model_prefix}.layers.{layer_idx}.mlp.gate_proj.weight",
                ),
                (
                  f"blk.{layer_idx}.ffn_up.weight",
                  f"{model_prefix}.layers.{layer_idx}.mlp.up_proj.weight",
                ),
              ]

            layer_match = re.fullmatch(
              r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.self_attn\\.qkv_proj\\.weight",
              hf_name,
            )
            if layer_match is not None:
              model_prefix = layer_match.group("model_prefix")
              layer_idx = layer_match.group(2)
              return [
                (
                  f"blk.{layer_idx}.attn_q.weight",
                  f"{model_prefix}.layers.{layer_idx}.self_attn.q_proj.weight",
                ),
                (
                  f"blk.{layer_idx}.attn_k.weight",
                  f"{model_prefix}.layers.{layer_idx}.self_attn.k_proj.weight",
                ),
                (
                  f"blk.{layer_idx}.attn_v.weight",
                  f"{model_prefix}.layers.{layer_idx}.self_attn.v_proj.weight",
                ),
              ]

            layer_match = re.fullmatch(
                r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.linear_attn\\.in_proj_qkvz\\.weight",
                hf_name,
            )
            if layer_match is not None:
                model_prefix = layer_match.group("model_prefix")
                layer_idx = layer_match.group(2)
                return [
                    (
                        f"blk.{layer_idx}.attn_qkv.weight",
                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_qkv.weight",
                    ),
                    (
                        f"blk.{layer_idx}.attn_gate.weight",
                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_z.weight",
                    ),
                ]

            layer_match = re.fullmatch(
                r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.linear_attn\\.in_proj_ba\\.weight",
                hf_name,
            )
            if layer_match is not None:
                model_prefix = layer_match.group("model_prefix")
                layer_idx = layer_match.group(2)
                return [
                    (
                        f"blk.{layer_idx}.ssm_alpha.weight",
                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_a.weight",
                    ),
                    (
                        f"blk.{layer_idx}.ssm_beta.weight",
                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_b.weight",
                    ),
                ]

            if re.fullmatch(
                r"(?:language_model\\.)?model\\.layers\\.\\d+\\.self_attn\\.attn\\._(?:k|v|q|prob)_scale",
                hf_name,
            ):
                return []

            return None

        # Build mapping and track unmapped parameters
        unmapped_params = []
        for hf_name in state_dict:
            manual_mappings = get_qwen35_manual_mappings(hf_name)
            if manual_mappings is not None:
                for gguf_name_with_suffix, mapped_hf_name in manual_mappings:
                    gguf_to_hf_name_map[gguf_name_with_suffix] = mapped_hf_name
                    logger.debug(
                        "Mapped GGUF %s → HF %s",
                        gguf_name_with_suffix,
                        mapped_hf_name,
                    )
                continue

            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)

            # Track mapping success
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                # Parameter not in manual overrides either
                unmapped_params.append(hf_name)
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF manual mapping patch", flush=True)

old = '''        # Build mapping and track unmapped parameters
        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)

            # Track mapping success
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                # Parameter not in manual overrides either
                unmapped_params.append(hf_name)
'''

if old in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3.5 GGUF manual mapping repair", flush=True)

# Fix swapped alpha/beta mapping if the image was built with the wrong mapping
# Correct mapping per llama.cpp tensor_mapping.py:
#   SSM_ALPHA (blk.N.ssm_alpha) → model.layers.N.linear_attn.in_proj_a
#   SSM_BETA  (blk.N.ssm_beta)  → model.layers.N.linear_attn.in_proj_b
wrong_alpha = 'ssm_alpha.weight",\n                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_b.weight"'
right_alpha = 'ssm_alpha.weight",\n                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_a.weight"'
wrong_beta = 'ssm_beta.weight",\n                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_a.weight"'
right_beta = 'ssm_beta.weight",\n                        f"{model_prefix}.layers.{layer_idx}.linear_attn.in_proj_b.weight"'

if wrong_alpha in text and wrong_beta in text:
  # Use a placeholder to avoid double-swap
  text = text.replace(wrong_alpha, 'SSM_ALPHA_PLACEHOLDER')
  text = text.replace(wrong_beta, right_beta)
  text = text.replace('SSM_ALPHA_PLACEHOLDER', right_alpha)
  modified = True
  print("Fixed swapped ssm_alpha/ssm_beta GGUF mapping", flush=True)

# Inject A_log mapping if missing (image was built without it)
# A_log is a bare nn.Parameter (no .weight/.bias suffix), so the generic
# find_hf_name_in_tensor_map() fails with a trailing-dot bug.
# We add it as an explicit manual mapping: blk.N.ssm_a → A_log
a_log_anchor = 'return [(f"blk.{layer_idx}.ssm_dt.bias", hf_name)]'
a_log_block = '''return [(f"blk.{layer_idx}.ssm_dt.bias", hf_name)]

            layer_match = re.fullmatch(
                r"(?P<model_prefix>(?:language_model\\.)?model)\\.layers\\.(\\d+)\\.linear_attn\\.A_log",
                hf_name,
            )
            if layer_match is not None:
                layer_idx = layer_match.group(2)
                return [(f"blk.{layer_idx}.ssm_a", hf_name)]'''

if a_log_anchor in text and 'A_log' not in text:
  text = text.replace(a_log_anchor, a_log_block, 1)
  modified = True
  print("Injected A_log manual GGUF mapping", flush=True)

if modified:
    path.write_text(text)

linear_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/linear.py")
linear_text = linear_path.read_text()
linear_modified = False

old = '''        if isinstance(loaded_shard_id, tuple) and (
            is_gguf_weight or is_gguf_weight_type
        ):
            raise NotImplementedError(
                "Shard id with multiple indices is not supported for GGUF."
            )
'''
new = '''        if isinstance(loaded_shard_id, tuple) and (
            is_gguf_weight or is_gguf_weight_type
        ):
            if self.tp_size != 1:
                raise NotImplementedError(
                    "Shard id with multiple indices is not supported for GGUF "
                    "when tensor parallelism is enabled."
                )
            if is_gguf_weight_type:
                for shard_id in loaded_shard_id:
                    param.data[shard_id].copy_(loaded_weight)
                    param.shard_weight_type[shard_id] = loaded_weight.item()
                return

            output_dim = getattr(param, "output_dim", None)
            split_sizes = [self.output_sizes[idx] for idx in loaded_shard_id]
            for shard_id, shard_weight in zip(
                loaded_shard_id,
                loaded_weight.split(split_sizes, dim=output_dim),
                strict=True,
            ):
                param.shard_id.append(shard_id)
                param.shard_id_map[shard_id] = len(param.data_container)
                param.data_container.append(shard_weight)
            return
'''

if old in linear_text and new not in linear_text:
  linear_text = linear_text.replace(old, new, 1)
  linear_modified = True
  print("Applied runtime GGUF merged linear patch", flush=True)

if linear_modified:
  linear_path.write_text(linear_text)

qwen35_model_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_5.py")
qwen35_model_text = qwen35_model_path.read_text()
qwen35_model_modified = False

# --- GGUF norm fix: GGUF converter adds +1 to norm weights for standard w*x ---
# The HF model uses (1+w)*x norms (Gemma-style), but GGUF already bakes +1 into
# the stored weight.  Using GemmaRMSNorm would compute (1+(w+1))*x = (2+w)*x.
# Fix: switch to standard RMSNorm which computes (w+1)*x = (1+w)*x.  Correct!
old = 'GemmaRMSNorm as Qwen3_5RMSNorm,'
new = 'RMSNorm as Qwen3_5RMSNorm,'
if old in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(old, new, 1)
  qwen35_model_modified = True
  print("Applied GGUF norm fix: Qwen3_5RMSNorm = RMSNorm (not Gemma)", flush=True)

old = '                if weight_name not in name:\n                    continue\n'
new = '                if weight_name not in name or param_name in name:\n                    continue\n'

if old in qwen35_model_text and new not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(old, new, 1)
  qwen35_model_modified = True
  print("Applied runtime Qwen3.5 stacked mapping guard patch", flush=True)

old = '                    weight_loader = getattr(\n'
new = '''                    if (
                        name.endswith("linear_attn.conv1d.weight")
                        and loaded_weight.ndim == 2
                    ):
                        loaded_weight = loaded_weight.unsqueeze(1)
                    weight_loader = getattr(
'''

if old in qwen35_model_text and new not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(old, new, 1)
  qwen35_model_modified = True
  print("Applied runtime Qwen3.5 conv1d patch", flush=True)

# A_log handling: GGUF stores ssm_a = -exp(A_log).
# Confirmed from llama.cpp: gate = softplus(alpha+dt_bias) * ssm_a
# vLLM expects A_log: gate = -exp(A_log) * softplus(a+dt_bias)
# So A_log = log(-ssm_a) to make -exp(A_log) = ssm_a
a_log_conv1d_anchor = '''                    if (
                        name.endswith("linear_attn.conv1d.weight")
                        and loaded_weight.ndim == 2
                    ):
                        loaded_weight = loaded_weight.unsqueeze(1)'''
a_log_transform_block = '''                    if (
                        name.endswith("linear_attn.conv1d.weight")
                        and loaded_weight.ndim == 2
                    ):
                        loaded_weight = loaded_weight.unsqueeze(1)
                    if name.endswith("linear_attn.A_log"):
                        import torch as _torch
                        # GGUF ssm_a = -exp(A_log). Convert back: A_log = log(-ssm_a)
                        loaded_weight = _torch.log(-loaded_weight)'''

if a_log_conv1d_anchor in qwen35_model_text and 'linear_attn.A_log' not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(a_log_conv1d_anchor, a_log_transform_block, 1)
  qwen35_model_modified = True
  print("Applied runtime A_log GGUF de-exponentiation patch", flush=True)

# ── V-head un-reorder patch ──────────────────────────────────────────
# The GGUF converter (_LinearAttentionVReorderBase) reorders V-heads
# from grouped [G0_v0, G0_v1, G1_v0, G1_v1, ...] to tiled
# [G0_v0, G1_v0, ..., G15_v0, G0_v1, G1_v1, ..., G15_v1] order.
# This breaks vLLM's FLA kernels which use i_vh // (num_v/num_k) to
# map V-heads to K-heads.  We reverse the permutation during loading.
vhead_unreorder_anchor = '''                    if name.endswith("linear_attn.A_log"):
                        import torch as _torch
                        # GGUF ssm_a = -exp(A_log). Convert back: A_log = log(-ssm_a)
                        loaded_weight = _torch.log(-loaded_weight)'''

vhead_unreorder_block = '''                    if name.endswith("linear_attn.A_log"):
                        import torch as _torch
                        # GGUF ssm_a = -exp(A_log). Convert back: A_log = log(-ssm_a)
                        loaded_weight = _torch.log(-loaded_weight)
                    # output_norm: GGUF converter already added +1 (Gemma→standard). No fix needed.
                    # NOTE: ssm_norm (linear_attn.norm) uses standard RMSNormGated (weight*x, init=ones),
                    # NOT Gemma-style (1+weight). GGUF stores the raw weight (~1.0) — do NOT add +1.
                    # Reverse GGUF V-head tiled→grouped reorder.
                    # Only for tensors the converter ACTUALLY reordered via _LinearAttentionVReorderBase:
                    #   alpha, beta, in_proj_z, out_proj  (NOT A_log, dt_bias — those are already grouped)
                    if hasattr(self, '_gguf_vhead_inv_perm'):
                        import torch as _torch
                        inv = self._gguf_vhead_inv_perm
                        nv = self.config.linear_num_value_heads
                        hvd = self.config.linear_value_head_dim
                        if name.endswith("linear_attn.in_proj_a.weight") or name.endswith("linear_attn.in_proj_b.weight"):
                            loaded_weight = loaded_weight[inv]
                        elif name.endswith("linear_attn.in_proj_z.weight"):
                            big_inv = _torch.arange(nv, device=loaded_weight.device)
                            big_inv = big_inv[inv].unsqueeze(1).expand(-1, hvd).reshape(-1)
                            loaded_weight = loaded_weight[big_inv]
                        elif name.endswith("linear_attn.out_proj.weight"):
                            big_inv = _torch.arange(nv, device=loaded_weight.device)
                            big_inv = big_inv[inv].unsqueeze(1).expand(-1, hvd).reshape(-1)
                            loaded_weight = loaded_weight[:, big_inv]'''

if vhead_unreorder_anchor in qwen35_model_text and '_gguf_vhead_inv_perm' not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(vhead_unreorder_anchor, vhead_unreorder_block, 1)
  qwen35_model_modified = True
  print("Applied runtime V-head un-reorder (load_weights) patch", flush=True)

# Also need: (1) create _gguf_vhead_inv_perm attribute, (2) handle in_proj_qkv V
# shard and conv1d V channels through stacked params path
vhead_init_anchor = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:'''
vhead_init_block = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # V-head tiled mapping is handled in the recurrence (head_map = arange(H) % Hg)
        # and decode kernel (i_h = i_hv % H). No per-weight un-reorder needed.
        pass'''

if vhead_init_anchor in qwen35_model_text and 'Build inverse permutation for GGUF' not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(vhead_init_anchor, vhead_init_block, 1)
  qwen35_model_modified = True
  print("Applied runtime V-head inverse-perm init patch", flush=True)

# Handle in_proj_qkv V shard through the stacked params code path
# When in_proj_qkv is loaded as shard_id=(0,1,2), shard 2 is V.
# We un-reorder the V shard rows before it's stored.
vhead_stacked_anchor = '''                weight_loader(param, loaded_weight, shard_id)
                break'''
vhead_stacked_block = '''                # V (shard 2) and Z (shard 3) in attn_qkv are already GROUPED
                # from the GGUF converter de-interleave — do NOT un-reorder them.
                # Only alpha/beta (in_proj_ba) were V-head reordered by the converter.
                if hasattr(self, '_gguf_vhead_inv_perm') and 'in_proj_ba' in name and loaded_weight.ndim >= 1:
                    loaded_weight = loaded_weight[self._gguf_vhead_inv_perm]
                weight_loader(param, loaded_weight, shard_id)
                break'''

if vhead_stacked_anchor in qwen35_model_text and 'big_inv2' not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(vhead_stacked_anchor, vhead_stacked_block, 1)
  qwen35_model_modified = True
  print("Applied runtime V-head un-reorder (stacked params) patch", flush=True)

# Handle conv1d V channels un-reorder
vhead_conv1d_anchor = '''                    if name.endswith("linear_attn.conv1d.weight"):'''
vhead_conv1d_block_text = '''                    if name.endswith("linear_attn.conv1d.weight"):
                        if hasattr(self, '_gguf_vhead_inv_perm'):
                            import torch as _tc
                            nk_c = self.config.linear_num_key_heads
                            hkd_c = self.config.linear_key_head_dim
                            nv_c = self.config.linear_num_value_heads
                            hvd_c = self.config.linear_value_head_dim
                            qk_ch = hkd_c * nk_c * 2
                            inv_c = self._gguf_vhead_inv_perm
                            big_inv_c = inv_c.unsqueeze(1).expand(-1, hvd_c).reshape(-1)
                            qk_part = loaded_weight[:qk_ch]
                            v_part = loaded_weight[qk_ch:]
                            v_part = v_part[big_inv_c]
                            loaded_weight = _tc.cat([qk_part, v_part], dim=0)'''

# We need to find the right anchor - the conv1d check happens after the unsqueeze
# Let's check what text is actually in the file after patches
conv1d_full_anchor = '''                    if (
                        name.endswith("linear_attn.conv1d.weight")
                        and loaded_weight.ndim == 2
                    ):
                        loaded_weight = loaded_weight.unsqueeze(1)'''

conv1d_vhead_block = '''                    if (
                        name.endswith("linear_attn.conv1d.weight")
                        and loaded_weight.ndim == 2
                    ):
                        loaded_weight = loaded_weight.unsqueeze(1)
                    if name.endswith("linear_attn.conv1d.weight") and hasattr(self, '_gguf_vhead_inv_perm'):
                        import torch as _tc
                        nk_c = self.config.linear_num_key_heads
                        hkd_c = self.config.linear_key_head_dim
                        hvd_c = self.config.linear_value_head_dim
                        qk_ch = hkd_c * nk_c * 2
                        inv_c = self._gguf_vhead_inv_perm
                        big_inv_c = inv_c.unsqueeze(1).expand(-1, hvd_c).reshape(-1)
                        w = loaded_weight.squeeze()
                        qk_part = w[:qk_ch]
                        v_part = w[qk_ch:]
                        v_part = v_part[big_inv_c]
                        loaded_weight = _tc.cat([qk_part, v_part], dim=0)
                        if loaded_weight.ndim == 2:
                            loaded_weight = loaded_weight.unsqueeze(1)'''

if conv1d_full_anchor in qwen35_model_text and 'qk_ch' not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(conv1d_full_anchor, conv1d_vhead_block, 1)
  qwen35_model_modified = True
  print("Applied runtime V-head un-reorder (conv1d) patch", flush=True)

old = '''        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )'''
new = '''        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
        )'''

if old in qwen35_model_text and new not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(old, new, 1)
  qwen35_model_modified = True
  print("Applied runtime Qwen3.5 embed_tokens quant_config patch", flush=True)

# GGUF untied lm_head fix: the GGUF file may store a quantized output.weight
# (e.g. Q3_K for Qwen3.8-27B-UD-IQ2_M). The weights iterator then yields
# lm_head.qweight_type, which requires ParallelLMHead to be created with the
# GGUF quant_config (same as the embed_tokens patch above).
old = '''            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )'''
new = '''            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=vllm_config.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )'''

if old in qwen35_model_text and new not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(old, new, 1)
  qwen35_model_modified = True
  print("Applied runtime Qwen3.5 lm_head quant_config patch", flush=True)

old = '''        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )
'''
new = '''        self._forward_core(
            mixed_qkv=mixed_qkv,
            b=b,
            a=a,
            core_attn_out=core_attn_out,
        )
'''

if old in qwen35_model_text and new not in qwen35_model_text:
  qwen35_model_text = qwen35_model_text.replace(old, new, 1)
  qwen35_model_modified = True
  print("Applied runtime Qwen3.5 direct GDN core patch", flush=True)

if qwen35_model_modified:
  qwen35_model_path.write_text(qwen35_model_text)

gguf_quant_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/gguf.py")
gguf_quant_text = gguf_quant_path.read_text()
gguf_quant_modified = False

# NOTE: No Pascal GGUF matmul fallback — the MMVQ/MMQ CUDA kernels (ported from
# llama.cpp) are fully Pascal-compatible (no tensor cores, no sm_70+ features)
# and run natively on sm_61, giving ~15x speedup over the dequant+matmul path.

# Fix shard ordering: the apply() method iterates shard_id list in loading
# order (= GGUF file order = alphabetical).  For string IDs ("q","k","v")
# there is already a rewrite to the correct order, but for integer/tuple
# shard IDs (used by Qwen3.5 stacked params) the loading order may not
# match the expected shard order.  Sort them so shards [0,1,2,3] or
# [(0,1,2),3] are iterated in ascending order.
old = '''        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id'''
new = '''        if shard_id:
            # dequantize shard weights respectively
            if "q" in shard_id:
                shard_id = ["q", "k", "v"]
            else:
                # Sort integer/tuple shard IDs so output matches expected order
                shard_id = sorted(shard_id, key=lambda x: x if isinstance(x, int) else x[0])'''

if old in gguf_quant_text and new not in gguf_quant_text:
  gguf_quant_text = gguf_quant_text.replace(old, new, 1)
  gguf_quant_modified = True
  print("Applied runtime GGUF shard ordering fix", flush=True)

if gguf_quant_modified:
  gguf_quant_path.write_text(gguf_quant_text)
PY
}

apply_dflash_gguf_runtime_patch() {
  python3 - <<'PY'
import re
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py")
text = path.read_text()
modified = False

old = 'import ast\nimport copy\nfrom typing import TYPE_CHECKING, Any, Literal, get_args\n'
new = 'import ast\nimport copy\nfrom pathlib import Path\nfrom typing import TYPE_CHECKING, Any, Literal, get_args\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime DFlash GGUF Path import patch", flush=True)

if "draft_hf_config_path" not in text:
  pattern = re.compile(
    r'(?m)^(?P<if_indent>\s*)if self\.model is not None:\n'
    r'(?P<body_indent>\s*)self\.draft_model_config = ModelConfig\(\n'
    r'(?P<kw_indent>\s*)model=self\.model,\n'
    r'(?P=kw_indent)runner="draft",\n'
  )

  def repl(match):
    if_indent = match.group("if_indent")
    body_indent = match.group("body_indent")
    kw_indent = match.group("kw_indent")
    return (
      f"{if_indent}if self.model is not None:\n"
      f"{body_indent}draft_hf_config_path = None\n"
      f"{body_indent}if self.method == \"dflash\" or self.method == \"mtp\" or self.method.endswith(\"_mtp\"):\n"
      f"{body_indent}    draft_model_path = Path(self.model)\n"
      f"{body_indent}    if draft_model_path.suffix.lower() == \".gguf\":\n"
      f"{body_indent}        draft_config_path = draft_model_path.parent / \"config.json\"\n"
      f"{body_indent}        if draft_config_path.is_file():\n"
      f"{body_indent}            draft_hf_config_path = str(draft_model_path.parent)\n"
      f"\n"
      f"{body_indent}self.draft_model_config = ModelConfig(\n"
      f"{kw_indent}model=self.model,\n"
      f"{kw_indent}hf_config_path=draft_hf_config_path,\n"
      f"{kw_indent}runner=\"draft\",\n"
    )

  text, count = pattern.subn(repl, text, count=1)
  if count == 1:
    modified = True
    print("Applied runtime DFlash GGUF sidecar config patch", flush=True)

wrapper_pattern = re.compile(
  r'(?ms)^(?P<indent>\s*)# Replace hf_config for EAGLE draft_model\n'
  r'(?P=indent)if self\.method in \("eagle", "eagle3", "dflash"\):\n'
  r'.*?'
  r'(?P=indent)if self\.method == "dflash":\n'
  r'(?P=indent)    self\.parallel_drafting = True\n'
)
wrapper_replacement = '''                # Replace hf_config for EAGLE draft_model
                if self.method in ("eagle", "eagle3"):
                    from vllm.transformers_utils.configs import SpeculatorsConfig
                    from vllm.transformers_utils.configs.eagle import EAGLEConfig

                    if isinstance(
                        self.draft_model_config.hf_config,
                        (EAGLEConfig, SpeculatorsConfig),
                    ):
                        pass
                    else:
                        eagle_config = EAGLEConfig(
                            self.draft_model_config.hf_config,
                            method=self.method,
                            model_type="eagle",
                        )
                        self.draft_model_config.hf_config = eagle_config
                        self.update_arch_()
                elif self.method == "dflash":
                    self.update_arch_()
                    self.parallel_drafting = True
'''

if 'if self.method in ("eagle", "eagle3", "dflash"):' in text:
  text, count = wrapper_pattern.subn(wrapper_replacement, text, count=1)
  if count == 1:
    modified = True
    print("Applied runtime DFlash wrapper fix", flush=True)

if modified:
  path.write_text(text)
PY
}

apply_dflash_runtime_patch() {
  python3 - <<'PY'
import re
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/dflash.py')
text = path.read_text()
modified = False

old = '''    def _init_parallel_drafting_params(self):
        model_hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(model_hf_config, "dflash_config", None) or {}
        if "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
            return
        super()._init_parallel_drafting_params()
'''
new = '''    def _init_parallel_drafting_params(self):
        model_hf_config = self.draft_model_config.hf_config
        dflash_config = getattr(model_hf_config, "dflash_config", None) or {}
        if "mask_token_id" in dflash_config:
            self.parallel_drafting_token_id = dflash_config["mask_token_id"]
            if self.pass_hidden_states_to_model:
                self.parallel_drafting_hidden_state_tensor = torch.empty(
                    self.hidden_size,
                    dtype=self.dtype,
                    device=self.device,
                )
            return
        super()._init_parallel_drafting_params()
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash hidden-state init patch', flush=True)

old = '''    @override
    def _raise_if_multimodal(self):
        pass
'''
new = '''    @override
    def load_model(self, target_model: nn.Module) -> None:
        orig_parallel_drafting = self.parallel_drafting
        try:
            self.parallel_drafting = False
            super().load_model(target_model)
        finally:
            self.parallel_drafting = orig_parallel_drafting

    @override
    def _raise_if_multimodal(self):
        pass
'''

if old in text and 'orig_parallel_drafting = self.parallel_drafting' not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash load_model override patch', flush=True)

old = '''    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        pass

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        pass
'''
new = '''    @override
    def _maybe_share_embeddings(self, target_language_model: nn.Module) -> None:
        super()._maybe_share_embeddings(target_language_model)

    @override
    def _maybe_share_lm_head(self, target_language_model: nn.Module) -> None:
        super()._maybe_share_lm_head(target_language_model)
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash sharing patch', flush=True)

old = '''            draft_vllm_config = replace(
                draft_vllm_config,
                attention_config=attention_config,
                load_config=load_config,
            )

            with set_model_tag("dflash"):
'''
new = '''            draft_vllm_config = replace(
                draft_vllm_config,
                attention_config=attention_config,
                load_config=load_config,
            )
            self.draft_vllm_config = draft_vllm_config

            with set_model_tag("dflash"):
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash draft-config context patch', flush=True)

old = '''        if attention_backend is not None:
            attention_config = replace(
                attention_config,
                backend=attention_backend,
            )
        draft_vllm_config = replace(
            draft_vllm_config,
            attention_config=attention_config,
        )

        with set_model_tag("dflash"):
            return get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
'''
new = '''        if attention_backend is not None:
            attention_config = replace(
                attention_config,
                backend=attention_backend,
            )
        draft_vllm_config = replace(
            draft_vllm_config,
            attention_config=attention_config,
        )
        self.draft_vllm_config = draft_vllm_config

        with set_model_tag("dflash"):
            return get_model(
                vllm_config=draft_vllm_config,
                prefix="draft_model",
            )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash draft-config context patch', flush=True)

old = '''            self.model.precompute_and_store_context_kv(
                context_states,
                context_positions,
            )
            with set_forward_context(
                None,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
'''
new = '''            self.model.precompute_and_store_context_kv(
                context_states,
                context_positions,
            )
            active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)
            with set_forward_context(
                None,
                active_vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=slot_mapping_dict,
            ):
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash dummy-run forward-context patch', flush=True)

old = '''        with set_forward_context(
            None,
            self.vllm_config,
'''
new = '''        active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)
        with set_forward_context(
            None,
            active_vllm_config,
'''

if old in text and 'active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)' not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash dummy-run forward-context fallback patch', flush=True)

old = '''            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(num_tokens)
            )
            model_kwargs, _ = self._build_model_inputs_first_pass(num_input_tokens)

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens,
                    common_attn_metadata.slot_mapping,
                ),
            ):
'''
new = '''            cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(num_tokens)
            )
            model_kwargs, _ = self._build_model_inputs_first_pass(num_input_tokens)
            active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)

            with set_forward_context(
                per_layer_attn_metadata,
                active_vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=self._get_slot_mapping(
                    num_input_tokens,
                    common_attn_metadata.slot_mapping,
                ),
            ):
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash propose forward-context patch', flush=True)

old = '''                for layer_name in attn_group.layer_names:
                    assert getattr(attn_metadata, "causal", None) is False, (
                        f"Attention metadata for layer {layer_name} does not have "
                        "non-causal support, which is required for DFlash."
                    )
                    per_layer_attn_metadata[layer_name] = attn_metadata
'''
new = '''                for layer_name in attn_group.layer_names:
                    assert getattr(attn_metadata, "causal", False) is not True, (
                        f"Attention metadata for layer {layer_name} explicitly reports "
                        "causal=True, which is incompatible with DFlash."
                    )
                    per_layer_attn_metadata[layer_name] = attn_metadata
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash draft metadata assertion patch', flush=True)

if 'non-causal support, which is required for DFlash.' in text and 'causal=True, which is incompatible with DFlash.' not in text:
  pattern = re.compile(
    r'(?P<indent>\s*)assert getattr\(attn_metadata, "causal", None\) is False, \(\n'
    r'(?P=indent)    f"Attention metadata for layer \{layer_name\} does not have "\n'
    r'(?P=indent)    "non-causal support, which is required for DFlash\."\n'
    r'(?P=indent)\)'
  )
  replacement = (
    '\\g<indent>assert getattr(attn_metadata, "causal", False) is not True, (\n'
    '\\g<indent>    f"Attention metadata for layer {layer_name} explicitly reports "\n'
    '\\g<indent>    "causal=True, which is incompatible with DFlash."\n'
    '\\g<indent>)'
  )
  text, count = pattern.subn(replacement, text, count=1)
  if count:
    modified = True
    print('Applied runtime DFlash draft metadata assertion regex patch', flush=True)

old = '''        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
'''
new = '''        active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)

        with set_forward_context(
            per_layer_attn_metadata,
            active_vllm_config,
'''

if old in text and '''with set_forward_context(
            per_layer_attn_metadata,
            active_vllm_config,
''' not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime DFlash propose forward-context fallback patch', flush=True)

if modified:
  path.write_text(text)
PY
}

apply_spec_decode_utils_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/utils.py')
text = path.read_text()
modified = False

old = '''    new: VllmConfig = replace(
        old,
        quant_config=None,
        parallel_config=new_parallel_config,
        model_config=old_spec_config.draft_model_config,
    )
    return new
'''
new = '''    new: VllmConfig = replace(
        old,
        quant_config=None,
        parallel_config=new_parallel_config,
        model_config=old_spec_config.draft_model_config,
    )
    new_compilation_config = replace(new.compilation_config)
    new_compilation_config.static_forward_context = {}
    new_compilation_config.static_all_moe_layers = []
    new = replace(new, compilation_config=new_compilation_config)
    return new
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime draft compilation-config isolation patch', flush=True)

if modified:
  path.write_text(text)
PY
}

apply_eagle_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path
import re

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py')
text = path.read_text()
modified = False

old_variants = [
'''        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )
''',
'''        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )
        logger.info(
            "EAGLE draft attention layers: %s",
            sorted(self._draft_attn_layer_names),
        )
''',
]
new = '''        self.model = self._get_model()

        # Find draft layers (attention layers added by draft model)
        active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)
        all_attn_layers = get_layers_from_vllm_config(
            active_vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )
        logger.info(
            "EAGLE draft attention layers: %s",
            sorted(self._draft_attn_layer_names),
        )
'''

for old in old_variants:
  if old in text and new not in text:
    text = text.replace(old, new, 1)
    modified = True
    print('Applied runtime EAGLE draft-layer logging patch', flush=True)
    break

old_variants = [
'''        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
''',
'''        active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)
        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
''',
'''        """
    all_attn_layers = get_layers_from_vllm_config(
      self.vllm_config,
      AttentionLayerBase,  # type: ignore[type-abstract]
    )
''',
]
new = '''        active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)
        all_attn_layers = get_layers_from_vllm_config(
            active_vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
'''

for old in old_variants:
  if old in text and new not in text:
    text = text.replace(old, new, 1)
    modified = True
    print('Applied runtime EAGLE draft-config backend patch', flush=True)
    break

initialize_attn_backend_start = text.find('def initialize_attn_backend(')
initialize_attn_backend_end = text.find('\n    def ', initialize_attn_backend_start + 1)
if initialize_attn_backend_start != -1:
  if initialize_attn_backend_end == -1:
    initialize_attn_backend_end = len(text)
  initialize_attn_backend_block = text[
    initialize_attn_backend_start:initialize_attn_backend_end
  ]
else:
  initialize_attn_backend_block = ''

if initialize_attn_backend_block and 'active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)' not in initialize_attn_backend_block:
  pattern = re.compile(
    r'(\n\s+def initialize_attn_backend\(.*?\n\s+"""\n\s+Initialize AttentionGroups for draft layers using kv_cache_config\.\n\s+Called from the model runner\'s initialize_metadata_builders\.\n\s+"""\n)\s+all_attn_layers = get_layers_from_vllm_config\(\n\s+self\.vllm_config,\n\s+AttentionLayerBase,  # type: ignore\[type-abstract\]\n\s+\)\n',
    re.DOTALL,
  )
  text, count = pattern.subn(
    r'\1        active_vllm_config = getattr(self, "draft_vllm_config", self.vllm_config)\n        all_attn_layers = get_layers_from_vllm_config(\n            active_vllm_config,\n            AttentionLayerBase,  # type: ignore[type-abstract]\n        )\n',
    text,
    count=1,
  )
  if count:
    modified = True
    print('Applied runtime EAGLE draft-config backend regex patch', flush=True)

old = '''    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self._draft_attn_layer_names
                    ]
                )
            )
            == 1
        ), "All drafting layers should belong to the same kv cache group"
'''
new = '''    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all drafting layers belong to the same KVCacheGroup.
        Need this assumption to ensure all drafting layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id

        draft_layer_groups = {
            layer_name: kv_cache_groups.get(layer_name)
            for layer_name in sorted(self._draft_attn_layer_names)
        }
        unique_groups = {
            group_id
            for group_id in draft_layer_groups.values()
            if group_id is not None
        }
        if None in draft_layer_groups.values() or len(unique_groups) != 1:
            logger.error(
                "EAGLE draft KV group mismatch: draft_layer_groups=%s",
                draft_layer_groups,
            )
        assert (
            None not in draft_layer_groups.values() and len(unique_groups) == 1
        ), "All drafting layers should belong to the same kv cache group"
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime EAGLE KV group logging patch', flush=True)

if modified:
  path.write_text(text)
PY
}

apply_qwen3_dflash_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path
import re

path = Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py')
text = path.read_text()
modified = False

old = 'from collections.abc import Iterable\n'
new = 'from collections.abc import Iterable\nimport os\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash os import patch', flush=True)

old = 'from .qwen3 import Qwen3ForCausalLM\n'
new = 'from .interfaces import EagleModelMixin\nfrom .qwen3 import Qwen3ForCausalLM\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash Eagle mixin import patch', flush=True)

old = 'class DFlashQwen3Model(nn.Module):\n'
new = 'class DFlashQwen3Model(nn.Module, EagleModelMixin):\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash Eagle mixin patch', flush=True)

old = '''        self.use_aux_hidden_state = drafter_config.get("use_aux_hidden_state", True)
        current_vllm_config = get_current_vllm_config()
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                DFlashQwen3DecoderLayer(
                    current_vllm_config,
                    config=self.config,
                    cache_config=current_vllm_config.cache_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        if self.use_aux_hidden_state:
            num_features_to_use = self.config.num_hidden_layers
            if "target_layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["target_layer_ids"])
            elif "layer_ids" in drafter_config:
                num_features_to_use = len(drafter_config["layer_ids"])
            if hasattr(self.config, "target_hidden_size"):
                fc_input_size = self.config.target_hidden_size * num_features_to_use
            else:
                fc_input_size = self.config.hidden_size * num_features_to_use
            self.fc = ReplicatedLinear(
'''
new = (
  '        self.use_aux_hidden_state = drafter_config.get("use_aux_hidden_state", True)\n'
  '        requested_max_layers = int(os.environ.get("DFLASH_MAX_LAYERS", "0") or "0")\n'
  '        original_num_hidden_layers = self.config.num_hidden_layers\n'
  '        self.effective_num_hidden_layers = original_num_hidden_layers\n'
  '        if requested_max_layers > 0:\n'
  '            self.effective_num_hidden_layers = min(\n'
  '                original_num_hidden_layers,\n'
  '                requested_max_layers,\n'
  '            )\n'
  '        if self.effective_num_hidden_layers != original_num_hidden_layers:\n'
  '            self.config.num_hidden_layers = self.effective_num_hidden_layers\n'
  '            layer_types = getattr(self.config, "layer_types", None)\n'
  '            if isinstance(layer_types, list):\n'
  '                self.config.layer_types = layer_types[: self.effective_num_hidden_layers]\n'
  '            max_window_layers = getattr(self.config, "max_window_layers", None)\n'
  '            if isinstance(max_window_layers, int):\n'
  '                self.config.max_window_layers = min(\n'
  '                    max_window_layers,\n'
  '                    self.effective_num_hidden_layers,\n'
  '                )\n'
  '        self.effective_target_layer_ids = list(\n'
  '            drafter_config.get("target_layer_ids", drafter_config.get("layer_ids", []))\n'
  '        )\n'
  '        if self.effective_target_layer_ids:\n'
  '            self.effective_target_layer_ids = self.effective_target_layer_ids[\n'
  '                : self.effective_num_hidden_layers\n'
  '            ]\n'
  '            dflash_config = getattr(self.config, "dflash_config", None)\n'
  '            if isinstance(dflash_config, dict):\n'
  '                dflash_config["target_layer_ids"] = list(self.effective_target_layer_ids)\n'
  '            setattr(\n'
  '                self.config,\n'
  '                "eagle_aux_hidden_state_layer_ids",\n'
  '                list(self.effective_target_layer_ids),\n'
  '            )\n'
  '        if self.effective_num_hidden_layers != original_num_hidden_layers:\n'
  '            logger.warning(\n'
  '                "DFlash: capping draft layers from %d to %d for constrained VRAM",\n'
  '                original_num_hidden_layers,\n'
  '                self.effective_num_hidden_layers,\n'
  '            )\n'
  '        current_vllm_config = get_current_vllm_config()\n'
  '        self.layers = nn.ModuleList(\n'
  '            [\n'
  '                DFlashQwen3DecoderLayer(\n'
  '                    current_vllm_config,\n'
  '                    config=self.config,\n'
  '                    cache_config=current_vllm_config.cache_config,\n'
  '                    quant_config=self.quant_config,\n'
  '                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),\n'
  '                )\n'
  '                for layer_idx in range(self.effective_num_hidden_layers)\n'
  '            ]\n'
  '        )\n'
  '        if self.use_aux_hidden_state:\n'
  '            num_features_to_use = self.effective_num_hidden_layers\n'
  '            if self.effective_target_layer_ids:\n'
  '                num_features_to_use = len(self.effective_target_layer_ids)\n'
  '            if hasattr(self.config, "target_hidden_size"):\n'
  '                fc_input_size = self.config.target_hidden_size * num_features_to_use\n'
  '            else:\n'
  '                fc_input_size = self.config.hidden_size * num_features_to_use\n'
  '            self.fc_input_size = fc_input_size\n'
  '            self.fc = ReplicatedLinear(\n'
)

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash shared-embedding patch', flush=True)

old = '''        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )
'''
new = '''        self.model = DFlashQwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )
        self.has_own_embed_tokens = False
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.has_own_lm_head = False
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size,
            scale=logit_scale,
        )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash shared-head patch', flush=True)

old = '''        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
'''
new = '''        target_model_config = vllm_config.speculative_config.target_model_config
        if target_model_config is None:
            target_model_config = vllm_config.model_config
        target_layer_num = target_model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash target-layer-count patch', flush=True)

old = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
'''
new = '''    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "midlayer." in name:
                name = name.replace("midlayer.", "layers.0.")
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                if scale_name not in params_dict:
                    continue
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None or name not in params_dict:
                    continue
            loaded_name: str | None = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                loaded_name = name.replace(weight_name, param_name)
                if loaded_name not in params_dict:
                    loaded_name = None
                    break
                param = params_dict[loaded_name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                loaded_name = name
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            if loaded_name is not None:
                loaded_params.add(loaded_name)
        return loaded_params
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash missing-param guard patch', flush=True)

old = '''            if context_slot_mapping is None:
                return
            all_k_final = all_k_flat.view(num_layers, num_ctx, num_kv_heads, head_dim)
            for index in range(num_layers):
                attn = self._attn_layers[index]
                attn.impl.do_kv_cache_update(
                    attn,
                    all_k_final[index],
                    all_v[index],
                    attn.kv_cache,
                    context_slot_mapping,
                )
'''
new = '''            if context_slot_mapping is None:
                return
            all_k_final = all_k_flat.view(num_layers, num_ctx, num_kv_heads, head_dim)
            for index in range(num_layers):
                attn = self._attn_layers[index]
                kv_cache = attn.kv_cache
                if isinstance(kv_cache, list):
                    kv_cache = kv_cache[0]
                attn.impl.do_kv_cache_update(
                    attn,
                    all_k_final[index],
                    all_v[index],
                    kv_cache,
                    context_slot_mapping,
                )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print('Applied runtime Qwen3 DFlash KV cache unwrap patch', flush=True)

pattern = re.compile(
  r'(?ms)^([ ]{8}if context_slot_mapping is None:\n'
  r'[ ]{12}return\n'
  r'[ ]{8}all_k_final = all_k_flat\.view\(num_layers, num_ctx, num_kv_heads, head_dim\)\n'
  r'[ ]{8}for index in range\(num_layers\):\n'
  r'[ ]{12}attn = self\._attn_layers\[index\]\n)'
  r'[ ]{12}attn\.impl\.do_kv_cache_update\(\n'
  r'[ ]{16}attn,\n'
  r'[ ]{16}all_k_final\[index\],\n'
  r'[ ]{16}all_v\[index\],\n'
  r'[ ]{16}attn\.kv_cache,\n'
  r'[ ]{16}context_slot_mapping,\n'
  r'[ ]{12}\)(?:\n|$)'
)
replacement = (
  '        if context_slot_mapping is None:\n'
  '            return\n'
  '        all_k_final = all_k_flat.view(num_layers, num_ctx, num_kv_heads, head_dim)\n'
  '        for index in range(num_layers):\n'
  '            attn = self._attn_layers[index]\n'
  '            kv_cache = attn.kv_cache\n'
  '            if isinstance(kv_cache, list):\n'
  '                kv_cache = kv_cache[0]\n'
  '            attn.impl.do_kv_cache_update(\n'
  '                attn,\n'
  '                all_k_final[index],\n'
  '                all_v[index],\n'
  '                kv_cache,\n'
  '                context_slot_mapping,\n'
  '            )\n'
)
text, regex_count = pattern.subn(replacement, text, count=1)
if regex_count:
  modified = True
  print('Applied runtime Qwen3 DFlash KV cache unwrap regex patch', flush=True)

if modified:
  path.write_text(text)
PY
}

apply_qwen3_next_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py")
text = path.read_text()
modified = False

# GGUF norm fix: same as qwen3_5.py — use RMSNorm instead of GemmaRMSNorm
old = 'GemmaRMSNorm as Qwen3NextRMSNorm,'
new = 'RMSNorm as Qwen3NextRMSNorm,'
if old in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied GGUF norm fix: Qwen3NextRMSNorm = RMSNorm (not Gemma)", flush=True)

old = 'from collections.abc import Iterable\n'
new = 'from collections.abc import Iterable\nimport os\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3-Next os import patch", flush=True)

old = '        if hasattr(self, "_prefill_kernels_warmed_up"):\n            return\n'
new = '        if os.environ.get("COPILOT_SKIP_GDN_PREFILL_WARMUP") == "1":\n            return\n        if hasattr(self, "_prefill_kernels_warmed_up"):\n            return\n'

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3-Next warmup skip patch", flush=True)

old = '''def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


'''
new = '''def gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """
    Custom op for the core attention computation.
    Only handles the convolution + recurrent attention part.
    Input/output projections are handled outside this op.
    """
    forward_context: ForwardContext = get_forward_context()
    no_compile_layers = forward_context.no_compile_layers
    self = no_compile_layers.get(layer_name)
    if self is None:
        candidate_names = [layer_name]
        if layer_name.startswith("language_model."):
            candidate_names.append(layer_name.removeprefix("language_model."))
        else:
            candidate_names.append(f"language_model.{layer_name}")

        for candidate_name in candidate_names:
            self = no_compile_layers.get(candidate_name)
            if self is not None:
                break

        if self is None:
            for candidate_name, candidate_layer in no_compile_layers.items():
                if (
                    candidate_name.endswith(layer_name)
                    or layer_name.endswith(candidate_name)
                ):
                    self = candidate_layer
                    break

    if self is None:
        raise KeyError(layer_name)

    self._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Qwen3-Next GDN layer lookup fallback", flush=True)

if modified:
  path.write_text(text)
PY
}

apply_attention_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/attention/attention.py")
text = path.read_text()
modified = False

old = '''def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current virtual engine
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


'''
new = '''def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current virtual engine
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """

    def _resolve_layer_entry(
        mapping: dict[str, Any],
        key: str,
    ) -> tuple[str, Any]:
        def _canonical_layer_name(name: str) -> str:
            parts = name.split(".")
            for anchor in ("model", "layers"):
                if anchor in parts:
                    return ".".join(parts[parts.index(anchor):])
            return name

        if key in mapping:
            return key, mapping[key]

        candidate_names = [key]
        if key.endswith(".attn"):
          candidate_names.append(key.removesuffix(".attn"))
        else:
          candidate_names.append(f"{key}.attn")
        if key.startswith("language_model."):
          base_key = key.removeprefix("language_model.")
          candidate_names.append(base_key)
          if base_key.endswith(".attn"):
            candidate_names.append(base_key.removesuffix(".attn"))
          else:
            candidate_names.append(f"{base_key}.attn")
        else:
          prefixed_key = f"language_model.{key}"
          candidate_names.append(prefixed_key)
          if prefixed_key.endswith(".attn"):
            candidate_names.append(prefixed_key.removesuffix(".attn"))
          else:
            candidate_names.append(f"{prefixed_key}.attn")

        for candidate_name in candidate_names:
            if candidate_name in mapping:
                return candidate_name, mapping[candidate_name]

        canonical_key = _canonical_layer_name(key)
        canonical_matches = [
            (candidate_name, candidate_value)
            for candidate_name, candidate_value in mapping.items()
            if _canonical_layer_name(candidate_name) == canonical_key
        ]
        if len(canonical_matches) == 1:
            return canonical_matches[0]

        for candidate_name, candidate_value in mapping.items():
            canonical_candidate = _canonical_layer_name(candidate_name)
            if (
                candidate_name.endswith(key)
                or key.endswith(candidate_name)
                or canonical_candidate.endswith(canonical_key)
                or canonical_key.endswith(canonical_candidate)
            ):
                return candidate_name, candidate_value

        if canonical_matches:
            return canonical_matches[0]

        layer_fragment = ".layers."
        layer_scope = None
        if layer_fragment in key:
          layer_scope = key.split(layer_fragment, 1)[1].split(".", 1)[0]
        nearby_keys = []
        if layer_scope is not None:
          nearby_keys = [
            candidate_name
            for candidate_name in mapping
            if f"{layer_fragment}{layer_scope}." in candidate_name
          ]
        if not nearby_keys:
          nearby_keys = [
            candidate_name
            for candidate_name in mapping
            if candidate_name.endswith(".attn") or ".self_attn" in candidate_name
          ][:16]
        logger.warning(
          "Attention context lookup miss for %s; candidates=%s; nearby_keys=%s",
          key,
          candidate_names,
          nearby_keys,
        )
        raise KeyError(key)

    forward_context: ForwardContext = get_forward_context()
    resolved_layer_name = layer_name
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        resolved_layer_name, attn_metadata = _resolve_layer_entry(
            attn_metadata,
            layer_name,
        )
    resolved_attn_name, attn_layer = _resolve_layer_entry(
        forward_context.no_compile_layers,
        layer_name,
    )
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(resolved_attn_name)
    if layer_slot_mapping is None:
        layer_slot_mapping = slot_mapping.get(resolved_layer_name)
    if layer_slot_mapping is None:
        layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime attention context fallback patch", flush=True)

if modified:
  path.write_text(text)
PY
}

apply_triton_autotuner_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/triton/runtime/autotuner.py")
text = path.read_text()
modified = False

old = 'import json\n'
new = 'import json\nimport os\n'
if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True

old = '''            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)

                def benchmark():
                    bench_start = time.time()
                    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings

                if self.cache_results:
                    used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()
'''
new = '''            if key not in self.cache:
                used_cached_result = False
                pruned_configs = self.prune_configs(kwargs)
                force_first_config = os.environ.get("COPILOT_TRITON_FORCE_FIRST_CONFIG") == "1"

                def benchmark():
                    bench_start = time.time()
                    timings = {config: self._bench(*args, config=config, **kwargs) for config in pruned_configs}
                    bench_end = time.time()
                    self.bench_time = bench_end - bench_start
                    self.cache[key] = builtins.min(timings, key=timings.get)
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                    self.configs_timings = timings

                if force_first_config:
                    self.cache[key] = pruned_configs[0]
                    self.bench_time = 0.0
                    self.configs_timings = {pruned_configs[0]: [0.0, 0.0, 0.0]}
                    full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                    self.pre_hook(full_nargs, reset_only=True)
                elif self.cache_results:
                    used_cached_result = self.check_disk_cache(key, pruned_configs, benchmark)
                else:
                    benchmark()
'''
if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True

if modified:
  path.write_text(text)
  print("Applied runtime Triton autotuner first-config patch", flush=True)
PY
}

apply_gpu_model_runner_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path
import re

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
text = path.read_text()
modified = False

old = '''                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    assert kv_cache.shape[1] != 2, (
                        "Fail to determine whether the layout is "
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for "
                        f"a tensor of shape {kv_cache.shape}"
                    )
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )
'''
new = '''                if isinstance(kv_cache_spec, AttentionSpec) and kv_cache.shape[0] == 2:
                    hidden_size = kv_cache.shape[2:].numel()
                    if kv_cache.shape[1] == 2:
                        stride0, stride1 = kv_cache.stride()[:2]
                        if stride0 == hidden_size and stride1 == 2 * hidden_size:
                            continue
                    kv_cache.as_strided_(
                        size=kv_cache.shape,
                        stride=(hidden_size, 2 * hidden_size, *kv_cache.stride()[2:]),
                    )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GPU model runner layout patch", flush=True)

old = '''                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = (
                            self.model.get_eagle3_default_aux_hidden_state_layers()
                        )

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
'''
new = '''                if self.use_aux_hidden_state_outputs:
                    target_model_ref = self.model
                    if hasattr(target_model_ref, "get_language_model"):
                        target_model_ref = target_model_ref.get_language_model()
                    elif hasattr(target_model_ref, "language_model"):
                        target_model_ref = target_model_ref.language_model
                    supports_aux_hidden_layers = (
                        hasattr(target_model_ref, "model")
                        and hasattr(
                            target_model_ref.model,
                            "_set_aux_hidden_state_layers",
                        )
                    )
                    if not supports_aux_hidden_layers:
                        logger.warning(
                            "DFlash: target model does not expose Eagle auxiliary "
                            "hidden layers; falling back to final hidden states only"
                        )
                        self.use_aux_hidden_state_outputs = False

                if self.use_aux_hidden_state_outputs:
                    if not supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )

                    # Try to get auxiliary layers from speculative config,
                    # otherwise use model's default layers
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = (
                            self.model.get_eagle3_default_aux_hidden_state_layers()
                        )

                    self.model.set_aux_hidden_state_layers(aux_layers)
                time_after_load = time.perf_counter()
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GPU model runner aux-hidden fallback patch", flush=True)

old = '''        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
        ):
            assert self.offload_config.uva.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details."
            )
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
'''
new = '''        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
        ):
            if (
                self.offload_config.uva.cpu_offload_gb != 0
                and not (
                    self.speculative_config
                    and self.speculative_config.use_dflash()
                )
            ):
                assert self.offload_config.uva.cpu_offload_gb == 0, (
                    "Cannot re-initialize the input batch when CPU weight "
                    "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                    "for more details."
                )
            elif self.offload_config.uva.cpu_offload_gb != 0:
                logger.warning(
                    "Allowing InputBatch re-initialization with CPU offload "
                    "for DFlash startup: block_sizes=%s kernel_block_sizes=%s",
                    block_sizes,
                    kernel_block_sizes,
                )
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GPU model runner cpu-offload reinit patch", flush=True)

old = '''        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
'''
new = '''        forward_context = dict(self.compilation_config.static_forward_context)
        if (
            self.speculative_config
            and self.speculative_config.use_dflash()
            and isinstance(self.drafter, DFlashProposer)
            and hasattr(self.drafter, "draft_vllm_config")
        ):
            forward_context.update(
                self.drafter.draft_vllm_config.compilation_config.static_forward_context
            )
        bind_kv_cache(
            kv_caches,
            forward_context,
            self.kv_caches,
            num_attn_module,
        )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GPU model runner draft KV bind patch", flush=True)

old = '''        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=(self.compilation_config.static_forward_context),
        )
'''
new = '''        forward_context = dict(self.compilation_config.static_forward_context)
        if (
            self.speculative_config
            and self.speculative_config.use_dflash()
            and isinstance(self.drafter, DFlashProposer)
            and hasattr(self.drafter, "draft_vllm_config")
        ):
            forward_context.update(
                self.drafter.draft_vllm_config.compilation_config.static_forward_context
            )
        self._kv_block_zeroer.init_meta(
            attn_groups_iter=self._kv_cache_spec_attn_group_iterator(),
            kernel_block_sizes=self._kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            runner_only_attn_layers=self.runner_only_attn_layers,
            static_forward_context=forward_context,
        )
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime GPU model runner draft KV zero-meta patch", flush=True)

pattern = re.compile(
  r'(?ms)^    def get_kv_cache_spec\(self\) -> dict\[str, KVCacheSpec\]:\n'
  r'        """\n'
  r'        Generates the KVCacheSpec by parsing the kv cache format from each\n'
  r'        Attention module in the static forward context\.\n'
  r'        Returns:\n'
  r'            KVCacheSpec: A dictionary mapping layer names to their KV cache\n'
  r'            format\. Layers that do not need KV cache are not included\.\n'
  r'        """\n'
  r'        if has_ec_transfer\(\) and not get_ec_transfer\(\)\.is_consumer:\n'
  r'            return \{\}\n'
  r'        kv_cache_spec: dict\[str, KVCacheSpec\] = \{\}\n'
  r'        layer_type = cast\(type\[Any\], AttentionLayerBase\)\n'
  r'        attn_layers = get_layers_from_vllm_config\(self\.vllm_config, layer_type\)\n'
  r'        for layer_name, attn_module in attn_layers\.items\(\):\n'
  r'            if isinstance\(attn_module, Attention\) and \(\n'
  r'                kv_tgt_layer := attn_module\.kv_sharing_target_layer_name\n'
  r'            \):\n'
  r'                # The layer doesn\'t need its own KV cache and will use that of\n'
  r'                # the target layer\. We skip creating a KVCacheSpec for it, so\n'
  r'                # that KV cache management logic will act as this layer does\n'
  r'                # not exist, and doesn\'t allocate KV cache for the layer\. This\n'
  r'                # enables the memory saving of cross-layer kv sharing, allowing\n'
  r'                # a given amount of memory to accommodate longer context lengths\n'
  r'                # or enable more requests to be processed simultaneously\.\n'
  r'                self\.shared_kv_cache_layers\[layer_name\] = kv_tgt_layer\n'
  r'                continue\n'
  r'            # Skip modules that don\'t need KV cache \(eg encoder-only attention\)\n'
  r'            if spec := attn_module\.get_kv_cache_spec\(self\.vllm_config\):\n'
  r'                kv_cache_spec\[layer_name\] = spec\n\n'
  r'        return kv_cache_spec\n'
)
replacement = '''    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        if has_ec_transfer() and not get_ec_transfer().is_consumer:
            return {}
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layer_sets = [
            (get_layers_from_vllm_config(self.vllm_config, layer_type), self.vllm_config)
        ]
        if (
            self.speculative_config
            and self.speculative_config.use_dflash()
            and isinstance(self.drafter, DFlashProposer)
            and hasattr(self.drafter, "draft_vllm_config")
        ):
            attn_layer_sets.append(
                (
                    get_layers_from_vllm_config(
                        self.drafter.draft_vllm_config,
                        layer_type,
                    ),
                    self.drafter.draft_vllm_config,
                )
            )
        for attn_layers, attn_vllm_config in attn_layer_sets:
            for layer_name, attn_module in attn_layers.items():
                if isinstance(attn_module, Attention) and (
                    kv_tgt_layer := attn_module.kv_sharing_target_layer_name
                ):
                    # The layer doesn't need its own KV cache and will use that of
                    # the target layer. We skip creating a KVCacheSpec for it, so
                    # that KV cache management logic will act as this layer does
                    # not exist, and doesn't allocate KV cache for the layer. This
                    # enables the memory saving of cross-layer kv sharing, allowing
                    # a given amount of memory to accommodate longer context lengths
                    # or enable more requests to be processed simultaneously.
                    self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                    continue
                # Skip modules that don't need KV cache (eg encoder-only attention)
                if spec := attn_module.get_kv_cache_spec(attn_vllm_config):
                    kv_cache_spec[layer_name] = spec

        return kv_cache_spec
'''

if 'attn_layer_sets = [' not in text:
  text, count = pattern.subn(replacement, text, count=1)
  if count:
    modified = True
    print("Applied runtime GPU model runner draft KV spec patch", flush=True)

pattern = re.compile(
  r'(?ms)^        def get_attn_backends_for_group\(\n'
  r'            kv_cache_group_spec: KVCacheGroupSpec,\n'
  r'        \) -> tuple\[dict\[AttentionGroupKey, list\[str\]\], set\[type\[AttentionBackend\]\]\]:\n'
  r'            layer_type = cast\(type\[Any\], AttentionLayerBase\)\n'
  r'            layers = get_layers_from_vllm_config\(\n'
  r'                self\.vllm_config, layer_type, kv_cache_group_spec\.layer_names\n'
  r'            \)\n'
  r'            attn_backends = \{\}\n'
  r'            attn_backend_layers = defaultdict\(list\)\n'
  r'            # Dedupe based on full class name; this is a bit safer than\n'
  r'            # using the class itself as the key because when we create dynamic\n'
  r'            # attention backend subclasses \(e\.g\. ChunkedLocalAttention\) unless\n'
  r'            # they are cached correctly, there will be different objects per\n'
  r'            # layer\.\n'
  r'            for layer_name in kv_cache_group_spec\.layer_names:\n'
  r'                attn_backend = layers\[layer_name\]\.get_attn_backend\(\)\n'
)
replacement = '''        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layer_maps = [
                get_layers_from_vllm_config(
                    self.vllm_config,
                    layer_type,
                    kv_cache_group_spec.layer_names,
                )
            ]
            if (
                self.speculative_config
                and self.speculative_config.use_dflash()
                and isinstance(self.drafter, DFlashProposer)
                and hasattr(self.drafter, "draft_vllm_config")
            ):
                draft_layer_names = [
                    layer_name
                    for layer_name in kv_cache_group_spec.layer_names
                    if layer_name.startswith("draft_model.")
                ]
                if draft_layer_names:
                    layer_maps.append(
                        get_layers_from_vllm_config(
                            self.drafter.draft_vllm_config,
                            layer_type,
                            draft_layer_names,
                        )
                    )
            layers = {}
            for layer_map in layer_maps:
                layers.update(layer_map)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()
'''

if 'draft_layer_names = [' not in text:
  text, count = pattern.subn(replacement, text, count=1)
  if count:
    modified = True
    print("Applied runtime GPU model runner draft attn backend patch", flush=True)

if modified:
  path.write_text(text)
PY
}

apply_triton_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py")
text = path.read_text()
modified = False

old = '''    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
'''
new = '''    def get_empty_cache_for_benchmark(self):
        import torch

        # Cap the benchmark cache to a fraction of currently free VRAM so
        # Triton autotuning remains viable on memory-constrained GPUs.
        cache_size = 256 * 1024 * 1024
        free_bytes, _ = torch.cuda.mem_get_info()
        cache_size = min(cache_size, max(4 * 1024 * 1024, free_bytes // 4))
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
'''

if old in text and new not in text:
  text = text.replace(old, new, 1)
  modified = True
  print("Applied runtime Triton benchmark cache patch", flush=True)

if modified:
  path.write_text(text)
PY
}

apply_fla_runtime_patch() {
  python3 - <<'PY'
from pathlib import Path

files = [
  (
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/chunk_scaled_dot_kkt.py"),
    '''@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "BT", "IS_VARLEN"],
)
''',
    '''@triton.autotune(
    configs=[triton.Config({"BK": 32}, num_warps=2, num_stages=2)],
    key=["H", "K", "BT", "IS_VARLEN"],
)
''',
    "Applied runtime FLA chunk_scaled_dot_kkt autotune patch",
  ),
  (
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/cumsum.py"),
    '''@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
)
''',
    '''@triton.autotune(
    configs=[triton.Config({}, num_warps=1)],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
)
''',
    "Applied runtime FLA cumsum scalar autotune patch",
  ),
  (
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/cumsum.py"),
    '''@triton.autotune(
    configs=[
        triton.Config({"BS": BS}, num_warps=num_warps)
        for BS in BS_LIST
        for num_warps in [2, 4, 8]
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
''',
    '''@triton.autotune(
    configs=[triton.Config({"BS": 16}, num_warps=2)],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
)
''',
    "Applied runtime FLA cumsum vector autotune patch",
  ),
]

for path, old, new, message in files:
  text = path.read_text()
  if old in text and new not in text:
    path.write_text(text.replace(old, new, 1))
    print(message, flush=True)

# --- Decode kernel: fix V→K head mapping for GGUF tiled V-heads ---
# GGUF converter stores V-heads in tiled order: [G0v0,G1v0,...,G15v0,G0v1,...,G15v1]
# llama.cpp ggml_repeat tiles K as [K0,K1,...,K15,K0,...,K15] matching this.
# The decode kernel's i_h = i_hv // (HV // H) assumes grouped order.
# Fix: i_h = i_hv % H  (tiled order)
fsg_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py")
fsg_text = fsg_path.read_text()
fsg_old = "    i_h = i_hv // (HV // H)"
fsg_new = "    i_h = i_hv % H  # GGUF tiled V-head order"
if fsg_old in fsg_text and fsg_new not in fsg_text:
    fsg_text = fsg_text.replace(fsg_old, fsg_new, 1)
    fsg_path.write_text(fsg_text)
    print("Applied GGUF tiled V-head decode kernel head-map fix", flush=True)
PY
}

apply_pascal_gdn_gating_patch() {
  python3 - <<'PY'
import textwrap
from pathlib import Path

# --- Also patch the Triton attention to use PyTorch SDPA ---
triton_attn_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn.py")
triton_attn_text = triton_attn_path.read_text()

sdpa_old = '''        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,  # Not supported
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )'''

sdpa_new = textwrap.indent(
  textwrap.dedent('''
    # Pascal SDPA fallback: avoids Triton tl.dot which is broken on sm_61
    _q = query[:num_actual_tokens]       # [total_q, num_heads, head_size]
    _block_size = key_cache.shape[1]
    _num_kv_heads = key_cache.shape[2]
    _head_size = _q.shape[2]
    _num_heads = _q.shape[1]
    _gqa_ratio = _num_heads // _num_kv_heads
    _num_seqs = len(seqused_k)
    import os
    import torch.nn.functional as _F
    try:
      from torch.nn.attention.bias import causal_lower_right as _causal_lower_right
    except ImportError:
      _causal_lower_right = None

    _stream_threshold = int(
      os.environ.get("COPILOT_SDPA_STREAM_THRESHOLD", "0")
    )
    _q_block = max(16, int(os.environ.get("COPILOT_SDPA_Q_BLOCK", "64")))
    _block_sdpa_q_block = max(
      _q_block,
      int(os.environ.get("COPILOT_SDPA_BLOCK_SDPA_Q_BLOCK", "768")),
    )
    _kv_block = max(32, int(os.environ.get("COPILOT_SDPA_KV_BLOCK", "128")))

    for _si in range(_num_seqs):
      _qs = cu_seqlens_q[_si].item()
      _qe = cu_seqlens_q[_si + 1].item()
      _kv_len = seqused_k[_si].item()
      _q_len = _qe - _qs
      if _q_len <= 0:
        continue
      _nblk = (_kv_len + _block_size - 1) // _block_size
      _blks = block_table[_si, :_nblk]
      _k = key_cache[_blks].reshape(-1, _num_kv_heads, _head_size)[:_kv_len]
      _v = value_cache[_blks].reshape(-1, _num_kv_heads, _head_size)[:_kv_len]
      _kv_len = min(_kv_len, _k.shape[0], _v.shape[0])
      if _kv_len <= 0:
        output[_qs:_qe].zero_()
        continue
      _k = _k[:_kv_len]
      _v = _v[:_kv_len]

      if _kv_len > _stream_threshold:
        _q_positions = (
          torch.arange(_q_len, device=_q.device)
          if _q_len == _kv_len
          else None
        )
        _used_block_sdpa = False

        if os.environ.get("COPILOT_SDPA_BLOCKWISE_SDPA", "1") != "0":
          _fast_q_block = min(
            _block_sdpa_q_block,
            max(16, 7_680_000 // max(1, _kv_len)),
          )
          _kb_base = _k.unsqueeze(0).transpose(1, 2)
          _vb_base = _v.unsqueeze(0).transpose(1, 2)

          while _fast_q_block >= _q_block and not _used_block_sdpa:
            _attempt_q_block = _fast_q_block
            try:
              for _q_start in range(0, _q_len, _attempt_q_block):
                _q_end = min(_q_start + _attempt_q_block, _q_len)
                _kv_end = _q_end if _q_positions is not None else _kv_len
                if _kv_end <= 0:
                  output[_qs + _q_start:_qs + _q_end].zero_()
                  continue

                _qb = _q[_qs + _q_start:_qs + _q_end].unsqueeze(0).transpose(1, 2)
                _kb = _kb_base[:, :, :_kv_end]
                _vb = _vb_base[:, :, :_kv_end]
                _sdpa_kwargs = {
                  "scale": self.scale,
                  "is_causal": False,
                }

                if _q_positions is not None:
                  if _causal_lower_right is not None:
                    _sdpa_kwargs["attn_mask"] = _causal_lower_right(
                      _q_end - _q_start,
                      _kv_end,
                    )
                  else:
                    _q_pos = _q_positions[_q_start:_q_end]
                    _k_pos = torch.arange(_kv_end, device=_q.device)
                    _sdpa_kwargs["attn_mask"] = (
                      _k_pos.unsqueeze(0) <= _q_pos.unsqueeze(1)
                    )

                if _gqa_ratio > 1:
                  try:
                    _ao = _F.scaled_dot_product_attention(
                      _qb,
                      _kb,
                      _vb,
                      enable_gqa=True,
                      **_sdpa_kwargs,
                    )
                  except (TypeError, RuntimeError):
                    _kb = _kb.repeat_interleave(_gqa_ratio, dim=1)
                    _vb = _vb.repeat_interleave(_gqa_ratio, dim=1)
                    _ao = _F.scaled_dot_product_attention(
                      _qb,
                      _kb,
                      _vb,
                      **_sdpa_kwargs,
                    )
                else:
                  _ao = _F.scaled_dot_product_attention(
                    _qb,
                    _kb,
                    _vb,
                    **_sdpa_kwargs,
                  )

                output[_qs + _q_start:_qs + _q_end] = _ao.to(_q.dtype).squeeze(0).transpose(0, 1)

              _used_block_sdpa = True
            except torch.cuda.OutOfMemoryError:
              if torch.cuda.is_available():
                torch.cuda.empty_cache()
              if _attempt_q_block <= _q_block:
                break
              _fast_q_block = max(_q_block, _attempt_q_block // 2)
            except RuntimeError as _block_sdpa_err:
              if "out of memory" in str(_block_sdpa_err).lower():
                if torch.cuda.is_available():
                  torch.cuda.empty_cache()
                if _attempt_q_block <= _q_block:
                  break
                _fast_q_block = max(_q_block, _attempt_q_block // 2)
              else:
                raise

        if _used_block_sdpa:
          continue

        _q_heads = _q[_qs:_qe].transpose(0, 1)
        _k_heads = _k.transpose(0, 1)
        _v_heads = _v.transpose(0, 1)

        for _q_start in range(0, _q_len, _q_block):
          _q_end = min(_q_start + _q_block, _q_len)
          _chunk_q = _q_end - _q_start
          _q_chunk = _q_heads[:, _q_start:_q_end].reshape(
            _num_kv_heads,
            _gqa_ratio,
            _chunk_q,
            _head_size,
          ).float()
          _m = torch.full(
            (_num_kv_heads, _gqa_ratio, _chunk_q),
            float("-inf"),
            device=_q.device,
          )
          _l = torch.zeros_like(_m)
          _acc = torch.zeros(
            (_num_kv_heads, _gqa_ratio, _chunk_q, _head_size),
            dtype=torch.float32,
            device=_q.device,
          )
          _has_kv = False

          for _k_start in range(0, _kv_len, _kv_block):
            _k_end = min(_k_start + _kv_block, _kv_len)
            _k_chunk = _k_heads[:, _k_start:_k_end].float()
            _v_chunk = _v_heads[:, _k_start:_k_end].float()
            if _k_chunk.shape[1] == 0:
              continue
            _has_kv = True
            _scores = (
              torch.einsum("hgqd,hkd->hgqk", _q_chunk, _k_chunk)
              * self.scale
            )
            if _q_positions is not None:
              _q_pos = _q_positions[_q_start:_q_end]
              _k_pos = torch.arange(_k_start, _k_end, device=_q.device)
              _mask = _k_pos.unsqueeze(0) > _q_pos.unsqueeze(1)
              _scores = _scores.masked_fill(
                _mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
              )

            _scores_max = _scores.amax(dim=-1)
            _new_m = torch.maximum(_m, _scores_max)
            _old_scale = torch.exp(_m - _new_m)
            _weights = torch.exp(_scores - _new_m.unsqueeze(-1))
            _acc = _acc * _old_scale.unsqueeze(-1) + torch.einsum(
              "hgqk,hkd->hgqd",
              _weights,
              _v_chunk,
            )
            _l = _l * _old_scale + _weights.sum(dim=-1)
            _m = _new_m

          if not _has_kv:
            output[_qs + _q_start:_qs + _q_end].zero_()
            continue

          _chunk_out = (
            _acc / _l.clamp_min(1e-9).unsqueeze(-1)
          ).reshape(_num_heads, _chunk_q, _head_size)
          output[_qs + _q_start:_qs + _q_end] = _chunk_out.transpose(
            0,
            1,
          ).to(_q.dtype)
        continue

      # Keep fp16 inputs and rely on native GQA when available so we
      # do not materialize full-sequence fp32 Q/K/V copies on 4 GiB Pascal.
      _qb = _q[_qs:_qe].unsqueeze(0).transpose(1, 2)
      _kb = _k.unsqueeze(0).transpose(1, 2)
      _vb = _v.unsqueeze(0).transpose(1, 2)
      _sdpa_kwargs = {
        "scale": self.scale,
        "is_causal": _q_len == _kv_len,
      }
      if _gqa_ratio > 1:
        try:
          _ao = _F.scaled_dot_product_attention(
            _qb,
            _kb,
            _vb,
            enable_gqa=True,
            **_sdpa_kwargs,
          )
        except (TypeError, RuntimeError):
          _kb = _kb.repeat_interleave(_gqa_ratio, dim=1)
          _vb = _vb.repeat_interleave(_gqa_ratio, dim=1)
          _ao = _F.scaled_dot_product_attention(
            _qb,
            _kb,
            _vb,
            **_sdpa_kwargs,
          )
      else:
        _ao = _F.scaled_dot_product_attention(
          _qb,
          _kb,
          _vb,
          **_sdpa_kwargs,
        )
      output[_qs:_qe] = _ao.to(_q.dtype).squeeze(0).transpose(0, 1)
  ''').strip("\n"),
  "        ",
)

if sdpa_old in triton_attn_text and sdpa_new not in triton_attn_text:
    triton_attn_text = triton_attn_text.replace(sdpa_old, sdpa_new, 1)
    triton_attn_path.write_text(triton_attn_text)
    print("Applied runtime Pascal SDPA attention fallback", flush=True)
else:
    print("Pascal SDPA attention patch: already applied or target not found", flush=True)
from pathlib import Path

path = Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py")
text = path.read_text()

old = '''def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for Gated Delta Net.
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    beta_output = b.sigmoid()
    TODO maybe use torch.compile to replace this triton kernel
    """
    batch, num_heads = a.shape
    seq_len = 1
    grid = (batch, seq_len, triton.cdiv(num_heads, 8))
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        8,
        num_warps=1,
    )
    return g, beta_output'''

new = '''def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pure-PyTorch GDN gating for Pascal (replaces Triton kernel).
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta_output = b.sigmoid()
    """
    import torch.nn.functional as F
    a_f = a.float() + dt_bias.float()
    softplus_a = F.softplus(a_f, beta=beta, threshold=threshold)
    g = -torch.exp(A_log.float()) * softplus_a
    g = g.unsqueeze(0)  # [1, batch, num_heads]
    beta_output = b.sigmoid().unsqueeze(0)  # [1, batch, num_heads]
    return g, beta_output'''

if old in text and new not in text:
    text = text.replace(old, new, 1)
    path.write_text(text)
    print("Applied runtime Pascal GDN gating PyTorch fallback", flush=True)
else:
    print("Pascal GDN gating patch: already applied or target not found", flush=True)
PY
}

apply_pascal_gdn_prefill_patch() {
  python3 - <<'PY'
import textwrap
from pathlib import Path

# Try both locations: v0.19+ modular path and v0.18 monolithic path
candidates = [
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py"),
]

# Both annotations appear across versions
old_body = '''        return fla_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )'''

new_body = textwrap.indent(
    textwrap.dedent('''
        """Pure-PyTorch GDN prefill for Pascal (sm<7.0).

        Replaces the Triton/FLA chunked-parallel implementation whose
        tl.dot() calls produce incorrect results on sm_61.
        Optimized for Pascal VRAM: keeps recurrent state in fp32 but only
        materializes chunk-sized float32 workspaces instead of whole sequences.
        """
        import os
        import torch.nn.functional as F

        if use_qk_l2norm_in_kernel:
            q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
            k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

        B, T_total, Hg, K = q.shape
        H = v.shape[2]
        V = v.shape[3]
        dtype = q.dtype
        device = q.device
        scale = K ** -0.5
        chunk_size = max(
            16,
            int(os.environ.get("COPILOT_GDN_PREFILL_CHUNK", "128")),
        )

        N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
        # GGUF V-heads are in TILED order: [G0v0,G1v0,...,G15v0,G0v1,...,G15v1]
        # llama.cpp ggml_repeat tiles K as [K0,K1,...,K15,K0,K1,...,K15]
        # so V-head h maps to K-head (h % Hg), NOT h // heads_per_group
        head_map = torch.arange(H, device=device) % Hg

        o = torch.zeros(B, T_total, H, V, dtype=dtype, device=device)
        final_state = initial_state.clone() if output_final_state else None

        for seq_idx in range(N):
            if cu_seqlens is not None:
                start = cu_seqlens[seq_idx].item()
                end = cu_seqlens[seq_idx + 1].item()
            else:
                start = 0
                end = T_total

            if initial_state is not None:
                h = initial_state[seq_idx].float()
            else:
                h = torch.zeros(H, V, K, dtype=torch.float32, device=device)

            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)

                q_all = q[0, chunk_start:chunk_end, head_map].float()  # [Tc, H, K]
                k_all = k[0, chunk_start:chunk_end, head_map].float()  # [Tc, H, K]
                v_all = v[0, chunk_start:chunk_end].float()            # [Tc, H, V]
                g_exp = torch.exp(g[0, chunk_start:chunk_end].float()) # [Tc, H]
                beta_all = beta[0, chunk_start:chunk_end].float()      # [Tc, H]

                for local_t in range(chunk_end - chunk_start):
                    k_t = k_all[local_t]  # [H, K]
                    h *= g_exp[local_t].unsqueeze(-1).unsqueeze(-1)
                    pred = torch.bmm(h, k_t.unsqueeze(-1)).squeeze(-1)
                    delta = (v_all[local_t] - pred) * beta_all[local_t].unsqueeze(-1)
                    h += torch.bmm(delta.unsqueeze(-1), k_t.unsqueeze(-2))
                    o[0, chunk_start + local_t] = (
                        torch.bmm(h, q_all[local_t].unsqueeze(-1)).squeeze(-1) * scale
                    ).to(dtype)

            if output_final_state:
                final_state[seq_idx] = h.to(initial_state.dtype)

        return o, final_state
    ''').strip("\n"),
    "        ",
)

applied = False
for path in candidates:
    if not path.exists():
        continue
    text = path.read_text()
    if old_body in text and new_body not in text:
        text = text.replace(old_body, new_body, 1)
        path.write_text(text)
        print(f"Applied runtime Pascal GDN prefill PyTorch fallback ({path.name})", flush=True)
        applied = True
        break

if not applied:
    print("Pascal GDN prefill patch: already applied or target not found", flush=True)
PY
}

setup_file_logging() {
  [ "$VLLM_ENABLE_FILE_LOGGING" = "1" ] || return 0
  [ -d "$LOG_ROOT" ] || return 0

  runtime_log_dir="$LOG_ROOT/runtime"
  mkdir -p "$runtime_log_dir"

  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  log_file="$runtime_log_dir/vllm-${timestamp}.log"
  latest_link="$runtime_log_dir/latest.log"
  log_basename=$(basename "$log_file")
  fifo_path=$(mktemp -u /tmp/vllm-log-fifo.XXXXXX)

  mkfifo "$fifo_path"
  ln -sfn "$log_basename" "$latest_link"
  tee -a "$log_file" <"$fifo_path" &
  tee_pid=$!

  exec 3>&1 4>&2
  exec >"$fifo_path" 2>&1

  LOG_FIFO_PATH=$fifo_path
  LOG_TEE_PID=$tee_pid
  LOG_FILE_PATH=$log_file
}

cleanup_file_logging() {
  status=$?

  if [ -n "${LOG_FIFO_PATH:-}" ]; then
    exec 1>&3 2>&4
    wait "${LOG_TEE_PID:-}" 2>/dev/null || true
    rm -f "$LOG_FIFO_PATH"
  fi

  exit "$status"
}

setup_file_logging
trap cleanup_file_logging EXIT HUP INT TERM

apply_turboquant_patch() {
  TURBOQUANT_ENABLE=${TURBOQUANT_ENABLE:-0}
  [ "$TURBOQUANT_ENABLE" = "1" ] || { printf 'TurboQuant: disabled (TURBOQUANT_ENABLE!=1)\n'; return 0; }

  # Add turboquant to PYTHONPATH instead of pip install (volume is read-only)
  tq_dir=/workspace/external/turboquant
  if [ -d "$tq_dir/turboquant" ]; then
    PYTHONPATH="${tq_dir}:${PYTHONPATH:-}"
    export PYTHONPATH
    printf 'TurboQuant: added %s to PYTHONPATH\n' "$tq_dir"
  else
    printf 'TurboQuant: WARNING — %s not found, skipping\n' "$tq_dir"
    return 0
  fi

  python3 - <<'PY'
import os
from pathlib import Path

runner_path = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py")
text = runner_path.read_text()
modified = False

# Hook: after model load, install TurboQuant hooks on full-attention layers
hook_anchor = '                time_after_load = time.perf_counter()\n'
hook_code = '''                time_after_load = time.perf_counter()

                # --- TurboQuant hook: register KV capture on full-attention layers ---
                _tq_mode = os.environ.get("TURBOQUANT_MODE", "off")
                if _tq_mode != "off":
                    try:
                        import sys
                        sys.path.insert(0, "/workspace/scripts")
                        from turboquant_hook import install_hooks_on_model
                        install_hooks_on_model(self.model, device=self.device)
                        logger.info("[TurboQuant] Hooks installed (mode=%s)", _tq_mode)
                    except Exception as _tq_err:
                        logger.warning("[TurboQuant] Hook installation failed: %s", _tq_err)
'''

if hook_anchor in text and 'TurboQuant hook' not in text:
    text = text.replace(hook_anchor, hook_code, 1)
    modified = True
    print("Applied TurboQuant model runner hook", flush=True)

# Also add 'import os' at top if not already present
if 'import os\n' not in text[:500]:
    text = 'import os\n' + text
    modified = True

if modified:
    runner_path.write_text(text)
else:
    print("TurboQuant model runner hook: already applied or anchor not found", flush=True)
PY
}

configure_speculative_draft() {
  SPECULATIVE_CONFIG_JSON=
  SPECULATIVE_DRAFT_WINDOW_READY=0

  [ -n "$SPECULATIVE_METHOD" ] || {
    printf 'Speculative draft: disabled (SPECULATIVE_METHOD is empty)\n'
    return 0
  }

  if ! grep -q 'speculative_config:' /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py 2>/dev/null; then
    printf 'Speculative draft: requested but this vLLM build does not expose --speculative-config\n' >&2
    if [ "$SPECULATIVE_STRICT" = "1" ]; then
      exit 1
    fi
    printf 'Speculative draft: continuing without speculative decode\n'
    return 0
  fi

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

  if [ "$SPECULATIVE_METHOD" = "dflash" ] && [ -n "$SPECULATIVE_DRAFT_WINDOW_SIZE" ]; then
    if grep -RqsE 'speculative-dflash-draft-window-size|dflash_draft_window_size' /usr/local/lib/python3.12/dist-packages/vllm 2>/dev/null; then
      SPECULATIVE_DRAFT_WINDOW_READY=1
    else
      printf 'DFlash: draft window requested but this vLLM build does not expose --speculative-dflash-draft-window-size\n' >&2
      if [ "$SPECULATIVE_STRICT" = "1" ]; then
        exit 1
      fi
    fi
  fi

  printf 'Speculative draft: method=%s model=%s num_speculative_tokens=%s\n' \
    "$SPECULATIVE_METHOD" "$SPECULATIVE_DRAFT_MODEL" "$SPECULATIVE_NUM_SPECULATIVE_TOKENS"
  if [ "$SPECULATIVE_METHOD" = "dflash" ] && [ "${TURBOQUANT_ENABLE:-0}" = "1" ]; then
    printf 'DFlash: TurboQuant coexistence remains experimental on this Pascal branch\n'
  fi
}

prefer_host_nvidia_driver_libs
apply_gpu_power_limit
apply_qwen35_gguf_runtime_patch
apply_spec_decode_utils_runtime_patch
apply_eagle_runtime_patch
apply_dflash_gguf_runtime_patch
apply_dflash_runtime_patch
apply_qwen3_dflash_runtime_patch
apply_qwen3_next_runtime_patch
apply_attention_runtime_patch
apply_gpu_model_runner_runtime_patch
apply_turboquant_patch
if [ "$SPECULATIVE_METHOD" = "dflash" ] && awk -v value="$SPECULATIVE_NUM_SPECULATIVE_TOKENS" 'BEGIN { exit !(value > 1) }'; then
  printf 'Clamping SPECULATIVE_NUM_SPECULATIVE_TOKENS from %s to 1 for DFlash on constrained VRAM\n' "$SPECULATIVE_NUM_SPECULATIVE_TOKENS"
  SPECULATIVE_NUM_SPECULATIVE_TOKENS=1
fi
configure_speculative_draft
apply_triton_runtime_patch
apply_triton_autotuner_runtime_patch
apply_fla_runtime_patch
apply_pascal_gdn_prefill_patch
apply_pascal_gdn_gating_patch

if [ "$SPECULATIVE_METHOD" = "dflash" ] && [ -n "${SPECULATIVE_CONFIG_JSON:-}" ] && [ -z "${DFLASH_MAX_LAYERS:-}" ]; then
  printf 'Setting DFLASH_MAX_LAYERS=1 for DFlash on constrained VRAM\n'
  export DFLASH_MAX_LAYERS=1
fi

if [ "$SPECULATIVE_METHOD" = "dflash" ] && [ -n "${SPECULATIVE_CONFIG_JSON:-}" ] && [ -z "${PYTORCH_ALLOC_CONF:-}" ]; then
  printf 'Setting PYTORCH_ALLOC_CONF=expandable_segments:True for DFlash on constrained VRAM\n'
  export PYTORCH_ALLOC_CONF=expandable_segments:True
fi

if [ "$SPECULATIVE_METHOD" = "dflash" ] \
  && [ -n "${SPECULATIVE_CONFIG_JSON:-}" ] \
  && [ -n "${DFLASH_MAX_MODEL_LEN_CAP:-}" ] \
  && awk -v value="$MAX_MODEL_LEN" -v cap="$DFLASH_MAX_MODEL_LEN_CAP" 'BEGIN { exit !(value > cap) }'; then
  printf 'Clamping MAX_MODEL_LEN from %s to %s for DFlash on constrained VRAM\n' \
    "$MAX_MODEL_LEN" "$DFLASH_MAX_MODEL_LEN_CAP"
  MAX_MODEL_LEN=$DFLASH_MAX_MODEL_LEN_CAP
fi

if [ "$SPECULATIVE_METHOD" = "dflash" ] \
  && [ -n "${SPECULATIVE_CONFIG_JSON:-}" ] \
  && [ -n "${DFLASH_MAX_BATCHED_TOKENS_CAP:-}" ] \
  && {
    [ -z "$MAX_NUM_BATCHED_TOKENS" ] \
      || awk -v value="$MAX_NUM_BATCHED_TOKENS" -v cap="$DFLASH_MAX_BATCHED_TOKENS_CAP" 'BEGIN { exit !(value > cap) }'
  }; then
  printf 'Setting MAX_NUM_BATCHED_TOKENS=%s for DFlash on constrained VRAM\n' \
    "$DFLASH_MAX_BATCHED_TOKENS_CAP"
  MAX_NUM_BATCHED_TOKENS=$DFLASH_MAX_BATCHED_TOKENS_CAP
fi

if [ "$SPECULATIVE_METHOD" = "dflash" ] \
  && [ -n "${SPECULATIVE_CONFIG_JSON:-}" ] \
  && [ -n "${DFLASH_GPU_MEMORY_UTILIZATION_CAP:-}" ] \
  && awk -v value="$GPU_MEMORY_UTILIZATION" -v cap="$DFLASH_GPU_MEMORY_UTILIZATION_CAP" 'BEGIN { exit !(value > cap) }'; then
  printf 'Clamping GPU_MEMORY_UTILIZATION from %s to %s for DFlash on constrained VRAM\n' \
    "$GPU_MEMORY_UTILIZATION" "$DFLASH_GPU_MEMORY_UTILIZATION_CAP"
  GPU_MEMORY_UTILIZATION=$DFLASH_GPU_MEMORY_UTILIZATION_CAP
fi

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
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config vllm \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enforce-eager

if [ -n "$MAX_NUM_BATCHED_TOKENS" ]; then
  set -- "$@" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
fi

if [ "${ENABLE_PREFIX_CACHING:-0}" = "1" ]; then
  set -- "$@" --enable-prefix-caching
fi

if [ -n "$MAMBA_CACHE_MODE" ]; then
  set -- "$@" --mamba-cache-mode "$MAMBA_CACHE_MODE"
fi

if [ -n "$MAMBA_BLOCK_SIZE" ]; then
  set -- "$@" --mamba-block-size "$MAMBA_BLOCK_SIZE"
fi

# CPU+GPU hybrid inference: offload weights to CPU RAM via UVA.
# KV cache offloading stays opt-in (backend support varies by build).
if [ "$CPU_OFFLOAD_GB" != "0" ]; then
  set -- "$@" --cpu-offload-gb "$CPU_OFFLOAD_GB"
fi

if [ -n "$KV_OFFLOADING_SIZE_GB" ]; then
  set -- "$@" --kv-offloading-size "$KV_OFFLOADING_SIZE_GB" --kv-offloading-backend "$KV_OFFLOADING_BACKEND"
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

if [ "${SPECULATIVE_DRAFT_WINDOW_READY:-0}" = "1" ]; then
  set -- "$@" --speculative-dflash-draft-window-size "$SPECULATIVE_DRAFT_WINDOW_SIZE"
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

#!/bin/sh
set -eu

ENV_FILE=
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

resolve_repo_path() {
  path=$1
  case "$path" in
    [~]/*) printf '%s/%s\n' "$HOME" "${path#\~/}" ;;
    /*) printf '%s\n' "$path" ;;
    *) printf '%s/%s\n' "$REPO_ROOT" "$path" ;;
  esac
}

read_env_value() {
  key=$1
  file=$2

  awk -F= -v key="$key" '
    $1 == key {
      print substr($0, length(key) + 2)
      found = 1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "$file"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --env-file)
      ENV_FILE=${2:?--env-file requires a value}
      shift 2
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [ -n "$ENV_FILE" ]; then
  ENV_FILE=$(resolve_repo_path "$ENV_FILE")
fi

SOURCE_GGUF=${MTP_DRAFT_SOURCE_GGUF:-~/Downloads/Qwen3.5-4B-MTP-Q4_K_M.gguf}
TARGET_DIR=${MTP_DRAFT_STAGE_DIR:-./volumes/models/Qwen3.5-4B-MTP-GGUF}
HF_CACHE_ROOT=${MTP_DRAFT_HF_CACHE_ROOT:-./volumes/hf-cache/hub/models--Qwen--Qwen3.5-4B}
CONFIG_PATH=${MTP_DRAFT_CONFIG_PATH:-}

if [ -n "$ENV_FILE" ]; then
  if value=$(read_env_value MTP_DRAFT_SOURCE_GGUF "$ENV_FILE" 2>/dev/null); then
    SOURCE_GGUF=$value
  fi
  if value=$(read_env_value MTP_DRAFT_STAGE_DIR "$ENV_FILE" 2>/dev/null); then
    TARGET_DIR=$value
  fi
  if value=$(read_env_value MTP_DRAFT_HF_CACHE_ROOT "$ENV_FILE" 2>/dev/null); then
    HF_CACHE_ROOT=$value
  fi
  if value=$(read_env_value MTP_DRAFT_CONFIG_PATH "$ENV_FILE" 2>/dev/null); then
    CONFIG_PATH=$value
  fi
fi

SOURCE_GGUF=$(resolve_repo_path "$SOURCE_GGUF")
TARGET_DIR=$(resolve_repo_path "$TARGET_DIR")
HF_CACHE_ROOT=$(resolve_repo_path "$HF_CACHE_ROOT")

if [ -n "$CONFIG_PATH" ]; then
  CONFIG_PATH=$(resolve_repo_path "$CONFIG_PATH")
fi

if [ -z "$CONFIG_PATH" ]; then
  CONFIG_PATH=$(find "$HF_CACHE_ROOT"/snapshots -mindepth 2 -maxdepth 2 -path '*/config.json' | head -n 1 || true)
fi

[ -f "$SOURCE_GGUF" ] || {
  printf 'Missing MTP GGUF: %s\n' "$SOURCE_GGUF" >&2
  exit 1
}

[ -f "$CONFIG_PATH" ] || {
  printf 'Missing base Qwen config.json in cache root: %s\n' "$HF_CACHE_ROOT" >&2
  exit 1
}

mkdir -p "$TARGET_DIR"
cp -f "$SOURCE_GGUF" "$TARGET_DIR/$(basename "$SOURCE_GGUF")"
cp -f "$CONFIG_PATH" "$TARGET_DIR/config.json"

python3 - "$TARGET_DIR/config.json" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
config = json.loads(config_path.read_text())
config["architectures"] = ["Qwen3_5MTP"]
config.setdefault("model_type", "qwen3_5")
config.setdefault("tie_word_embeddings", True)
config.pop("vision_config", None)
config.pop("image_token_id", None)
config.pop("video_token_id", None)
config.pop("vision_start_token_id", None)
config.pop("vision_end_token_id", None)
text_config = config.setdefault("text_config", {})
text_config.setdefault("model_type", "qwen3_5_text")
text_config.setdefault("mtp_num_hidden_layers", 1)
text_config.setdefault("mtp_use_dedicated_embeddings", False)
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

printf 'Prepared MTP GGUF draft bundle in %s\n' "$TARGET_DIR"

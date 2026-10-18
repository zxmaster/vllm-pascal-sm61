#!/bin/sh
set -eu

ENV_FILE=
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

resolve_repo_path() {
  path=$1
  case "$path" in
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

SOURCE_GGUF=${DFLASH_DRAFT_SOURCE_GGUF:-./artifacts/dflash/Qwen3.5-4B-DFlash-Q4_K_M.gguf}
TARGET_DIR=${DFLASH_DRAFT_STAGE_DIR:-./volumes/models/Qwen3.5-4B-DFlash-GGUF}
HF_CACHE_ROOT=${DFLASH_DRAFT_HF_CACHE_ROOT:-./volumes/hf-cache/hub/models--z-lab--Qwen3.5-4B-DFlash}
CONFIG_PATH=${DFLASH_DRAFT_CONFIG_PATH:-}
MODULE_PATH=${DFLASH_DRAFT_MODULE_PATH:-}

if [ -n "$ENV_FILE" ]; then
  if value=$(read_env_value DFLASH_DRAFT_SOURCE_GGUF "$ENV_FILE" 2>/dev/null); then
    SOURCE_GGUF=$value
  fi
  if value=$(read_env_value DFLASH_DRAFT_STAGE_DIR "$ENV_FILE" 2>/dev/null); then
    TARGET_DIR=$value
  fi
  if value=$(read_env_value DFLASH_DRAFT_HF_CACHE_ROOT "$ENV_FILE" 2>/dev/null); then
    HF_CACHE_ROOT=$value
  fi
  if value=$(read_env_value DFLASH_DRAFT_CONFIG_PATH "$ENV_FILE" 2>/dev/null); then
    CONFIG_PATH=$value
  fi
  if value=$(read_env_value DFLASH_DRAFT_MODULE_PATH "$ENV_FILE" 2>/dev/null); then
    MODULE_PATH=$value
  fi
fi

SOURCE_GGUF=$(resolve_repo_path "$SOURCE_GGUF")
TARGET_DIR=$(resolve_repo_path "$TARGET_DIR")
HF_CACHE_ROOT=$(resolve_repo_path "$HF_CACHE_ROOT")

if [ -n "$CONFIG_PATH" ]; then
  CONFIG_PATH=$(resolve_repo_path "$CONFIG_PATH")
fi

if [ -n "$MODULE_PATH" ]; then
  MODULE_PATH=$(resolve_repo_path "$MODULE_PATH")
fi

if [ -z "$CONFIG_PATH" ]; then
  CONFIG_PATH=$(find "$HF_CACHE_ROOT"/snapshots -mindepth 2 -maxdepth 2 -path '*/config.json' | head -n 1 || true)
fi

if [ -z "$MODULE_PATH" ]; then
  MODULE_PATH=$(find "$HF_CACHE_ROOT"/snapshots -mindepth 2 -maxdepth 2 -path '*/dflash.py' | head -n 1 || true)
fi

[ -f "$SOURCE_GGUF" ] || {
  printf 'Missing DFlash GGUF: %s\n' "$SOURCE_GGUF" >&2
  printf 'Expected a vendored project copy; set DFLASH_DRAFT_SOURCE_GGUF if using a different path.\n' >&2
  exit 1
}

[ -f "$CONFIG_PATH" ] || {
  printf 'Missing DFlash config.json in cache root: %s\n' "$HF_CACHE_ROOT" >&2
  exit 1
}

mkdir -p "$TARGET_DIR"
cp -f "$SOURCE_GGUF" "$TARGET_DIR/$(basename "$SOURCE_GGUF")"
cp -f "$CONFIG_PATH" "$TARGET_DIR/config.json"

if [ -n "$MODULE_PATH" ] && [ -f "$MODULE_PATH" ]; then
  cp -f "$MODULE_PATH" "$TARGET_DIR/dflash.py"
fi

python3 - "$TARGET_DIR/config.json" <<'PY'
import json
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
config = json.loads(config_path.read_text())
config["architectures"] = ["DFlashDraftModel"]
config.setdefault("model_type", "qwen3")
config.setdefault("tie_word_embeddings", True)
config.setdefault("auto_map", {"AutoModel": "dflash.DFlashDraftModel"})
config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY

printf 'Prepared DFlash GGUF draft bundle in %s\n' "$TARGET_DIR"

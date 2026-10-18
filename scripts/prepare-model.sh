#!/bin/sh
# prepare-model.sh — download the reference model (Qwen3.5-4B IQ4_XS) and its tokenizer.
#
# The compose stacks mount:
#   volumes/models     -> /models        GGUF weights
#   volumes/hf-cache   -> huggingface cache (tokenizer, resolved with HF_HUB_OFFLINE=1)
#   volumes/hf-configs -> /hf-configs    config.json used as --hf-config-path
#
# Nothing here is redistributed by this repository: the weights come from their
# publisher under their own license, the tokenizer/config from the upstream HF model.
#
# Usage:
#   ./scripts/prepare-model.sh              # download + verify
#   ./scripts/prepare-model.sh --no-verify  # skip the sha256 check of the GGUF
#   ./scripts/prepare-model.sh --check      # report what is present, download nothing
set -eu

GGUF_REPO=${GGUF_REPO:-unsloth/Qwen3.5-4B-GGUF}
GGUF_FILE=${GGUF_FILE:-Qwen3.5-4B-IQ4_XS.gguf}
GGUF_SIZE=${GGUF_SIZE:-2477053088}
GGUF_SHA256=${GGUF_SHA256:-658a9e7e406deb06d0179755e3c14f6a82915a4be4962a2f92a64d948d2e572f}
TOK_REPO=${TOK_REPO:-Qwen/Qwen3.5-4B}
TOK_FILES="config.json tokenizer.json tokenizer_config.json vocab.json merges.txt preprocessor_config.json video_preprocessor_config.json"

VERIFY=1
CHECK_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --no-verify) VERIFY=0 ;;
    --check) CHECK_ONLY=1 ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

MODEL_DIR=volumes/models/Qwen3.5-4B-IQ4_XS-GGUF
GGUF_PATH=$MODEL_DIR/$GGUF_FILE
CONFIG_DIR=volumes/hf-configs/Qwen3.5-4B-text
export HF_HOME=${HF_HOME:-$PWD/volumes/hf-cache}

mkdir -p "$MODEL_DIR" "$CONFIG_DIR" "$HF_HOME"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

hf_ok=0
command -v hf >/dev/null 2>&1 && hf_ok=1

# --- 1. GGUF weights -------------------------------------------------------
if [ -f "$GGUF_PATH" ] && [ "$(wc -c <"$GGUF_PATH")" = "$GGUF_SIZE" ]; then
  echo "== weights already present: $GGUF_PATH"
else
  if [ "$CHECK_ONLY" = "1" ]; then
    echo "!! weights missing: $GGUF_PATH"
  else
    echo "== downloading $GGUF_FILE (~2.3 GiB) from $GGUF_REPO"
    if [ "$hf_ok" = "1" ]; then
      hf download "$GGUF_REPO" "$GGUF_FILE" --local-dir "$MODEL_DIR" || exit 1
    else
      curl -fL --retry 5 --retry-delay 5 --retry-all-errors --connect-timeout 20 --progress-bar \
        -o "$GGUF_PATH.part" "https://huggingface.co/$GGUF_REPO/resolve/main/$GGUF_FILE" || exit 1
      mv "$GGUF_PATH.part" "$GGUF_PATH"
    fi
    size=$(wc -c <"$GGUF_PATH" 2>/dev/null || echo 0)
    if [ "$size" != "$GGUF_SIZE" ]; then
      echo "!! unexpected size: $size bytes (expected $GGUF_SIZE)" >&2
      exit 1
    fi
    echo "== size ok: $GGUF_SIZE bytes"
  fi
fi

if [ -f "$GGUF_PATH" ] && [ "$VERIFY" = "1" ]; then
  echo "== verifying sha256 (takes a few seconds for 2.3 GiB)"
  actual=$(sha256_of "$GGUF_PATH")
  if [ "$actual" != "$GGUF_SHA256" ]; then
    echo "!! sha256 mismatch:" >&2
    echo "   expected $GGUF_SHA256" >&2
    echo "   actual   $actual" >&2
    exit 1
  fi
  echo "== sha256 ok"
fi

# --- 2. tokenizer + config -------------------------------------------------
if [ "$CHECK_ONLY" = "0" ]; then
  if [ "$hf_ok" = "1" ]; then
    echo "== downloading tokenizer from $TOK_REPO into $HF_HOME"
    include_args=""
    for f in $TOK_FILES; do include_args="$include_args --include $f"; done
    # shellcheck disable=SC2086
    hf download "$TOK_REPO" $include_args >/dev/null || {
      echo "!! tokenizer download failed" >&2; exit 1; }
  else
    echo "== hf CLI not found — fetching tokenizer files with curl into $CONFIG_DIR"
    for f in $TOK_FILES; do
      [ -f "$CONFIG_DIR/$f" ] && continue
      curl -fL --retry 3 --retry-delay 3 --connect-timeout 20 -o "$CONFIG_DIR/$f" \
        "https://huggingface.co/$TOK_REPO/resolve/main/$f" || true
    done
  fi
fi

# config.json is what --hf-config-path points at
if [ ! -f "$CONFIG_DIR/config.json" ] && [ "$CHECK_ONLY" = "0" ]; then
  snap_config=$(find "$HF_HOME/hub/models--$(printf '%s' "$TOK_REPO" | sed 's|/|--|g')/snapshots" -name config.json 2>/dev/null | head -1 || true)
  if [ -n "$snap_config" ]; then
    cp "$snap_config" "$CONFIG_DIR/config.json"
  fi
  [ -f "$CONFIG_DIR/config.json" ] || { echo "!! could not place $CONFIG_DIR/config.json" >&2; exit 1; }
fi

# --- 3. report -------------------------------------------------------------
echo
echo "== state"
if [ -f "$GGUF_PATH" ]; then
  printf '   weights   %s (%s bytes)\n' "$GGUF_PATH" "$(wc -c <"$GGUF_PATH")"
else
  printf '   weights   MISSING (%s)\n' "$GGUF_PATH"
fi
tok_json=$(find "$HF_HOME/hub" -path "*models--$(printf '%s' "$TOK_REPO" | sed 's|/|--|g')*" -name tokenizer.json 2>/dev/null | head -1 || true)
if [ -n "$tok_json" ]; then
  printf '   tokenizer %s\n' "$tok_json"
elif [ -f "$CONFIG_DIR/tokenizer.json" ]; then
  printf '   tokenizer %s (local dir; set TOKENIZER_MODEL=/hf-configs/Qwen3.5-4B-text)\n' "$CONFIG_DIR/tokenizer.json"
else
  printf '   tokenizer MISSING\n'
fi
if [ -f "$CONFIG_DIR/config.json" ]; then
  printf '   config    %s\n' "$CONFIG_DIR/config.json"
else
  printf '   config    MISSING (%s)\n' "$CONFIG_DIR/config.json"
fi
echo
echo "   next: SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-source-cache.sh"
echo "         SM61_ENV_FILE=.env.v031 ./scripts/build-sm61-image.sh"
echo "         VLLM_ENV_FILE=.env.v031 docker compose -f compose.sm61-v031.yaml up -d"

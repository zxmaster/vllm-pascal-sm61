#!/bin/sh
set -eu

container_name=${VLLM_CONTAINER_NAME:-vllm-openai-v029}
state_dir=${OPENCODE_VLLM_STATE_DIR:-$HOME/.config/opencode}
base_url_file=${OPENCODE_VLLM_BASE_URL_FILE:-$state_dir/vllm-local-base-url.txt}
base_url=${OPENCODE_VLLM_BASE_URL:-}

case "$base_url" in
  http://*|https://*) ;;
  '') ;;
  *)
    echo "Ignoring invalid OPENCODE_VLLM_BASE_URL '$base_url'; resolving container IP instead." >&2
    base_url=
    ;;
esac

if [ -z "$base_url" ]; then
  container_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container_name" 2>/dev/null || true)
  if [ -z "$container_ip" ]; then
    echo "Could not resolve IP for container '$container_name'. Start the v0.28 vLLM stack first (or set VLLM_CONTAINER_NAME)." >&2
    exit 1
  fi
  base_url="http://$container_ip:8000/v1"
fi

mkdir -p "$state_dir"
printf '%s\n' "$base_url" > "$base_url_file"

export OPENCODE_VLLM_BASE_URL="$base_url"
export OPENCODE_VLLM_BASE_URL_FILE="$base_url_file"
exec opencode "$@"

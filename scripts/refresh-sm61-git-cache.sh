#!/bin/sh
set -eu

CACHE_ROOT=${CACHE_ROOT:-docker/sm61/git-cache/repos}
STATE_ROOT=$(dirname "$CACHE_ROOT")
TORCH_URL=${TORCH_URL:-https://github.com/pytorch/pytorch.git}
TORCH_REF=${TORCH_REF:-${SM61_TORCH_REF:-v2.10.0}}
VLLM_URL=${VLLM_URL:-https://github.com/vllm-project/vllm.git}
VLLM_REF=${VLLM_REF:-${SM61_VLLM_REF:-v0.18.0}}
GIT_SUBMODULE_JOBS=${GIT_SUBMODULE_JOBS:-${SM61_GIT_SUBMODULE_JOBS:-8}}

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT

ensure_mirror() {
  url=$1
  rel=${url#https://}
  rel=${rel#http://}
  case "$rel" in
    github.com/*) ;;
    *)
      printf 'Skipping unsupported remote: %s\n' "$url" >&2
      return 0
      ;;
  esac
  case "$rel" in
    *.git) ;;
    *) rel="$rel.git" ;;
  esac

  mirror_path="$CACHE_ROOT/$rel"
  mkdir -p "$(dirname "$mirror_path")"
  if [ -d "$mirror_path" ]; then
    printf 'Updating mirror %s\n' "$url"
    git -C "$mirror_path" remote set-url origin "$url"
    git -C "$mirror_path" fetch --prune --tags origin
  else
    printf 'Creating mirror %s\n' "$url"
    git clone --mirror "$url" "$mirror_path"
  fi
}

collect_urls() {
  name=$1
  url=$2
  ref=$3
  repo_dir="$workdir/$name"

  printf 'Scanning %s at %s\n' "$url" "$ref" >&2
  git clone --depth 1 --filter=blob:none --branch "$ref" "$url" "$repo_dir" >/dev/null 2>&1
  git -C "$repo_dir" submodule update --init --recursive --depth 1 --jobs "$GIT_SUBMODULE_JOBS" >/dev/null 2>&1

  {
    printf '%s\n' "$url"
    git -C "$repo_dir" submodule foreach --recursive 'git config --get remote.origin.url' 2>/dev/null || true
  } | sort -u
}

mkdir -p "$CACHE_ROOT"

collect_urls torch "$TORCH_URL" "$TORCH_REF" | while IFS= read -r url; do
  [ -n "$url" ] && ensure_mirror "$url"
done

collect_urls vllm "$VLLM_URL" "$VLLM_REF" | while IFS= read -r url; do
  [ -n "$url" ] && ensure_mirror "$url"
done

date -u +%Y-%m-%dT%H:%M:%SZ > "$STATE_ROOT/.ready"
printf 'Git cache ready at %s\n' "$CACHE_ROOT"

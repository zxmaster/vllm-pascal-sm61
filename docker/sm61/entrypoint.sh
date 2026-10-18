#!/bin/sh
set -eu

if [ -f /workspace/scripts/start-vllm.sh ]; then
  exec sh /workspace/scripts/start-vllm.sh "$@"
fi

exec sh /usr/local/bin/start-vllm.sh "$@"

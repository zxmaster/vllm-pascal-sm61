#!/bin/sh
set -eu

BASE_URL=${1:-${BASE_URL:-http://vllm.test}}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B-GGUF}
MODE=${2:-${MODE:-nonthinking}}
PROMPT=${3:-${PROMPT:-Reply with exactly: ready}}
ENDPOINT=${4:-${ENDPOINT:-chat}}

curl_api() {
  if [ -n "$API_KEY" ]; then
    curl -L -fsS -H "Authorization: Bearer $API_KEY" "$@"
  else
    curl -L -fsS "$@"
  fi
}

case "$ENDPOINT" in
  chat)
    API_PATH=/v1/chat/completions
    ;;
  completion)
    API_PATH=/v1/completions
    ;;
  *)
    echo "Unknown endpoint: $ENDPOINT" >&2
    echo "Valid endpoints: chat, completion" >&2
    exit 2
    ;;
esac

case "$MODE" in
  baseline)
    if [ "$ENDPOINT" = chat ]; then
      REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "max_tokens": 16,
  "temperature": 0
}
EOF
)
    else
      REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "prompt": "$PROMPT",
  "max_tokens": 16,
  "temperature": 0
}
EOF
)
    fi
    ;;
  nonthinking)
    if [ "$ENDPOINT" = completion ]; then
      echo "Mode nonthinking requires ENDPOINT=chat" >&2
      exit 2
    fi
    REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.8,
  "presence_penalty": 1.5,
  "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}
EOF
)
    ;;
  nonthinking-repetition)
    if [ "$ENDPOINT" = completion ]; then
      echo "Mode nonthinking-repetition requires ENDPOINT=chat" >&2
      exit 2
    fi
    REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "max_tokens": 64,
  "temperature": 0.7,
  "top_p": 0.8,
  "presence_penalty": 1.8,
  "repetition_penalty": 1.05,
  "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}
EOF
)
    ;;
  coding)
    if [ "$ENDPOINT" = chat ]; then
      REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "max_tokens": 192,
  "temperature": 0.6,
  "top_p": 0.95,
  "presence_penalty": 0.0,
  "top_k": 20,
  "chat_template_kwargs": {"enable_thinking": false}
}
EOF
)
    else
      REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "prompt": "$PROMPT",
  "max_tokens": 192,
  "temperature": 0.6,
  "top_p": 0.95,
  "presence_penalty": 0.0,
  "top_k": 20
}
EOF
)
    fi
    ;;
  thinking)
    if [ "$ENDPOINT" = chat ]; then
      REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "messages": [
    {"role": "user", "content": "$PROMPT"}
  ],
  "max_tokens": 128,
  "temperature": 1.0,
  "top_p": 0.95,
  "presence_penalty": 1.5,
  "top_k": 20
}
EOF
)
    else
      REQUEST=$(cat <<EOF
{
  "model": "$MODEL_NAME",
  "prompt": "$PROMPT",
  "max_tokens": 128,
  "temperature": 1.0,
  "top_p": 0.95,
  "presence_penalty": 1.5,
  "top_k": 20
}
EOF
)
    fi
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Valid modes: baseline, nonthinking, nonthinking-repetition, coding, thinking" >&2
    exit 2
    ;;
esac

curl_api "$BASE_URL$API_PATH" \
  -H 'Content-Type: application/json' \
  -d "$REQUEST" | jq '{message: (.choices[0].message // null), text: (.choices[0].text // null), finish_reason: .choices[0].finish_reason, usage: .usage}'

#!/bin/sh
set -eu

BASE_URL=${1:-${BASE_URL:-http://vllm.test}}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B-GGUF}
WAIT_ATTEMPTS=${WAIT_ATTEMPTS:-180}
WAIT_SECONDS=${WAIT_SECONDS:-2}
SMOKE_PROMPT=${SMOKE_PROMPT:-Hi}
SMOKE_MAX_TOKENS=${SMOKE_MAX_TOKENS:-8}

curl_api() {
  if [ -n "$API_KEY" ]; then
    curl -L -fsS -H "Authorization: Bearer $API_KEY" "$@"
  else
    curl -L -fsS "$@"
  fi
}

print_json() {
  if command -v jq >/dev/null 2>&1; then
    jq .
  else
    python3 -m json.tool
  fi
}

printf 'Waiting for %s/v1/models\n' "$BASE_URL"
ready=0
for _ in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl_api "$BASE_URL/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep "$WAIT_SECONDS"
done

if [ "$ready" != "1" ]; then
  printf 'Timed out waiting for %s/v1/models after %ss\n' \
    "$BASE_URL" "$((WAIT_ATTEMPTS * WAIT_SECONDS))" >&2
  exit 1
fi

curl_api "$BASE_URL/v1/models" | print_json

curl_api "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"$MODEL_NAME\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$SMOKE_PROMPT\"}
    ],
    \"max_tokens\": $SMOKE_MAX_TOKENS,
    \"temperature\": 0,
    \"chat_template_kwargs\": {\"enable_thinking\": false}
  }" | print_json

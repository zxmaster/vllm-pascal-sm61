#!/bin/sh
set -eu

BASE_URL=${1:-${BASE_URL:-http://vllm.test}}
PROMPT_FILE=${2:-${PROMPT_FILE:-}}
RUN_NAME=${3:-${RUN_NAME:-}}
API_KEY=${API_KEY:-EMPTY}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B-GGUF}
OUTPUT_DIR=${OUTPUT_DIR:-output}
ENDPOINTS=${ENDPOINTS:-chat}
CASES=${CASES:-baseline,nonthinking,nonthinking-repetition,coding}
REQUEST_FILE=${REQUEST_FILE:-}
CUSTOM_LABEL=${CUSTOM_LABEL:-custom}

DEFAULT_PROMPT='Create a single-file landing page for vLLM Pascal GPU Edition. Return only HTML. Requirements: embed CSS and JavaScript in the same file; bold retro-futuristic design inspired by NVIDIA green, terminal black, and steel gray; hero section, feature cards, benchmark-style stats, a compatibility warning for unsupported Pascal GPUs, a setup checklist, and a call-to-action button; include subtle animations; make it responsive; no external assets or libraries; make the copy confident and technical rather than marketing fluff.'

if [ -n "$PROMPT_FILE" ]; then
  PROMPT=$(cat "$PROMPT_FILE")
else
  PROMPT=$DEFAULT_PROMPT
fi

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

slugify() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-'
}

RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LABEL=${RUN_NAME:-landing-page-bench}
RUN_ID="${RUN_STAMP}-$(slugify "$RUN_LABEL")"
RUN_DIR="$OUTPUT_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

cleanup() {
  rm -f "${REQUEST_TMP:-}" "${RESPONSE_TMP:-}"
}
trap cleanup EXIT INT TERM

REQUEST_TMP=$(mktemp)
RESPONSE_TMP=$(mktemp)

build_request() {
  case "$1:$2" in
    baseline:chat)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          messages: [
            {role: "user", content: $prompt}
          ],
          max_tokens: 192,
          temperature: 0.7
        }'
      ;;
    baseline:completion)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          prompt: $prompt,
          max_tokens: 192,
          temperature: 0.7
        }'
      ;;
    nonthinking:chat)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          messages: [
            {role: "user", content: $prompt}
          ],
          max_tokens: 512,
          temperature: 0.7,
          top_p: 0.8,
          presence_penalty: 1.5,
          repetition_penalty: 1.0,
          top_k: 20,
          chat_template_kwargs: {
            enable_thinking: false
          }
        }'
      ;;
    nonthinking-repetition:chat)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          messages: [
            {role: "user", content: $prompt}
          ],
          max_tokens: 512,
          temperature: 0.7,
          top_p: 0.8,
          presence_penalty: 1.8,
          repetition_penalty: 1.05,
          top_k: 20,
          chat_template_kwargs: {
            enable_thinking: false
          }
        }'
      ;;
    coding:chat)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          messages: [
            {role: "user", content: $prompt}
          ],
          max_tokens: 1024,
          temperature: 0.6,
          top_p: 0.95,
          presence_penalty: 0.0,
          repetition_penalty: 1.0,
          top_k: 20,
          chat_template_kwargs: {
            enable_thinking: false
          }
        }'
      ;;
    coding:completion)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          prompt: $prompt,
          max_tokens: 1024,
          temperature: 0.6,
          top_p: 0.95,
          presence_penalty: 0.0,
          repetition_penalty: 1.0,
          top_k: 20
        }'
      ;;
    thinking:chat)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          messages: [
            {role: "user", content: $prompt}
          ],
          max_tokens: 1024,
          temperature: 1.0,
          top_p: 0.95,
          presence_penalty: 1.5,
          repetition_penalty: 1.0,
          top_k: 20
        }'
      ;;
    thinking:completion)
      jq -n \
        --arg model "$MODEL_NAME" \
        --arg prompt "$PROMPT" \
        '{
          model: $model,
          prompt: $prompt,
          max_tokens: 1024,
          temperature: 1.0,
          top_p: 0.95,
          presence_penalty: 1.5,
          repetition_penalty: 1.0,
          top_k: 20
        }'
      ;;
    *)
      echo "Unsupported case/endpoint combination: $1 $2" >&2
      exit 2
      ;;
  esac
}

run_request() {
  result_file=$1
  endpoint=$2
  case_name=$3

  case "$endpoint" in
    chat)
      url="$BASE_URL/v1/chat/completions"
      ;;
    completion)
      url="$BASE_URL/v1/completions"
      ;;
    *)
      echo "Unsupported endpoint: $endpoint" >&2
      exit 2
      ;;
  esac

  : > "$RESPONSE_TMP"
  if [ -n "$API_KEY" ]; then
    CURL_META=$(curl -L -sS \
      -o "$RESPONSE_TMP" \
      -w '%{http_code}\n%{time_total}' \
      -H "Authorization: Bearer $API_KEY" \
      -H 'Content-Type: application/json' \
      -d @"$REQUEST_TMP" \
      "$url")
  else
    CURL_META=$(curl -L -sS \
      -o "$RESPONSE_TMP" \
      -w '%{http_code}\n%{time_total}' \
      -H 'Content-Type: application/json' \
      -d @"$REQUEST_TMP" \
      "$url")
  fi

  HTTP_CODE=$(printf '%s\n' "$CURL_META" | sed -n '1p')
  TIME_TOTAL=$(printf '%s\n' "$CURL_META" | sed -n '2p')
  CREATED_AT=$(timestamp_utc)

  if jq -e . >/dev/null 2>&1 < "$RESPONSE_TMP"; then
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg created_at "$CREATED_AT" \
      --arg case_name "$case_name" \
      --arg endpoint "$endpoint" \
      --arg base_url "$BASE_URL" \
      --arg model_name "$MODEL_NAME" \
      --arg http_code "$HTTP_CODE" \
      --arg time_total "$TIME_TOTAL" \
      --arg prompt_file "$PROMPT_FILE" \
      --arg request_file "$REQUEST_FILE" \
      --slurpfile request_json "$REQUEST_TMP" \
      --slurpfile response_json "$RESPONSE_TMP" \
      '
      def choice0: ($response_json[0].choices[0] // {});
      def text_content: (choice0.message.content // choice0.text // "");
      def reasoning_text: (choice0.message.reasoning // choice0.reasoning // null);
      def bang_count: (text_content | explode | map(select(. == 33)) | length);
      {
        run_id: $run_id,
        created_at: $created_at,
        case: $case_name,
        endpoint: $endpoint,
        base_url: $base_url,
        model_name: $model_name,
        prompt_file: (if $prompt_file == "" then null else $prompt_file end),
        request_file: (if $request_file == "" then null else $request_file end),
        request: $request_json[0],
        response: $response_json[0],
        extracted: {
          reasoning: reasoning_text,
          content: text_content,
          content_preview: (text_content | .[:240]),
          finish_reason: (choice0.finish_reason // null),
          usage: ($response_json[0].usage // null)
        },
        diagnostics: {
          http_code: ($http_code | tonumber),
          time_total_seconds: ($time_total | tonumber),
          bang_count: bang_count,
          has_repeated_bangs: (text_content | contains("!!!!!!!!")),
          content_length: (text_content | length)
        }
      }' > "$result_file"
  else
    jq -n \
      --arg run_id "$RUN_ID" \
      --arg created_at "$CREATED_AT" \
      --arg case_name "$case_name" \
      --arg endpoint "$endpoint" \
      --arg base_url "$BASE_URL" \
      --arg model_name "$MODEL_NAME" \
      --arg http_code "$HTTP_CODE" \
      --arg time_total "$TIME_TOTAL" \
      --arg prompt_file "$PROMPT_FILE" \
      --arg request_file "$REQUEST_FILE" \
      --slurpfile request_json "$REQUEST_TMP" \
      --rawfile response_raw "$RESPONSE_TMP" \
      '{
        run_id: $run_id,
        created_at: $created_at,
        case: $case_name,
        endpoint: $endpoint,
        base_url: $base_url,
        model_name: $model_name,
        prompt_file: (if $prompt_file == "" then null else $prompt_file end),
        request_file: (if $request_file == "" then null else $request_file end),
        request: $request_json[0],
        response: null,
        response_raw: $response_raw,
        diagnostics: {
          http_code: ($http_code | tonumber),
          time_total_seconds: ($time_total | tonumber),
          parse_error: true
        }
      }' > "$result_file"
  fi
}

INDEX=1

if [ -n "$REQUEST_FILE" ]; then
  cp "$REQUEST_FILE" "$REQUEST_TMP"
  RESULT_FILE=$(printf '%s/%02d-%s-chat.json' "$RUN_DIR" "$INDEX" "$CUSTOM_LABEL")
  run_request "$RESULT_FILE" chat "$CUSTOM_LABEL"
else
  for endpoint in $(printf '%s' "$ENDPOINTS" | tr ',' ' '); do
    for case_name in $(printf '%s' "$CASES" | tr ',' ' '); do
      build_request "$case_name" "$endpoint" > "$REQUEST_TMP"
      RESULT_FILE=$(printf '%s/%02d-%s-%s.json' "$RUN_DIR" "$INDEX" "$case_name" "$endpoint")
      run_request "$RESULT_FILE" "$endpoint" "$case_name"
      INDEX=$((INDEX + 1))
    done
  done
fi

jq -s \
  --arg run_id "$RUN_ID" \
  --arg created_at "$(timestamp_utc)" \
  '{
    run_id: $run_id,
    created_at: $created_at,
    results: [
      .[] | {
        case,
        endpoint,
        http_code: .diagnostics.http_code,
        time_total_seconds: .diagnostics.time_total_seconds,
        finish_reason: (.extracted.finish_reason // null),
        bang_count: (.diagnostics.bang_count // null),
        has_repeated_bangs: (.diagnostics.has_repeated_bangs // false),
        content_preview: (.extracted.content_preview // null)
      }
    ]
  }' "$RUN_DIR"/*.json > "$RUN_DIR/summary.json"

printf 'Wrote benchmark artifacts to %s\n' "$RUN_DIR"
printf 'Summary: %s\n' "$RUN_DIR/summary.json"

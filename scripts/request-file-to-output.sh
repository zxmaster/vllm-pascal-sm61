#!/bin/sh
set -eu

REQUEST_FILE=${1:-}
RUN_NAME=${2:-file-request}
CUSTOM_LABEL=${3:-custom}
BASE_URL=${BASE_URL:-http://vllm.test}
API_KEY=${API_KEY:-EMPTY}
ENDPOINT=${ENDPOINT:-}
OUTPUT_DIR=${OUTPUT_DIR:-output}

if [ -z "$REQUEST_FILE" ]; then
  echo "Usage: sh ./scripts/request-file-to-output.sh <request-file> [run-name] [custom-label]" >&2
  exit 2
fi

if [ -n "$ENDPOINT" ]; then
  python3 ./scripts/run_custom_request.py \
    --base-url "$BASE_URL" \
    --endpoint "$ENDPOINT" \
    --request-file "$REQUEST_FILE" \
    --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --custom-label "$CUSTOM_LABEL" \
    --api-key "$API_KEY"
else
  python3 ./scripts/run_custom_request.py \
    --base-url "$BASE_URL" \
    --request-file "$REQUEST_FILE" \
    --run-name "$RUN_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --custom-label "$CUSTOM_LABEL" \
    --api-key "$API_KEY"
fi

#!/bin/sh
set -eu

# validate-v028.sh — one-shot validation battery for the v0.28 migration.
# Runs the same tests as the v0.18 baseline (short prompts, thermal-safe),
# with CPU/GPU monitoring, and saves everything under logs/tests|thermal.
#
# Usage: ./scripts/validate-v028.sh [--max-tokens N] [--tag v028-plain]

TAG=${TAG:-v028}
MAX_TOKENS=${MAX_TOKENS:-60}
BASE_URL=${BASE_URL:-http://172.18.100.7:8000}
TS=$(date -u +%Y%m%dT%H%M%SZ)

echo "== validation run: $TAG ($TS) =="

python3 scripts/monitor_thermal.py --out "logs/thermal/$TAG-$TS.csv" &
MONPID=$!
trap 'kill $MONPID 2>/dev/null || true' EXIT INT TERM

python3 scripts/test_baseline.py \
  --base "$BASE_URL" \
  --max-tokens "$MAX_TOKENS" \
  --out "logs/tests/$TAG-$TS.json" || RC=$?
RC=${RC:-0}

kill $MONPID 2>/dev/null || true
trap - EXIT INT TERM

echo "== thermal summary =="
awk -F, 'NR>1{
  if($2!="NA"&&$2+0>gmax)gmax=$2+0;
  if($2!="NA"&&(gmin==""||$2+0<gmin))gmin=$2+0;
  if($5!="NA"&&$5+0>umax)umax=$5+0;
  if($8!="NA"&&$8+0>cmax)cmax=$8+0;
  if($10!="NA"&&$10+0>tmax)tmax=$10+0
} END{
  printf "GPU temp: %s..%s C | util max %s%% | CPU freq max %s MHz | CPU temp max %s C\n",
    gmin, gmax, umax, cmax, tmax
}' "logs/thermal/$TAG-$TS.csv"

echo "== results: logs/tests/$TAG-$TS.json =="
exit "$RC"

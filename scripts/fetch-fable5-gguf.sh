#!/usr/bin/env bash
# fetch-fable5-gguf.sh — загрузка GGUF-файнтюна MiniCPM5-2B-Claude-Fable5-1 (Thinking/Agentic).
#
# Зачем: заменить базовый MiniCPM5-2B (IQ4_XS) более натренированным файнтюном
# GnLOLot/MiniCPM5-2B-Claude-Fable5-1-Thinking-Agentic-GGUF (агентские результаты чуть лучше).
#
# Рецепт: возобновляемый curl-цикл до ТОЧНОГО размера + sha256-сверка с LFS-эталоном
# (hf download на телефонных хотспотах может зависнуть молча — не ждать).
#
# Использование: bash fetch-fable5-gguf.sh [q8|q4|all]   (по умолчанию all = Q8_0, затем Q4_K_M)
set -u
REPO="GnLOLot/MiniCPM5-2B-Claude-Fable5-1-Thinking-Agentic-GGUF"
BASE="https://huggingface.co/${REPO}/resolve/main"
DEST="${HOME}/Projects/vllm-pascal/volumes/models/MiniCPM5-2B-Fable5-GGUF"
LOG_DIR="${HOME}/logs/model-downloads"
mkdir -p "$DEST" "$LOG_DIR"

Q8="MiniCPM5-2B-Claude-Fable5-1-Thinking-Agentic-Q8_0.gguf"
Q4="MiniCPM5-2B-Claude-Fable5-1-Thinking-Agentic-Q4_K_M.gguf"

case "${1:-all}" in
  q8)  files=("$Q8") ;;
  q4)  files=("$Q4") ;;
  all) files=("$Q8" "$Q4") ;;
  *)   echo "usage: $0 [q8|q4|all]"; exit 2 ;;
esac

# имя -> "<размер в байтах>:<sha256 LFS>"
declare -A EXPECT=(
  ["$Q8"]="2679710592:b333575a9ba89324b1c9631d0d53ad9c02b35e6eef1d368bd69621cff19606df"
  ["$Q4"]="1561318272:387aaabc72f07f26a2539f8b63452054516f1dd8a8ef4dcc1100eda1d2cedef1"
)

ts(){ date '+%F %T'; }

for f in "${files[@]}"; do
  exp="${EXPECT[$f]}"; exp_size="${exp%%:*}"; exp_sha="${exp##*:}"
  out="$DEST/$f"
  echo "[$(ts)] === start $f (ожидаем $exp_size байт) ==="
  attempt=0
  while :; do
    cur=0; [ -f "$out" ] && cur=$(stat -c%s "$out")
    [ "$cur" -eq "$exp_size" ] && break
    attempt=$((attempt+1))
    echo "[$(ts)] attempt $attempt: есть $cur / $exp_size байт"
    curl -L -C - --retry 6 --retry-all-errors --retry-delay 5 \
         --speed-limit 4096 --speed-time 90 --connect-timeout 20 \
         -o "$out" "$BASE/$f" || echo "[$(ts)] curl rc=$? (продолжаю цикл)"
    sleep 3
    if [ "$attempt" -ge 80 ]; then echo "[$(ts)] FAIL: превышен лимит попыток"; break; fi
  done
  cur=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$cur" -eq "$exp_size" ]; then
    sum=$(sha256sum "$out" | cut -d' ' -f1)
    if [ "$sum" = "$exp_sha" ]; then
      echo "[$(ts)] OK  sha256 совпал: $f"
    else
      echo "[$(ts)] FAIL sha256 не совпал: $sum != $exp_sha ($f)"
    fi
  else
    echo "[$(ts)] FAIL размер $cur != $exp_size ($f)"
  fi
done
echo "[$(ts)] === готово ==="
ls -la "$DEST" | sed 's/^/    /'

#!/usr/bin/env bash
# Live cne_server throughput: fixed max_tokens, wall-clock tok/s.
# Requires server already running with matching CNE_KERNELS (restart after toggle).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

URL="${CNE_SERVER_URL:-http://127.0.0.1:8080}"
MAX_TOKENS="${CNE_BENCH_MAX_TOKENS:-500}"
export CNE_KERNELS="${CNE_KERNELS:-1}"
RESULTS="${CNE_BENCH_RESULTS:-$ROOT/bench/results}"
LOG="${CNE_BENCH_LOG:-$RESULTS/lfm2-server-velocity.log}"
HISTORY="${CNE_BENCH_HISTORY:-$RESULTS/lfm2-server-velocity.history.tsv}"

mkdir -p "$RESULTS"

STARTED=$(date -Iseconds)
PAYLOAD=$(cat <<EOF
{"model":"lfm2","messages":[{"role":"user","content":"Count from 1 to 100 slowly, one number per line."}],"max_tokens":${MAX_TOKENS},"temperature":0,"stream":false}
EOF
)

echo "=== server velocity started ${STARTED} (CNE_KERNELS=${CNE_KERNELS}, max_tokens=${MAX_TOKENS}) ===" >> "$LOG"

T0=$(date +%s.%N)
RESP=$(curl -sf -H 'Content-Type: application/json' -d "$PAYLOAD" "${URL}/v1/chat/completions")
T1=$(date +%s.%N)
FINISHED=$(date -Iseconds)

ELAPSED=$(awk -v t0="$T0" -v t1="$T1" 'BEGIN { printf "%.3f", t1 - t0 }')

# Parse JSON with python3 (jq not required).
read -r USAGE_COMP FINISH_REASON <<< "$(python3 -c "
import json, sys
body = json.loads(sys.stdin.read())
usage = body.get('usage', {}).get('completion_tokens')
finish = body.get('choices', [{}])[0].get('finish_reason', '')
print(usage if usage is not None else '', finish)
" <<< "$RESP")"

if [[ -z "$USAGE_COMP" || "$USAGE_COMP" == "null" ]]; then
  USAGE_COMP=$(python3 -c "
import json, sys
c = json.loads(sys.stdin.read()).get('choices', [{}])[0].get('message', {}).get('content', '')
print(sum(1 for line in c.splitlines() if line.strip()))
" <<< "$RESP")
fi

TOK_S=""
if [[ -n "$USAGE_COMP" && "$USAGE_COMP" != "0" ]]; then
  TOK_S=$(awk -v n="$USAGE_COMP" -v e="$ELAPSED" 'BEGIN { if (e > 0) printf "%.2f", n / e }')
fi

echo "elapsed_s=${ELAPSED} completion_tokens=${USAGE_COMP} tok_s=${TOK_S:-} finish=${FINISH_REASON}" >> "$LOG"
echo "=== done ${FINISHED} ===" >> "$LOG"

if [[ ! -s "$HISTORY" ]]; then
  printf 'started\tfinished\tmoe_b3\tmax_tokens\tcompletion_tokens\telapsed_s\ttok_s\tfinish_reason\n' >> "$HISTORY"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$STARTED" "$FINISHED" "$CNE_KERNELS" "$MAX_TOKENS" "$USAGE_COMP" "$ELAPSED" "${TOK_S:-}" "$FINISH_REASON" >> "$HISTORY"

if [[ -n "$TOK_S" ]]; then
  echo "server: ${TOK_S} tok/s (${USAGE_COMP} tokens in ${ELAPSED}s, CNE_KERNELS=${CNE_KERNELS})"
else
  echo "server: could not compute tok/s (see ${LOG})"
  exit 1
fi

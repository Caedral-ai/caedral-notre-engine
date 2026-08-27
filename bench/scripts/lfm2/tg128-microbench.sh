#!/usr/bin/env bash
# llama-bench tg128 decode probe for LFM2 (custom kernels A/B via CNE_KERNELS).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

MODEL="${CNE_BENCH_MODEL:-models/lfm2-24b-a2b/LFM2-24B-A2B-Q4_K_M-prepared.gguf}"
BENCH="${LLAMA_BENCH:-/tmp/llama-bench-build/bin/llama-bench}"
RESULTS="${CNE_BENCH_RESULTS:-$ROOT/bench/results}"
LOG="$RESULTS/lfm2-tg128.log"
HISTORY="$RESULTS/lfm2-tg128.history.tsv"
EXIT="$RESULTS/lfm2-tg128.exit"

export CNE_KERNELS="${CNE_KERNELS:-1}"

mkdir -p "$RESULTS"

# Sweep: for i in {1..10}; do ./bench/scripts/lfm2/tg128-microbench.sh; done
#        column -t bench/results/lfm2-tg128.history.tsv
#
# A/B: CNE_KERNELS=0 ./bench/scripts/lfm2/tg128-microbench.sh

STARTED=$(date -Iseconds)
rm -f "$EXIT"
echo "=== llama-bench TG n=128 t=4 started ${STARTED} (CNE_KERNELS=${CNE_KERNELS}) ===" >> "$LOG"

(
  echo 1000 > /proc/self/oom_score_adj
  exec stdbuf -oL -eL "$BENCH" \
    -m "$MODEL" -t 4 -p 0 -n 128 -r 3 -b 512 -ub 512
) >> "$LOG" 2>&1

EXIT_CODE=$?
echo "$EXIT_CODE" > "$EXIT"
FINISHED=$(date -Iseconds)
echo "=== done ${FINISHED} exit=${EXIT_CODE} ===" >> "$LOG"

if [[ ! -s "$HISTORY" ]]; then
  printf 'started\tfinished\tmoe_b3\ttok_s\tstddev\texit\n' >> "$HISTORY"
fi

TOK_S=""
STDEV=""
if [[ "$EXIT_CODE" -eq 0 ]]; then
  TOK_LINE=$(grep 'tg128 |' "$LOG" | tail -1 || true)
  if [[ -n "$TOK_LINE" ]]; then
    TOK_S=$(sed -n 's/.*| *\([0-9.][0-9.]*\) ± \([0-9.][0-9.]*\).*/\1/p' <<< "$TOK_LINE")
    STDEV=$(sed -n 's/.*| *\([0-9.][0-9.]*\) ± \([0-9.][0-9.]*\).*/\2/p' <<< "$TOK_LINE")
  fi
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$STARTED" "$FINISHED" "$CNE_KERNELS" "${TOK_S:-}" "${STDEV:-}" "$EXIT_CODE" >> "$HISTORY"

if [[ -n "$TOK_S" ]]; then
  echo "tg128: ${TOK_S} ± ${STDEV} tok/s (CNE_KERNELS=${CNE_KERNELS}, logged to ${HISTORY})"
fi

exit "$EXIT_CODE"

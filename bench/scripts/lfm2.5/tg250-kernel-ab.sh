#!/usr/bin/env bash
# LFM2.5-8B-A1B kernel A/B: llama-bench tg250, pp512 in-run, t4, 5 reps/arm.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

MODEL="${CNE_BENCH_MODEL:-models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf}"
BENCH="${LLAMA_BENCH:-/tmp/llama-bench-build/bin/llama-bench}"
RESULTS="${CNE_BENCH_RESULTS:-$ROOT/bench/results}"
TSV="$RESULTS/lfm25-tg250-kernels.tsv"
LOG="$RESULTS/lfm25-tg250-kernels.log"

mkdir -p "$RESULTS"

if [[ ! -x "$BENCH" ]]; then
  echo "missing $BENCH — build with bench/README.md" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

if pgrep -x cne_server >/dev/null 2>&1; then
  echo "stop cne_server before microbench (OOM risk on 16 GB)" >&2
  exit 1
fi

if [[ ! -s "$TSV" ]]; then
  printf 'run_ts\tCNE_KERNELS\ttest\ttok_s\tstddev_ms\texit\n' >"$TSV"
fi

RUN_TS=$(date -Iseconds)
echo "=== LFM2.5 tg250 kernel A/B started $RUN_TS ===" | tee -a "$LOG"

for k in 1 0; do
  echo "--- CNE_KERNELS=$k ($(date -Iseconds)) ---" | tee -a "$LOG"
  set +e
  OUT=$(CNE_KERNELS=$k "$BENCH" \
    -m "$MODEL" -t 4 -p 512 -n 250 -r 5 -b 512 -ub 512 2>&1)
  EC=$?
  set -e
  echo "$OUT" | tee -a "$LOG"
  echo "exit=$EC" | tee -a "$LOG"

  while IFS= read -r line; do
    if [[ "$line" =~ tg250 ]]; then
      tok=$(sed -n 's/.*| *\([0-9.][0-9.]*\) ± \([0-9.][0-9.]*\).*/\1/p' <<<"$line")
      sd=$(sed -n 's/.*| *\([0-9.][0-9.]*\) ± \([0-9.][0-9.]*\).*/\2/p' <<<"$line")
      printf '%s\t%s\ttg250\t%s\t%s\t%s\n' "$RUN_TS" "$k" "${tok:-}" "${sd:-}" "$EC" >>"$TSV"
    fi
  done <<<"$OUT"
done

echo "=== done $(date -Iseconds) — see $TSV ===" | tee -a "$LOG"

python3 - "$TSV" "$RUN_TS" <<'PY'
import sys
from statistics import mean, stdev

tsv, run_ts = sys.argv[1], sys.argv[2]
rows = []
with open(tsv) as f:
    next(f, None)
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6 or parts[0] != run_ts:
            continue
        k, test, tok, sd = parts[1], parts[2], parts[3], parts[4]
        if test == "tg250" and tok:
            rows.append((int(k), float(tok)))

for k in (1, 0):
    vals = [v for kk, v in rows if kk == k]
    if not vals:
        print(f"CNE_KERNELS={k}: no tg250 samples")
        continue
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    print(f"CNE_KERNELS={k}: tg250 {m:.2f} ± {s:.2f} tok/s (n={len(vals)})")

on = [v for kk, v in rows if kk == 1]
off = [v for kk, v in rows if kk == 0]
if on and off:
    pct = 100.0 * (mean(on) - mean(off)) / mean(off)
    print(f"delta: {pct:+.1f}% (kernels on vs off)")
PY

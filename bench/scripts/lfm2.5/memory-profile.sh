#!/usr/bin/env bash
# LFM2.5-8B-A1B memory/velocity probes (reference: 16 GiB host).
# Logs RSS (VmHWM), tok/s, and stream hit-rate to bench/results/.
# Sustained decode (250 tok): CNE_BENCH_N_GEN=250 ./bench/scripts/lfm2.5/memory-profile.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${CNE_BENCH_MODEL:-models/lfm2.5-8b-a1b/lfm25-8b-a1b-UD-Q4_K_XL-prepared.gguf}"
RESULTS="${CNE_BENCH_RESULTS:-$ROOT/bench/results}"
TSV="$RESULTS/lfm25-memory.tsv"
N_GEN="${CNE_BENCH_N_GEN:-32}"

mkdir -p "$RESULTS"
if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

if [[ ! -s "$TSV" ]]; then
  printf 'run_ts\tprofile\tstream\tcap_gib\tdense\tctx\ttok_s\trss_gib\thit_pct\n' >"$TSV"
fi

run_case() {
  local name=$1 stream=$2 cap=$3 dense=$4 ctx=$5 rebind=$6
  local out tok rss hit
  echo "=== $name (stream=$stream cap=${cap}GiB dense=$dense ctx=$ctx) ==="
  out=$(CNE_STREAM=$stream CNE_DENSE=$dense CNE_KERNELS=1 CNE_THREADS=4 \
        CNE_CTX=$ctx CNE_SESSION_MAX=1 CNE_IGNORE_EOS=1 \
        ./build/tools/cne_bench "$MODEL" "$cap" "$N_GEN" "$N_GEN" "$rebind" 2>&1) || true
  tok=$(sed -n 's/^tok\/s *: //p' <<<"$out" | tail -1)
  rss=$(sed -n 's/.*rss=\([0-9.]*\) GiB.*/\1/p' <<<"$out" | tail -1)
  hit=$(sed -n 's/^cache hit rate *: \([0-9.]*\)%.*/\1/p' <<<"$out" | tail -1)
  [[ -z "$hit" ]] && hit="0"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Iseconds)" "$name" "$stream" "$cap" "$dense" "$ctx" "$tok" "$rss" "$hit" >>"$TSV"
  echo "$out" | tail -8
  echo
}

RUN_TS=$(date -Iseconds)
echo "=== LFM2.5 memory profile $RUN_TS ==="

# Phase 2 baseline (plan): no stream
run_case "baseline-mmap" 0 0 mmap 2048 0

# 4 GiB target: dense anon (no stream) — mmap drop + per-step expert trim
run_case "anon-4g-nostream" 0 0 anon 1024 0

# 4 GiB target: dense anon + 3 GiB expert cache (stream)
run_case "anon-4g-stream3g" 1 3 anon 1024 1

# Legacy stream probes (UD-Q4_K_XL without anon drop — not 4 GiB safe)
run_case "stream-1g-mmap" 1 1 mmap 1024 1

echo "results -> $TSV"
column -t "$TSV" | tail -6

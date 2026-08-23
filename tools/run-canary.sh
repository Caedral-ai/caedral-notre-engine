#!/bin/sh
# PL-T1 long-gen canary runner. Sequential MemoryMax-scoped model jobs -
# ONE at a time (PL-T0 hardware rule). Lossless arm (rebind=0) vs lossy arm
# (rebind=1 + optional QUALITY_FLAGS, e.g. future SOE_EXPERT_MASS/quality).
# Then drift_gate canary (degeneration loops, divergence) over both streams.
#
# usage: run-canary.sh [cap_gib] [n_gen] [ctx]
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL=${MODEL:-models/qwen3.6-35b-a3b-q8_0/Qwen3.6-35B-A3B-Q8_0-prepared.gguf}
CAP=${1:-7}
GEN=${2:-2048}
CTX=${3:-2304}
SCOPE_MEM=${SCOPE_MEM:-12G}
OUT=${OUT:-/tmp/opencode/canary}
QUALITY_FLAGS=${QUALITY_FLAGS:-}
mkdir -p "$OUT/ref" "$OUT/cand"

run_arm() {  # $1 rebind  $2 outprefix  $3 extra env
    systemd-run --user --scope -p MemoryMax=$SCOPE_MEM \
        env SOE_CTX=$CTX SOE_LANES=4 SOE_DUMP_LOGITS_EVERY=16 $3 \
        "$DIR/../build/tools/soe_streaming_bench" "$MODEL" "$CAP" "$GEN" 0 "$1" \
        > "$OUT/$2.toks" 2> "$OUT/$2.err"
}

echo "[canary] arm 1/2: lossless reference ($GEN tokens)"
run_arm 0 ref ""
echo "[canary] arm 2/2: lossy candidate"
run_arm 1 cand "$QUALITY_FLAGS"

python3 "$DIR/drift_gate.py" canary \
    "$OUT/ref.toks" "$OUT/cand.toks" \
    --min-tokens "$((GEN / 2))" \
    --loop-period-max "${LOOP_PERIOD_MAX:-24}" \
    --loop-repeat-min "${LOOP_REPEAT_MIN:-8}"
python3 "$DIR/drift_gate.py" logits \
    --ref-dir "$OUT/ref" --cand-dir "$OUT/cand" \
    ${KL_BAND:+--kl-band "$KL_BAND"} ${KL_MAX:+--kl-max "$KL_MAX"}

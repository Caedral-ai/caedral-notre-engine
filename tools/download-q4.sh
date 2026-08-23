#!/bin/sh
# L1 runtime artifact pipeline: fetch unsloth UD-Q4_K_XL + align expert
# tensors for streaming (soe-prepare). Idempotent: completed steps skipped.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR=models/qwen3.6-35b-a3b-q4_k_xl
OUT="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

"$DIR/download-model.sh" "$MODEL_DIR"

echo "[q4] aligning expert tensors (soe-prepare) ..."
"$DIR/../build/tools/soe_prepare" "$OUT"

echo "[q4] done -> ${OUT%.gguf}-prepared.gguf"

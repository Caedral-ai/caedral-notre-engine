#!/bin/sh
# L1 runtime artifact: unsloth dynamic q4 (UD-Q4_K_XL) + alignment.
# Downloads the official HF quant (22.4 GB), then 4096-aligns all expert
# tensors with soe-prepare (streaming requirement). Resumable (-C -).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR=models/qwen3.6-35b-a3b-q8_0
OUT="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"

if [ ! -f "$OUT" ] || [ "$(stat -c%s "$OUT")" -lt 22000000000 ]; then
    echo "[q4] downloading UD-Q4_K_XL (~22.4 GB) ..."
    curl -sL --retry 5 --retry-delay 10 -C - -o "$OUT" "$URL"
fi
echo "[q4] downloaded: $(stat -c%s "$OUT") bytes"

echo "[q4] aligning expert tensors (soe-prepare) ..."
"$DIR/../build/tools/soe_prepare" "$OUT"

echo "[q4] done -> ${OUT%.gguf}-prepared.gguf"

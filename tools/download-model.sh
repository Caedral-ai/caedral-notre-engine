#!/bin/sh
# download-model.sh — fetch + align the canonical runtime artifact.
#
# Downloads the unsloth dynamic q4 quant (UD-Q4_K_XL) with MTP layers
# preserved, then runs soe-prepare to 4096-align all expert tensors for
# O_DIRECT streaming. Resumable; verifies sha256 after download.
#
# Usage:
#   ./tools/download-model.sh
set -eu

REPO="unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_DIR="models/qwen3.6-35b-a3b-q4_k_xl"
DEST="$MODEL_DIR/$FILE"
URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/$FILE"

# single-instance guard
LOCK=/tmp/soe_download.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[dl] another instance is already running"; exit 1; }

mkdir -p "$MODEL_DIR"

need_fetch=1
if [ -f "$DEST" ]; then
    SZ=$(stat -c%s "$DEST")
    if [ "$SZ" -ge 22853663008 ]; then
        echo "[dl] artifact already present ($SZ bytes)"
        need_fetch=0
    else
        echo "[dl] resuming partial ($SZ / 22853663008 bytes)"
    fi
fi

if [ "$need_fetch" = 1 ]; then
    echo "[dl] downloading $FILE from $REPO ..."
    curl -sL --fail --retry 5 --retry-delay 10 --continue-at - \
         --speed-time 60 --speed-limit 10000 \
         -o "$DEST" "$URL"
    echo "[dl] downloaded: $(stat -c%s "$DEST") bytes"
fi

echo "[dl] aligning expert tensors (soe-prepare) ..."
./build/tools/soe_prepare "$DEST"

echo "[dl] done -> ${DEST%.gguf}-prepared.gguf"

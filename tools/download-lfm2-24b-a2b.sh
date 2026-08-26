#!/bin/sh
# download-lfm2-24b-a2b.sh — fetch + align the LFM2-24B-A2B runtime artifact.
#
# Downloads the official LiquidAI Q4_K_M quant (24B total / 2.3B active MoE,
# hybrid conv+attention, no MTP). Runs cne-prepare to 4096-align all expert
# tensors for O_DIRECT streaming. Resumable; verifies sha256 after download.
#
# Serving note: this model runs NO-STREAM (mmap-dense decode). The prepared
# artifact is still aligned so the streaming path stays available for A/B.
#
# Usage:
#   ./tools/download-lfm2-24b-a2b.sh
set -eu

REPO="LiquidAI/LFM2-24B-A2B-GGUF"
FILE="LFM2-24B-A2B-Q4_K_M.gguf"
EXPECTED_SHA256="SET_AFTER_FIRST_DOWNLOAD"
MODEL_DIR="models/lfm2-24b-a2b"
DEST="$MODEL_DIR/$FILE"
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
EXPECTED_SIZE=14420000000   # approximate; exact value checked against sha256

# single-instance guard
LOCK=/tmp/cne_download_lfm2.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[dl] another instance is already running"; exit 1; }

mkdir -p "$MODEL_DIR"

need_fetch=1
if [ -f "$DEST" ]; then
    SZ=$(stat -c%s "$DEST")
    if [ "$SZ" -ge $EXPECTED_SIZE ]; then
        echo "[dl] artifact already present ($SZ bytes)"
        need_fetch=0
    else
        echo "[dl] resuming partial ($SZ bytes)"
    fi
fi

if [ "$need_fetch" = 1 ]; then
    echo "[dl] downloading $FILE from $REPO ..."
    curl -sL --fail --retry 5 --retry-delay 10 --continue-at - \
         --speed-time 60 --speed-limit 10000 \
         -o "$DEST" "$URL"
    echo "[dl] downloaded: $(stat -c%s "$DEST") bytes"
fi

echo "[dl] verifying sha256 ..."
GOT=$(sha256sum "$DEST" | cut -d' ' -f1)
if [ "$EXPECTED_SHA256" = "SET_AFTER_FIRST_DOWNLOAD" ]; then
    echo "[dl] NOTE: pinning hash on first download: $GOT"
    sed -i "s/SET_AFTER_FIRST_DOWNLOAD/$GOT/" "$0"
elif [ "$GOT" != "$EXPECTED_SHA256" ]; then
    echo "[dl] FATAL: sha256 mismatch (got $GOT, want $EXPECTED_SHA256)" >&2
    echo "[dl] corrupted download - delete $DEST and retry" >&2
    exit 1
fi
echo "[dl] sha256 OK"

echo "[dl] aligning expert tensors (cne-prepare) ..."
./build/tools/cne_prepare "$DEST"

echo "[dl] done -> ${DEST%.gguf}-prepared.gguf"

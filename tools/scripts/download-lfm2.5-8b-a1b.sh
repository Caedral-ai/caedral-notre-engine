#!/bin/sh
# download-lfm2.5-8b-a1b.sh — fetch + align the LFM2.5-8B-A1B runtime artifact.
#
# Downloads AtomicChat imatrix UD-Q4_K_XL (8.5B total / ~1.5B active MoE,
# hybrid conv+attention, no MTP). Runs cne-prepare to 4096-align expert
# tensors for O_DIRECT streaming. Resumable; verifies sha256 after download.
#
# Usage:
#   ./tools/scripts/download-lfm2.5-8b-a1b.sh
set -eu

REPO="AtomicChat/LFM2.5-8B-A1B-GGUF"
FILE="lfm25-8b-a1b-UD-Q4_K_XL.gguf"
EXPECTED_SHA256="283b12943743c5bb54b7f9fc8f7076c8ea32d163a06fb0e6f525178e7232c588"
MODEL_DIR="models/lfm2.5-8b-a1b"
DEST="$MODEL_DIR/$FILE"
URL="https://huggingface.co/$REPO/resolve/main/$FILE"
EXPECTED_SIZE=5219053088

LOCK=/tmp/cne_download_lfm25.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[dl] another instance is already running"; exit 1; }

mkdir -p "$MODEL_DIR"

need_fetch=1
if [ -f "$DEST" ]; then
    SZ=$(stat -c%s "$DEST")
    if [ "$SZ" -ge "$EXPECTED_SIZE" ]; then
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
if [ "$GOT" != "$EXPECTED_SHA256" ]; then
    echo "[dl] FATAL: sha256 mismatch (got $GOT, want $EXPECTED_SHA256)" >&2
    echo "[dl] corrupted download - delete $DEST and retry" >&2
    exit 1
fi
echo "[dl] sha256 OK"

echo "[dl] aligning expert tensors (cne-prepare) ..."
./build/tools/cne_prepare "$DEST"

echo "[dl] done -> ${DEST%.gguf}-prepared.gguf"

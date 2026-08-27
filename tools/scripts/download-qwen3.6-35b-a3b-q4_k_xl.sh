#!/bin/sh
# download-qwen3.6-35b-a3b-q4_k_xl.sh — fetch + align the canonical runtime artifact.
#
# Downloads the unsloth dynamic q4 quant (UD-Q4_K_XL) with MTP layers
# preserved, then runs cne-prepare to 4096-align all expert tensors for
# O_DIRECT streaming. Resumable; verifies sha256 after download.
#
# Usage:
#   ./tools/scripts/download-qwen3.6-35b-a3b-q4_k_xl.sh
set -eu

REPO="unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
EXPECTED_SHA256="55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"
MODEL_DIR="models/qwen3.6-35b-a3b-q4_k_xl-mtp"
DEST="$MODEL_DIR/$FILE"
URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/$FILE"

# single-instance guard
LOCK=/tmp/cne_download.lock
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

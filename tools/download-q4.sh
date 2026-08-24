#!/bin/sh
# L1 runtime artifact pipeline: fetch unsloth UD-Q4_K_XL (MTP-preserved) +
# align expert tensors for streaming (soe-prepare). Idempotent + single-instance.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR=models/qwen3.6-35b-a3b-q4_k_xl
OUT="$MODEL_DIR/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
EXPECTED_SHA256="707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450"
EXPECTED_SIZE=22360456160

# single-instance guard: concurrent transfers corrupt the output file
LOCK=/tmp/soe_q4_download.lock
exec 9>"$LOCK"
flock -n 9 || { echo "[q4] another download is already running"; exit 1; }

need_fetch=1
if [ -f "$OUT" ]; then
    SZ=$(stat -c%s "$OUT")
    if [ "$SZ" -eq "$EXPECTED_SIZE" ] && \
       echo "${EXPECTED_SHA256}  $OUT" | sha256sum -c - > /dev/null 2>&1; then
        need_fetch=0
        echo "[q4] artifact already present and verified"
    elif [ "$SZ" -gt "$EXPECTED_SIZE" ]; then
        rm -f "$OUT"
    fi
fi

if [ "$need_fetch" = 1 ]; then
    echo "[q4] fetching UD-Q4_K_XL (22360456160 bytes; existing partial resumed)"
    curl -sL --retry 5 --retry-delay 10 --continue-at - \
         --speed-time 60 --speed-limit 10000 \
         -o "$OUT" "$URL"
fi

echo "[q4] downloaded: $(stat -c%s "$OUT") bytes"
echo "${EXPECTED_SHA256}  $OUT" | sha256sum -c - || {
    echo "[q4] CHECKSUM FAILED - deleting corrupt file"; rm -f "$OUT"; exit 1;
}
echo "[q4] checksum verified"

"$DIR/../build/tools/soe_prepare" "$OUT"

echo "[q4] done -> ${OUT%.gguf}-prepared.gguf"

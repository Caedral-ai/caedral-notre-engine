#!/usr/bin/env bash
# Download a supported model GGUF from Hugging Face.
#
# Usage:
#   ./tools/download-model.sh                    # default: Qwen3.6-35B-A3B UD-Q4_K_XL
#   ./tools/download-model.sh <dest-dir> [file]  # override destination / repo file
#
# Destination defaults to models/<model-id>/ inside the repo (gitignored).
# Downloads are resumable; the default file is verified against a known sha256.
set -euo pipefail

HF_REPO="${HF_REPO:-unsloth/Qwen3.6-35B-A3B-GGUF}"
DEFAULT_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
DEFAULT_SHA256=""   # pinned after first verified download
DEFAULT_DEST="models/qwen3.6-35b-a3b-q4_k_xl"

DEST_DIR="${1:-$DEFAULT_DEST}"
FILE_NAME="${2:-$DEFAULT_FILE}"
URL="https://huggingface.co/${HF_REPO}/resolve/main/${FILE_NAME}"
TARGET="${DEST_DIR}/${FILE_NAME}"

need_bytes() {
    curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc '0-9'
}

command -v curl >/dev/null || { echo "error: curl is required" >&2; exit 1; }

mkdir -p "$DEST_DIR"

SIZE=$(need_bytes)
if [[ -f "$TARGET" && "$(stat -c%s "$TARGET")" == "$SIZE" ]]; then
    echo "already complete: $TARGET"
else
    if [[ -f "$TARGET" ]]; then
        echo "resuming partial: $TARGET ($(stat -c%s "$TARGET") bytes)"
    fi
    echo "repo:   ${HF_REPO}"
    echo "file:   ${FILE_NAME} ($(( SIZE / 1024 / 1024 / 1024 )) GiB)"
    echo "url:    ${URL}"
    curl -L --fail --retry 5 --retry-delay 5 --continue-at - -o "$TARGET" "$URL"
fi

if [[ -n "$DEFAULT_SHA256" && "$FILE_NAME" == "$DEFAULT_FILE" ]]; then
    echo "verifying sha256..."
    echo "${DEFAULT_SHA256}  ${TARGET}" | sha256sum -c -
elif [[ -z "$DEFAULT_SHA256" ]]; then
    echo "note: no pinned sha256 for ${FILE_NAME}; skipping verification"
fi

echo "done: $TARGET"

#!/bin/sh
# P5 measured-best stable config: ~1.05-1.10 tok/s, RSS HWM <= 11.5 GiB,
# identity PASS (Qwen3.6-35B prepared file, 64 greedy tokens).
# Requirements: models/qwen3.6-35b-a3b-q8_0/Qwen3.6-35B-A3B-Q8_0-prepared.gguf
#   (generate with: ./build/tools/soe_prepare <original.gguf>)
# Measured 2026-08-22: anon policy = 27-33 major faults (mmap: 22k),
# hit-rate ~76%, fills O_DIRECT across 4 lanes.
set -e
DIR="$(dirname "$0")"
MODEL="${MODEL:-models/qwen3.6-35b-a3b-q8_0/Qwen3.6-35B-A3B-Q8_0-prepared.gguf}"
exec env SOE_DENSE=anon SOE_LANES=4 \
    "$DIR/../build/tools/soe_streaming_bench" "$MODEL" "${CAP:-7}" "${N_GEN:-64}" 0 1 "$@"

#!/bin/sh
# Perplexity measurement over a fixed text corpus (isolated protocol).
# Corpus: first 6300 words of the standard PTB test split, stored locally at
# models/eval/ptb16.test.txt (LDC-licensed: do NOT redistribute).
# Fetch once: https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt
#
# Uses a separate measurement build tree (product CMake stays lib-only):
#   cmake -B build-meas -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_MESSAGE_LOG_LEVEL=WARNING
#   cmake --build build-meas -j --target llama-perplexity
MODEL="${1:-models/qwen3.6-35b-a3b-q4_k_xl-mtp/Qwen3.6-35B-A3B-UD-Q4_K_XL-prepared.gguf}"
exec ./build-meas/bin/llama-perplexity -m "$MODEL" \
    -f models/eval/ptb16.test.txt -c 512 -ngl 0 -t 8 "$@"

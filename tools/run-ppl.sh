#!/bin/sh
# PL quality measurement: PTB-16 drift perplexity (fixed protocol).
# Corpus: first 6300 words of the standard PTB test split, stored locally at
# models/eval/ptb16.test.txt (LDC-licensed: do NOT redistribute).
# Fetch once: https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt
# Requires the isolated measurement build: see internal docs (build-meas).
# Lossless reference on prepared Qwen3.6-35B: PPL = 13.3874 +/- 0.62.
MODEL="${1:-models/qwen3.6-35b-a3b-q8_0/Qwen3.6-35B-A3B-Q8_0-prepared.gguf}"
exec ./build-meas/bin/llama-perplexity -m "$MODEL" \
    -f models/eval/ptb16.test.txt -c 512 -ngl 0 -t 8 "$@"

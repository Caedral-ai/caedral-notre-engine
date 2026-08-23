#!/usr/bin/env python3
"""drift-gate: quality comparison between two token streams (lossless vs lossy).

Usage:
    drift_gate.py <reference_tokens_file> <candidate_tokens_file> [--max-n 3]

Each input file: whitespace-separated token ids from one generation.
Reports:
  - exact-match ratio over the aligned length
  - first divergence index (-1 = identical)
  - n-gram overlap for n=1..max_n (Bleu-like precision on the candidate)
Exit code 0 if exact; 1 if divergent (informational - bands are chosen per
technique, see internal-docs/LOSSY_TRACK.md section 4).
"""
import sys
import argparse
from collections import Counter


def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--max-n", type=int, default=3)
    args = ap.parse_args()

    ref = open(args.reference).read().split()
    cand = open(args.candidate).read().split()
    if not ref or not cand:
        print("drift-gate: empty input")
        return 2

    aligned = min(len(ref), len(cand))
    exact = sum(1 for i in range(aligned) if ref[i] == cand[i])
    first_div = next((i for i in range(aligned) if ref[i] != cand[i]), -1)

    print(f"lengths: ref={len(ref)} cand={len(cand)} aligned={aligned}")
    print(f"exact match : {exact}/{aligned} ({100.0 * exact / aligned:.2f}%)")
    print(f"first diff  : index {first_div}"
          + (" (identical)" if first_div < 0 else ""))

    for n in range(1, args.max_n + 1):
        g_ref = ngrams(ref[:aligned], n)
        g_cand = ngrams(cand[:aligned], n)
        overlap = sum(min(c, g_ref[g]) for g, c in g_cand.items())
        total = max(1, aligned - n + 1)
        print(f"{n}-gram overlap vs reference: {100.0 * overlap / total:.2f}%")

    return 0 if first_div < 0 and len(ref) == len(cand) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""drift-gate: PL quality harness for lossy-vs-lossless comparisons.

Modes:
  tokens <ref> <cand>              token-stream drift (exact %, first div, n-grams)
  logits --ref-dir D --cand-dir D  per-token top-K KL between dumped logit files
  canary --ref F --cand F          long-generation degeneration detector

Logit KL approximation (documented): for each dumped token, take the K
highest-reference logits, renormalize BOTH vectors over that subset and sum
pr*log(pr/pc). This is subset-renormalized KL, NOT full-vocabulary KL: mass
outside the reference top-K is ignored. Good enough for band monitoring;
bands are configurable because the absolute values depend on K.

Exit codes: 0 = within bands / identical, 1 = out of band or divergent,
2 = usage/input error.
"""
import argparse
import os
import sys
from collections import Counter


# ---------------------------------------------------------------- tokens ---
def read_tokens(path):
    return open(path).read().split()


def compare_tokens(ref, cand):
    aligned = min(len(ref), len(cand))
    exact = sum(1 for i in range(aligned) if ref[i] == cand[i])
    first_div = next((i for i in range(aligned) if ref[i] != cand[i]), -1)
    return {
        "len_ref": len(ref),
        "len_cand": len(cand),
        "aligned": aligned,
        "exact": exact,
        "exact_ratio": exact / aligned if aligned else 0.0,
        "first_div": first_div,
    }


def print_tokens_report(res):
    print(f"lengths: ref={res['len_ref']} cand={res['len_cand']} "
          f"aligned={res['aligned']}")
    print(f"exact match : {res['exact']}/{res['aligned']} "
          f"({100.0 * res['exact_ratio']:.2f}%)")
    fd = res["first_div"]
    print(f"first diff  : index {fd}" + (" (identical)" if fd < 0 else ""))


def ngram_overlap(ref, cand, max_n):
    out = {}
    for n in range(1, max_n + 1):
        g_ref = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        g_cand = Counter(tuple(cand[i:i + n]) for i in range(len(cand) - n + 1))
        overlap = sum(min(c, g_ref[g]) for g, c in g_cand.items())
        out[n] = overlap / max(1, len(ref) - n + 1)
    return out


# ---------------------------------------------------------------- logits ---
def _load_f32(path):
    import numpy as np
    return np.fromfile(path, dtype=np.float32)


def _softmax(v):
    import numpy as np
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def kl_topk(ref, cand, k):
    """Renormalized top-K KL(ref||cand). Approximation - see module docstring."""
    import numpy as np
    idx = np.argsort(ref)[-k:]
    pr = _softmax(np.asarray(ref)[idx])
    pc = _softmax(np.asarray(cand)[idx])
    return float(np.sum(pr * np.log((pr + 1e-12) / (pc + 1e-12))))


def compare_logit_dirs(ref_dir, cand_dir, k):
    import numpy as np
    names = sorted(f for f in os.listdir(ref_dir) if f.endswith(".f32"))
    pairs, missing = [], []
    for f in names:
        p = os.path.join(cand_dir, f)
        (pairs if os.path.exists(p) else missing).append(f)
    kls = []
    for f in pairs:
        r = _load_f32(os.path.join(ref_dir, f))
        c = _load_f32(os.path.join(cand_dir, f))
        if r.size != c.size:
            raise ValueError(f"size mismatch in {f}: {r.size} vs {c.size}")
        kls.append(kl_topk(r, c, k))
    kls.sort()
    n = len(kls)
    stats = {
        "pairs": n,
        "missing_in_candidate": len(missing),
        "mean": sum(kls) / n if n else 0.0,
        "p95": kls[int(0.95 * (n - 1))] if n else 0.0,
        "max": kls[-1] if n else 0.0,
    }
    return stats


# ---------------------------------------------------------------- canary ---
def detect_loop(tokens, period_max, repeat_min):
    """Detect a trailing repetition pattern: returns dict(period, repeats,
    covered) for the most-covering period <= period_max with repeats >=
    repeat_min, else None."""
    n = len(tokens)
    best = None
    limit = min(period_max, max(1, n // max(1, repeat_min)))
    for p in range(1, limit + 1):
        reps = 0
        while n - (reps + 2) * p >= 0 and \
                tokens[n - (reps + 1) * p:n - reps * p] == \
                tokens[n - (reps + 2) * p:n - (reps + 1) * p]:
            reps += 1
        total = (reps + 1) * p
        if reps + 1 >= repeat_min and (best is None or total > best["covered"]):
            best = {"period": p, "repeats": reps + 1, "covered": total}
    return best


# -------------------------------------------------------------------- cli --
def cmd_tokens(a):
    ref = read_tokens(a.reference)
    cand = read_tokens(a.candidate)
    res = compare_tokens(ref, cand)
    print_tokens_report(res)
    for n, v in ngram_overlap(ref[:res["aligned"]], cand[:res["aligned"]],
                              a.max_n).items():
        print(f"{n}-gram overlap vs reference: {100.0 * v:.2f}%")
    return 0 if res["first_div"] < 0 and \
                res["len_ref"] == res["len_cand"] else 1


def cmd_logits(a):
    stats = compare_logit_dirs(a.ref_dir, a.cand_dir, a.top_k)
    print(f"logit pairs compared : {stats['pairs']}"
          f" (missing in candidate: {stats['missing_in_candidate']})")
    print(f"KL top-{a.top_k} mean     : {stats['mean']:.5f}"
          f"  (band <= {a.kl_band})")
    print(f"KL top-{a.top_k} p95      : {stats['p95']:.5f}")
    print(f"KL top-{a.top_k} max      : {stats['max']:.5f}"
          f"  (band <= {a.kl_max})")
    approx = ("approximation note: subset-renormalized KL over the "
              f"reference top-{a.top_k}; not full-vocabulary KL")
    print(approx)
    return 0 if stats["mean"] <= a.kl_band and stats["max"] <= a.kl_max else 1


def cmd_canary(a):
    ref = read_tokens(a.reference)
    cand = read_tokens(a.candidate)
    res = compare_tokens(ref, cand)
    print_tokens_report(res)
    for n, v in ngram_overlap(ref[:res["aligned"]], cand[:res["aligned"]],
                              3).items():
        print(f"{n}-gram overlap vs reference: {100.0 * v:.2f}%")

    fail = False
    if len(cand) < a.min_tokens:
        print(f"CANARY FAIL: generated {len(cand)} < min {a.min_tokens} tokens")
        fail = True
    else:
        print(f"length check: {len(cand)} >= {a.min_tokens} OK")
    loop = detect_loop(cand, a.loop_period_max, a.loop_repeat_min)
    if loop:
        print(f"CANARY FAIL: degeneration loop period={loop['period']} "
              f"repeats={loop['repeats']} covered={loop['covered']} tokens")
        fail = True
    else:
        print(f"loop check: none (period<={a.loop_period_max}, "
              f"repeats>={a.loop_repeat_min} scanned)")
    return 1 if fail else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drift-gate")
    sub = ap.add_subparsers(dest="cmd")

    pt = sub.add_parser("tokens", help="token-stream drift report")
    pt.add_argument("reference")
    pt.add_argument("candidate")
    pt.add_argument("--max-n", type=int, default=3)
    pt.set_defaults(fn=cmd_tokens)

    pl = sub.add_parser("logits", help="top-K KL band over dumped logits")
    pl.add_argument("--ref-dir", required=True)
    pl.add_argument("--cand-dir", required=True)
    pl.add_argument("--top-k", type=int, default=32)
    pl.add_argument("--kl-band", type=float, default=0.05,
                    help="mean KL ceiling")
    pl.add_argument("--kl-max", type=float, default=1.0,
                    help="per-token KL ceiling")
    pl.set_defaults(fn=cmd_logits)

    pc = sub.add_parser("canary", help="long-gen degeneration detector")
    pc.add_argument("reference")
    pc.add_argument("candidate")
    pc.add_argument("--loop-period-max", type=int, default=24)
    pc.add_argument("--loop-repeat-min", type=int, default=8)
    pc.add_argument("--min-tokens", type=int, default=1024)
    pc.set_defaults(fn=cmd_canary)

    # legacy: two bare positional files -> tokens mode
    if argv is not None and len(argv) >= 2 and not argv[0].startswith("-") \
            and argv[0] not in ("tokens", "logits", "canary") \
            and os.path.exists(argv[0]) and os.path.exists(argv[1]):
        argv = ["tokens"] + argv
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 2
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else []))

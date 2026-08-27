#!/usr/bin/env python3
"""Model-free unit tests for tools/drift_gate.py (perplexity / PL-T1).
Runs standalone: python3 test_drift_gate.py  (exit 0 = all passed)."""
import importlib.util
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, "..", "..", "tools", "drift_gate.py")

spec = importlib.util.spec_from_file_location("drift_gate", TOOL)
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_compare_tokens():
    r = dg.compare_tokens(["1", "2", "3"], ["1", "2", "3"])
    assert r["first_div"] == -1 and r["exact_ratio"] == 1.0
    r = dg.compare_tokens(["1", "2", "3"], ["1", "2"])
    assert r["aligned"] == 2 and r["first_div"] == -1  # prefix, not divergence
    r = dg.compare_tokens(["1", "2", "3", "4"], ["1", "2", "9", "4"])
    assert r["first_div"] == 2 and abs(r["exact_ratio"] - 0.75) < 1e-12


def test_detect_loop():
    periodic = ["5", "6", "7"] * 30
    loop = dg.detect_loop(periodic, period_max=24, repeat_min=8)
    assert loop is not None and loop["period"] == 3 and loop["repeats"] >= 8
    varied = [str((i * i * 37 + i * 11) % 9973) for i in range(500)]
    assert dg.detect_loop(varied, period_max=24, repeat_min=8) is None
    # short input cannot meet repeat requirement
    assert dg.detect_loop(["1", "1"], period_max=24, repeat_min=8) is None


def _softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


def test_kl_topk():
    import numpy as np
    ref = [3.0, 2.0, 1.0, 0.0]
    same = dg.kl_topk(ref, list(ref), 4)
    assert approx(same, 0.0) or same < 1e-9

    # independent expected-value computation for a perturbed candidate
    cand = [3.0, 2.0, 0.0, 1.0]
    pr = np.array(_softmax(ref))
    pc = np.array(_softmax(cand))
    expected = float(np.sum(pr * np.log((pr + 1e-12) / (pc + 1e-12))))
    got = dg.kl_topk(ref, cand, 4)
    assert approx(got, expected, 1e-9)

    # closer candidate -> smaller KL (monotonic sanity)
    near = dg.kl_topk(ref, [3.0, 2.0, 1.0, 0.2], 4)
    far = dg.kl_topk(ref, [0.2, 0.1, 3.0, 2.0], 4)
    assert near < far


def _run_cli(args):
    return subprocess.run(
        [sys.executable, TOOL] + args,
        capture_output=True, text=True)


def test_cli_exit_codes():
    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a.toks")
        b = os.path.join(td, "b.toks")
        open(a, "w").write("10 20 30\n" * 50)
        open(b, "w").write("10 20 30\n" * 50)
        r = _run_cli(["tokens", a, b])
        assert r.returncode == 0, r.stderr
        # degenerate candidate must fail canary
        open(b, "w").write(" ".join(["42", "43"] * 25))
        r = _run_cli(["canary", a, b, "--min-tokens", "20",
                      "--loop-period-max", "4", "--loop-repeat-min", "8"])
        assert r.returncode == 1 and "CANARY FAIL" in r.stderr + r.stdout
        # healthy longer candidate passes canary
        open(b, "w").write(" ".join(str((i * 13) % 977) for i in range(60)))
        r = _run_cli(["canary", a, b, "--min-tokens", "20",
                      "--loop-period-max", "4", "--loop-repeat-min", "8"])
        assert r.returncode == 0, r.stderr


def test_kl_cli_band(tmp="unused"):
    import numpy as np
    with tempfile.TemporaryDirectory() as td:
        rd = os.path.join(td, "ref"); cd = os.path.join(td, "cand")
        os.makedirs(rd); os.makedirs(cd)
        rng = np.random.default_rng(7)
        for i in range(5):
            base = rng.normal(0, 1, 128).astype(np.float32)
            base.tofile(os.path.join(rd, f"s{i:03d}.f32"))
            (base + rng.normal(0, 0.01, 128)).astype(np.float32).tofile(
                os.path.join(cd, f"s{i:03d}.f32"))
        tight = _run_cli(["logits", "--ref-dir", rd, "--cand-dir", cd,
                          "--kl-band", "0.000001"])
        assert tight.returncode == 1, "tiny band should fail on real noise"
        loose = _run_cli(["logits", "--ref-dir", rd, "--cand-dir", cd,
                          "--kl-band", "1000.0", "--kl-max", "1000.0"])
        assert loose.returncode == 0, loose.stderr
        # identical arrays must give KL == 0 under any band
        idd = os.path.join(td, "id"); os.makedirs(idd)
        for i in range(5):
            v = rng.normal(0, 1, 128).astype(np.float32)
            v.tofile(os.path.join(rd, f"x{i}.f32"))
            v.tofile(os.path.join(idd, f"x{i}.f32"))
        zero = _run_cli(["logits", "--ref-dir", rd, "--cand-dir", idd,
                         "--top-k", "32"])
        assert "KL top-32 mean     : 0.00000" in zero.stdout


def main():
    test_compare_tokens()
    test_detect_loop()
    test_kl_topk()
    test_cli_exit_codes()
    test_kl_cli_band()
    print("ALL DRIFT-GATE TESTS PASSED")


if __name__ == "__main__":
    main()

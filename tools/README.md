# Tools

CMake target `CNE_BUILD_TOOLS=ON` (default) builds everything below.

## Shipped (operators & CI)

| Binary | Purpose |
|---|---|
| `cne_prepare` | GGUF alignment / prepare pass |
| `cne_bench` | End-to-end throughput + identity checks |
| `cne_identity_gate` | Token-exact greedy parity harness |

## Operator scripts (`scripts/`)

Shell helpers — run from the repo root:

| Script | Purpose |
|---|---|
| `scripts/download-qwen3.6-35b-a3b-q4_k_xl.sh` | Fetch + verify + prepare Qwen artifact |
| `scripts/download-lfm2-24b-a2b.sh` | Fetch + verify + prepare LFM2-24B artifact |
| `scripts/download-lfm2.5-8b-a1b.sh` | Fetch + verify + prepare LFM2.5-8B-A1B artifact |
| `scripts/run-canary.sh` | PL-T1 long-gen canary (lossless vs lossy + drift gate) |
| `scripts/run-ppl.sh` | Perplexity over PTB corpus via `build-meas` llama-perplexity |

Example:

```sh
./tools/scripts/download-lfm2-24b-a2b.sh
./tools/scripts/run-canary.sh 7 2048 2304
```

## Quality harness

`drift_gate.py` — token/logit drift checks for lossy-vs-lossless comparisons.
Used by `scripts/run-canary.sh` and `tests/perplexity/test_drift_gate.py`.

## Development probes

Bisect and engine-development utilities — not required for serving:

| Binary | Purpose |
|---|---|
| `cne_graph_probe` | Graph node inspection |
| `cne_graph_census` | Decode-step op census + per-op wall time (kernel analysis) |
| `cne_rebind_probe` | Window rebind behavior |
| `cne_cb_semantics_probe` | Callback / fill semantics |
| `cne_touch_recorder` | Tensor touch recording |
| `cne_lru_sim` | Cache LRU simulation (standalone) |

Velocity harnesses live under [`bench/`](../bench/README.md), not here.

# Tools

CMake target `CNE_BUILD_TOOLS=ON` (default) builds everything below.

## Shipped (operators & CI)

| Binary | Purpose |
|---|---|
| `cne_prepare` | GGUF alignment / prepare pass |
| `cne_bench` | End-to-end throughput + identity checks |
| `cne_identity_gate` | Token-exact greedy parity harness |

Download helpers (shell, not CMake):

| Script | Purpose |
|---|---|
| `download-qwen3.6-35b-a3b-q4_k_xl.sh` | Fetch + verify + prepare Qwen artifact |
| `download-lfm2-24b-a2b.sh` | Fetch + verify + prepare LFM2 artifact |

## Development probes

Bisect and engine-development utilities — not required for serving:

| Binary | Purpose |
|---|---|
| `cne_graph_probe` | Graph node inspection |
| `cne_rebind_probe` | Window rebind behavior |
| `cne_cb_semantics_probe` | Callback / fill semantics |
| `cne_touch_recorder` | Tensor touch recording |
| `cne_lru_sim` | Cache LRU simulation (standalone) |

Velocity harnesses live under [`bench/`](../bench/README.md), not here.

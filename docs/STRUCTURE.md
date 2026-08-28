# Repository structure

Where code lives and what belongs where. Use this when adding files or
deciding whether something is product logic vs upstream kernel work.

## Top level

```
core/                 Engine library (cne_core) — GGUF, memory, streaming features
runtime/              llama.cpp seam + demand-serving runtime (cne_runtime)
server/               OpenAI-compatible HTTP/SSE server (cne_server)
cli/                  First-run setup (cne_setup)
tools/                CMake-built binaries + operator scripts
bench/                Velocity harnesses (scripts + local results)
tests/                Unit and integration tests
docs/                 Public documentation
models/               Local GGUF artifacts (gitignored)
third_party/          Pinned dependencies (llama.cpp fork)
internal-docs/        Private research / ops notes (separate git repo, gitignored)
```

## core/

Public API: `core/include/cne/`. Implementation under `core/src/`:

| Subdir | Responsibility |
|---|---|
| `gguf/` | GGUF reader, tensor classification, manifest registry |
| `memory/` | Memory budgets, regime classification |
| `features/streaming/` | Slice cache, O_DIRECT I/O, lane scheduler |

**Rule:** no llama.cpp includes in `core/`. No product logic in
`third_party/llama.cpp` beyond ggml-cpu compute hooks. Custom CPU kernels
live under `third_party/llama.cpp/ggml/src/ggml-cpu/cne/` (see `cne/README.md`).

## runtime/

Integration layer between `cne_core` and pinned llama.cpp:

| File / dir | Responsibility |
|---|---|
| `seam/` | Thin llama backend/version probe (`cne_adapter.*`) |
| `cne_runtime.cpp` | Shared boot sequence (bench + server) |
| `cne_stream_cb.cpp` | Demand-serving runtime (fills, cache, anon dense) |
| `cne_stream_spec.cpp` | Draft-MTP generation loop |

CMake target: `cne_runtime` (static library).

## tools/

| Kind | Location |
|---|---|
| Shipped binaries | Built by CMake: `cne_prepare`, `cne_bench`, `cne_identity_gate` |
| Dev probes | `cne_graph_probe`, `cne_graph_census`, `cne_rebind_probe`, … |
| Operator scripts | `tools/scripts/` — model download, canary, perplexity helpers |
| Quality harness | `drift_gate.py` (used by canary + `tests/perplexity/`) |

Velocity microbenches live under `bench/`, not `tools/`.

## tests/

| Subdir | What it tests |
|---|---|
| `gguf/` | GGUF reader / manifest |
| `memory/` | Budget and regime logic |
| `features/streaming/` | Slice cache and I/O |
| `boot/` | Runtime boot sequence (synthetic + optional live model) |
| `perplexity/` | PL drift-gate unit tests (`drift_gate.py`) |

## bench/

Scripts under `bench/scripts/`; results under `bench/results/` (gitignored).

## docs/

| File | Audience |
|---|---|
| `FEATURES.md` | Feature guide for operators |
| `SETUP.md` | Build and model fetch |
| `BENCHMARKS.md` | Reproducible measurement protocol |
| `models/` | Per-model serving notes |
| `STRUCTURE.md` | This file |

## internal-docs/

Separate git repository (listed in `.gitignore`). Organized by topic:

| Subdir | Contents |
|---|---|
| `kernels/` | CPU kernel playbook, velocity research |
| `chronicles/` | Postmortems, PL bench notes |
| `plans/` | Roadmaps and pivot docs |
| `ops/` | Server config, status, one-off issue notes |

## What not to move

- `third_party/llama.cpp` layout — submodule discipline
- `core/` module boundaries — already trimmed
- `server/cne-server.cpp` — single-file server is intentional

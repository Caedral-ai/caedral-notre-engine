# Contributing to Caedral Notre Engine

Rules and conventions for this repository. They exist to protect the two things
this project cannot compromise on: **correctness** (streaming must be
byte-identical to non-streaming) and **predictable memory** (explicit budgets,
never page-cache luck).

---

## 1. Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <imperative summary, max 72 chars>

[optional body: wrap at ~100 cols, explain WHY not WHAT]

[optional footer: BREAKING CHANGE:, Refs:, Gate:]
```

**Types**

| Type | Use for |
|---|---|
| `feat` | new functionality (cache, io, overlap…) |
| `fix` | bug fixes |
| `perf` | performance changes (must include benchmark numbers in body) |
| `refactor` | no behavior change |
| `test` | tests only |
| `docs` | documentation |
| `build` | build system, dependencies, submodules |
| `ci` | CI pipelines |
| `chore` | tooling, housekeeping |

**Scopes**: `core`, `gguf`, `moe`, `cache`, `io`, `memory`, `runtime`,
`metrics`, `server`, `cli`, `adapter`, `tests`.

Examples:

```
feat(cache): inflight dedupe with shared futures
perf(io): raise lane count default to 4 on NVMe (+18% tok/s, see bench/2026-08-21)
fix(gguf): validate expert span bounds before O_DIRECT read
```

**Rules**

- One logical change per commit. **Never mix** a refactor with a behavior change.
- Perf and its benchmark go together; correctness fixes come *before* perf work.
- Never rewrite published history on `main`.
- Reference the motivating design discussion when implementing a significant
  architectural decision.

## 2. Branches

```
feat/<topic>   fix/<topic>   bench/<topic>   docs/<topic>
```

Keep branches short-lived; rebase onto `main` before merge. Squash-merge WIP.

## 3. Code style

C++17. Formatting is enforced by the checked-in [`.clang-format`](.clang-format)
(4-space indent, 120-col limit, LLVM-based — close to llama.cpp).

- Run `clang-format -i <files>` before committing; formatting-only changes are
  their own commit (`style: ...`).
- **Naming**
  - namespaces: `cne` (all product code)
  - types/classes: `PascalCase` (`ExpertKey`, `ModelManifest`)
  - functions/methods: `snake_case`
  - private members: `trailing_underscore_`
  - constants: `kPascalCase` (`kMaxLanes`)
- Headers: `#pragma once`; public API lives in `core/include/cne/` only.
- No exceptions across module boundaries; return error codes / expected-like
  results at seams.
- Comments explain *why*; English only. Design docs may be PT-BR.
- No dead code, no commented-out blocks, no magic numbers tied to model layout.

## 4. Engineering invariants (non-negotiable)

These are non-negotiable — violating any of them blocks a merge:

1. **Metadata-driven**: never hardcode tensor names, axes, quant types or
   offsets.
2. **Fail closed**: incomplete expert discovery with `stream=1` aborts or falls
   back to mmap; never dereference nulls or continue silently.
3. **Lossless default**: nothing may alter the model's math silently; lossy
   modes are explicit opt-in flags.
4. **Memory budget invariant**: `engine_budget <= RAM_budget`; hard caps are
   never exceeded.
5. **Fork discipline**: the llama.cpp fork carries only the minimal
   expert-ready hook — zero product logic inside it.
6. **Separation of planes**: cache/policy/scheduler stay out of llama.cpp;
   HTTP stays out of the core runtime.
7. **No secrets, no weights**: never commit `.gguf`, API keys or local paths.

## 5. Correctness gates

No optimization enters the release without passing its gate.
Minimum bar for every PR:

| Gate | Criterion |
|---|---|
| Build | clean, no new warnings (`-Wall -Wextra`) |
| Stream identity | `stream ON == OFF`, token-exact |
| GGUF layout | 0 errors, all spans valid |
| Memory cap | steady-state RSS ≤ hard cap + tolerance |
| Upstream compat | seam tests pass against pinned llama.cpp |

Perf PRs additionally require paired A/B numbers (tok/s, p95 inter-token,
flash/token, hit-rate) from the benchmark protocol.

## 6. Tests

- Every feature lands with tests under the matching `tests/<area>/` directory.
- Bug fixes land with a regression test that fails without the fix.
- CI must run: build matrix, gate suite, stress (leak/RSS stability).

## 7. Dependencies

- `third_party/llama.cpp` is pinned; bump via dedicated `build(llama.cpp): pin
  vX.Y.Z` commit including upstream-compat gate results.
- **CNE llama fork:** kernel work lands on
  [trycaedral/llama.cpp](https://github.com/trycaedral/llama.cpp) branch
  `cne/lfm2-b3` (submodule URL in `.gitmodules`). Upstream sync remote:
  `upstream` → `ggml-org/llama.cpp`. Product logic stays in `core/` and
  `adapters/` — never in the fork beyond ggml-cpu compute hooks.
  LFM2 MoE hooks live in `ggml-cpu/repack.cpp` (`CNE_MOE_B3` toggle).
  x86 `4vx`/`2vx` are generic today — native AVX-VNNI fusion is the next
  kernel milestone (see `docs/models/lfm2-24b-a2b.md` § Kernel roadmap).
- Prefer vendored/minimal deps; no heavyweight frameworks in core.

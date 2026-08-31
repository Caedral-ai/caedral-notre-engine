# Changelog

Notable changes to Notre Engine. The project is **PRE-ALPHA**; versions below
are documentation snapshots, not production releases.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Documentation

- Position Notre Engine as a PRE-ALPHA local CPU MoE engine in the Notre
  research program. Caedral is a unified AI API with subscription plans,
  included usage pools, and optional on-demand overage — not “prepaid-only
  automation agency” infrastructure.
- Document current limitations: one request at a time, no engine-level tenant
  isolation, no multi-user batching, MTP vs `conversation_id` mutual exclusion,
  mixed-precision not default.
- Add `SECURITY.md` (responsible disclosure via security@caedral.com /
  [caedral.com/security](https://caedral.com/security); no fake certifications)
  and `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- Note that a default `cmake -B build` requires the `third_party/llama.cpp`
  submodule (`git submodule update --init`). Model downloads are optional and
  large; they are not required to compile. Core-only configure without llama
  succeeds; default configure without the submodule does not.

## [0.0.1] — PRE-ALPHA

Initial public tree: regime classification, expert streaming, draft-MTP
(stateless), OpenAI-compatible `cne_server`, optional `cne_gateway`, setup CLI,
and identity gates. See [README.md](README.md) and [docs/FEATURES.md](docs/FEATURES.md).

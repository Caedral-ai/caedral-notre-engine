# Security policy

Notre Engine is **PRE-ALPHA** local inference software. This document is a
responsible-disclosure policy, not a certification, audit report, or
compliance attestation. Caedral does not claim SOC 2, ISO 27001, HIPAA, PCI
DSS, or FedRAMP for this repository.

## Supported versions

Only the default branch of
[Caedral-ai/caedral-notre-engine](https://github.com/Caedral-ai/caedral-notre-engine)
is considered for security fixes. Pre-alpha snapshots have no SLA.

## What this engine is (threat model in one paragraph)

`cne_server` is a **single-user / single-decode-slot** process. It does not
provide tenant isolation, multi-tenant batching, or a hardened multi-user
control plane. The optional `cne_gateway` adds API keys and rate limits in
front of a localhost engine; that is an access proxy, not a security
boundary equivalent to the hosted [Caedral API](https://caedral.com).

Do not bind `cne_server` to a public interface. Do not treat local API keys
in `models/api_keys.txt` or `gateway/api_keys.local.txt` as production
secrets management.

## Reporting a vulnerability

Report Notre Engine issues the same way as other Caedral security reports:

1. Email **security@caedral.com** (preferred for sensitive reports).
2. Read the public policy: [caedral.com/security](https://caedral.com/security).

Please include:

- A description of the issue and the affected component (`cne_server`,
  `cne_gateway`, runtime, GGUF loader, …)
- Steps to reproduce, or a minimal proof of concept
- Impact (for example: unauthenticated access, path traversal, key leak)

**Do not** open a public GitHub issue with exploit details, model weights,
or key material.

### Safe harbor

If you make a good-faith effort to follow
[caedral.com/security](https://caedral.com/security) — including reasonable
time to investigate before public disclosure — Caedral will not initiate
legal action against you for research conducted in accordance with that
policy. Do not access other people’s data, disrupt production Caedral
services, or socially engineer Caedral personnel or users.

### Out of scope for this repo

- Denial-of-service against **production** Caedral infrastructure
  (`api.caedral.com` and related hosted systems)
- Physical security or social engineering
- Issues solely in upstream [llama.cpp](https://github.com/ggml-org/llama.cpp)
  that are not introduced by this project’s seam or fork hooks — report
  those upstream when they are not Caedral-specific
- Missing certifications or “please add SOC 2 badges”

## Secrets and model artifacts

- Never commit `.gguf` weights, `models/api_keys.txt`,
  `gateway/api_keys.local.txt`, or `gateway/gateway.json`.
- Example key files (`*.example.txt`) are placeholders only.
- Rotate any key that may have been committed or logged.

## Hosted Caedral API

Security for the unified Caedral API (accounts, billing, tenant isolation)
is documented at [caedral.com/security](https://caedral.com/security). This
engine is not that service.

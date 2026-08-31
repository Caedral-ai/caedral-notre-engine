#!/usr/bin/env bash
# Run the API-key gateway (Path B). Requires cne_server in API mode on CNE_UPSTREAM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/gateway"

if [[ ! -d .venv ]]; then
  python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
# shellcheck disable=SC1091
source .venv/bin/activate

if [[ -z "${CNE_GATEWAY_CONFIG:-}" ]]; then
  if [[ -f "$ROOT/gateway/gateway.json" ]]; then
    export CNE_GATEWAY_CONFIG="$ROOT/gateway/gateway.json"
  fi
fi

exec python -m cne_gateway

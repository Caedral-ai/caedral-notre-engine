#!/usr/bin/env bash
# Run the JWT API gateway (Path B). Requires cne_server in API mode on CNE_UPSTREAM.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/gateway"

if [[ ! -d .venv ]]; then
  python -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
# shellcheck disable=SC1091
source .venv/bin/activate

: "${CNE_GATEWAY_JWT_SECRET:?set CNE_GATEWAY_JWT_SECRET}"
: "${CNE_INTERNAL_API_KEY:?set CNE_INTERNAL_API_KEY (same as cne_server)}"

export CNE_GATEWAY_USERS_FILE="${CNE_GATEWAY_USERS_FILE:-$ROOT/gateway/users.example.txt}"
export CNE_UPSTREAM="${CNE_UPSTREAM:-http://127.0.0.1:8080}"
export CNE_GATEWAY_HOST="${CNE_GATEWAY_HOST:-127.0.0.1}"
export CNE_GATEWAY_PORT="${CNE_GATEWAY_PORT:-8090}"

exec python -m cne_gateway

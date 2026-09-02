#!/bin/sh
# Verify model path is readable (check mode), then exec cne_server with an
# absolute model path derived from server.json.
set -eu

resolve_model() {
  cfg="$1"
  cfg_dir=$(dirname "$cfg")
  model=$(sed -n 's/.*"model"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$cfg" | head -n1)
  case "$model" in
    "") return 1 ;;
    /*) printf '%s' "$model" ;;
    *) printf '%s' "$cfg_dir/$model" ;;
  esac
}

if [ "${1:-}" = "check" ]; then
  MODEL="${2:-}"
  if [ -z "$MODEL" ]; then
    echo "[entrypoint] usage: check <model-path>" >&2
    exit 1
  fi
  if [ ! -f "$MODEL" ]; then
    echo "[entrypoint] model not found: $MODEL" >&2
    echo "[entrypoint] /models listing:" >&2
    ls -laR /models >&2 || true
    exit 1
  fi
  echo "[entrypoint] model ok: $MODEL"
  exit 0
fi

CONFIG="${1:-/models/server.json}"
cd "$(dirname "$CONFIG")" 2>/dev/null || cd /models

MODEL=""
if [ -f "$CONFIG" ]; then
  MODEL=$(resolve_model "$CONFIG" || true)
fi

if [ -n "$MODEL" ] && [ ! -f "$MODEL" ]; then
  echo "[entrypoint] model not found: $MODEL (from $CONFIG)" >&2
  ls -laR "$(dirname "$CONFIG")" >&2 || true
  exit 1
fi

if [ -n "$MODEL" ]; then
  echo "[entrypoint] starting cne_server model=$MODEL" >&2
  exec cne_server --config "$CONFIG" "$MODEL" 0.0.0.0 8080
fi

echo "[entrypoint] starting cne_server (no model key in $CONFIG)" >&2
exec cne_server --config "$CONFIG" 0.0.0.0 8080

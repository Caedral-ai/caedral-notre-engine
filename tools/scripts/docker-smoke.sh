#!/usr/bin/env bash
# Smoke-test the engine Docker image using an existing local model (no download).
#
# Usage:
#   ./tools/scripts/docker-smoke.sh
#   ./tools/scripts/docker-smoke.sh --chat
#   ./tools/scripts/docker-smoke.sh --skip-build --keep

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CHAT=0
KEEP=0
SKIP_BUILD=0
STAGE=0
VERBOSE=0
for arg in "$@"; do
  case "$arg" in
    --chat) CHAT=1 ;;
    --keep) KEEP=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --stage) STAGE=1 ;;
    --verbose|-v) VERBOSE=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg (try --help)" >&2; exit 2 ;;
  esac
done

if ! command -v docker >/dev/null 2>&1; then
  echo "[smoke] docker not found in PATH" >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "[smoke] cannot access Docker (add user to group docker, then newgrp docker)" >&2
  exit 1
fi

SRC="${CNE_SMOKE_SRC_MODELS:-$ROOT/models}"
PORT="${CNE_PORT:-8080}"
OVERRIDE="$ROOT/.docker-test/compose.override.yml"
COMPOSE=(docker compose -f docker-compose.yml -f "$OVERRIDE")

if [[ ! -d "$SRC" ]]; then
  echo "[smoke] source models dir missing: $SRC" >&2
  exit 1
fi

mapfile -t PREPARED < <(find "$SRC" -name '*-prepared.gguf' -type f 2>/dev/null | sort)
if [[ ${#PREPARED[@]} -eq 0 ]]; then
  echo "[smoke] no *-prepared.gguf under $SRC" >&2
  exit 1
fi

PICK="${PREPARED[0]}"
for f in "${PREPARED[@]}"; do
  if [[ "$f" == *lfm2.5* ]] || [[ "$f" == *lfm25* ]]; then
    PICK="$f"
    break
  fi
done

REL="${PICK#"$SRC/"}"
REL="${REL#/}"

if [[ "$STAGE" -eq 1 ]]; then
  DST="$ROOT/.docker-test/models"
  echo "[smoke] staging copy (not hardlink) to $DST"
  rm -rf "$DST"
  mkdir -p "$DST/$(dirname "$REL")"
  cp -a "$PICK" "$DST/$REL"
  [[ -f "$SRC/server.json" ]] && cp "$SRC/server.json" "$DST/server.json"
  [[ -f "$SRC/api_keys.txt" ]] && cp "$SRC/api_keys.txt" "$DST/api_keys.txt"
else
  DST="$SRC"
  echo "[smoke] mounting source models dir directly: $DST"
fi

HOST_MODEL="$DST/$REL"
if [[ ! -f "$HOST_MODEL" ]]; then
  echo "[smoke] model file missing on host: $HOST_MODEL" >&2
  exit 1
fi

mkdir -p "$(dirname "$OVERRIDE")"
cat >"$OVERRIDE" <<EOF
services:
  cne_server:
    restart: "no"
    working_dir: /models
    volumes:
      - ${DST}:/models:ro
    ports:
      - "127.0.0.1:${PORT}:8080"
EOF

echo "[smoke] artifact: $PICK"
echo "[smoke] container path: /models/$REL"

URL="http://127.0.0.1:${PORT}/health"
# http_proxy in non-interactive shells can break localhost curl.
CURL=(curl -fsS --noproxy '*')

wait_for_health() {
  local wait_start=$SECONDS
  local max_wait=300
  local log_pid=""

  stop_logs() {
    [[ -n "$log_pid" ]] && kill "$log_pid" 2>/dev/null || true
    wait "$log_pid" 2>/dev/null || true
    log_pid=""
  }

  if [[ "$VERBOSE" -eq 1 ]]; then
    "${COMPOSE[@]}" logs -f cne_server >&2 &
    log_pid=$!
  fi

  while true; do
    if "${CURL[@]}" "$URL" >/dev/null 2>&1; then
      stop_logs
      echo $((SECONDS - wait_start))
      return 0
    fi

    local state
    state="$("${COMPOSE[@]}" ps -a --format '{{.State}}' cne_server 2>/dev/null | head -1 || true)"
    case "$state" in
      exited|dead)
        stop_logs
        echo "[smoke] container $state before /health responded" >&2
        "${COMPOSE[@]}" logs --tail=120 cne_server >&2 || true
        return 1
        ;;
      restarting)
        stop_logs
        echo "[smoke] container is restarting (likely crash loop)" >&2
        "${COMPOSE[@]}" logs --tail=120 cne_server >&2 || true
        return 1
        ;;
    esac

    local elapsed=$((SECONDS - wait_start))
    if (( elapsed >= max_wait )); then
      stop_logs
      echo "[smoke] timeout after ${elapsed}s" >&2
      echo "[smoke] last curl error:" >&2
      curl -v --noproxy '*' "$URL" >&2 || true
      "${COMPOSE[@]}" logs --tail=40 cne_server >&2 || true
      return 1
    fi

    printf '\r[smoke] waiting for /health (%ds)...' "$elapsed" >&2
    sleep 1
  done
}

print_health() {
  local elapsed="${1:-0}"
  printf '\r%-60s\r' '' >&2
  echo "" >&2
  echo "============================================" >&2
  echo "[smoke] health OK (${elapsed}s)" >&2
  echo "============================================" >&2
  local health_json
  health_json="$("${CURL[@]}" "$URL")"
  if command -v jq >/dev/null 2>&1; then
    echo "$health_json" | jq .
  else
    echo "$health_json" | python3 -m json.tool 2>/dev/null || echo "$health_json"
  fi
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[smoke] building image..."
  "${COMPOSE[@]}" build cne_server
else
  echo "[smoke] skipping build (--skip-build)"
fi

echo "[smoke] verifying mount inside image..."
if ! "${COMPOSE[@]}" run --rm --no-deps --entrypoint /docker-entrypoint.sh \
    cne_server check "/models/$REL"; then
  echo "[smoke] model not visible inside container — check Docker volume mount" >&2
  exit 1
fi

cleanup() {
  if [[ "$KEEP" -eq 1 ]]; then
    echo "[smoke] --keep: container left running" >&2
    return
  fi
  "${COMPOSE[@]}" down >/dev/null 2>&1 || true
}

# Already up? Skip tear-down and boot wait.
if "${CURL[@]}" "$URL" >/dev/null 2>&1; then
  echo "[smoke] engine already responding on $URL"
  trap cleanup EXIT
  print_health 0
else
  echo "[smoke] starting container..."
  "${COMPOSE[@]}" down --remove-orphans >/dev/null 2>&1 || true
  if ! "${COMPOSE[@]}" up -d --force-recreate cne_server; then
    echo "[smoke] docker compose up failed" >&2
    "${COMPOSE[@]}" logs --tail=80 cne_server >&2 || true
    exit 1
  fi

  trap cleanup EXIT

  mapped="$("${COMPOSE[@]}" port cne_server 8080 2>/dev/null || true)"
  if [[ -z "$mapped" ]]; then
    echo "[smoke] port ${PORT} not published" >&2
    "${COMPOSE[@]}" logs --tail=80 cne_server >&2 || true
    exit 1
  fi
  echo "[smoke] published ${mapped}"

  boot_secs="$(wait_for_health)" || exit 1
  print_health "$boot_secs"
fi

if [[ "$CHAT" -eq 1 ]]; then
  echo "[smoke] chat completion..." >&2
  chat_json="$("${CURL[@]}" "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: docker ok"}],"max_tokens":16,"stream":false,"chat_template_kwargs":{"enable_thinking":false}}')"
  if command -v jq >/dev/null 2>&1; then
    echo "$chat_json" | jq .
  else
    echo "$chat_json" | python3 -m json.tool 2>/dev/null || echo "$chat_json"
  fi
fi

echo "[smoke] done." >&2

#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT=8051
PORT="${1:-${MCP_SERVER_PORT:-$DEFAULT_PORT}}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="*"
export no_proxy="*"

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi
if [[ ! -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  echo "MCP environment is missing; run uv sync in $PROJECT_DIR" >&2
  exit 1
fi
if [[ ! -x "${NESSO_LOCAL_PYTHON:-$PROJECT_DIR/.nesso_venv/bin/python}" ]]; then
  echo "Local Nesso environment is missing; run ./setup_local_nesso.sh" >&2
  exit 1
fi

export MCP_SERVER_PORT="$PORT"
export NESSO_LOCAL_RUNS_ROOT="${NESSO_LOCAL_RUNS_ROOT:-$PROJECT_DIR/nesso_runs}"
export NESSO_LOCAL_PYTHON="${NESSO_LOCAL_PYTHON:-$PROJECT_DIR/.nesso_venv/bin/python}"
export NESSO_CACHE_DIR="${NESSO_CACHE_DIR:-$PROJECT_DIR/nesso_cache}"
export NESSO_MCP_BATCH_SIZE="${NESSO_MCP_BATCH_SIZE:-32}"
export NESSO_DATALOADER_BATCH_SIZE="${NESSO_DATALOADER_BATCH_SIZE:-32}"
export NESSO_PREPROCESS_WORKERS="${NESSO_PREPROCESS_WORKERS:-4}"
export NESSO_PREPROCESS_START_TIMEOUT_SECONDS="${NESSO_PREPROCESS_START_TIMEOUT_SECONDS:-600}"

cd "$PROJECT_DIR"
if [[ -f "$PROJECT_DIR/nesso_local_mcp.pid" ]]; then
  EXISTING_PID="$(<"$PROJECT_DIR/nesso_local_mcp.pid")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "Local Nesso MCP server is already running: pid=$EXISTING_PID" >&2
    exit 1
  fi
fi
setsid "$PROJECT_DIR/.venv/bin/python" \
  "$PROJECT_DIR/mcp_server_nesso_cofolding_local.py" \
  > "$PROJECT_DIR/nesso_local_mcp.log" 2>&1 < /dev/null &
echo "$!" > "$PROJECT_DIR/nesso_local_mcp.pid"
SERVER_PID="$!"
READY=0
for _attempt in {1..60}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Local Nesso MCP server exited during startup:" >&2
    tail -50 "$PROJECT_DIR/nesso_local_mcp.log" >&2
    exit 1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.5
done
if (( READY == 0 )); then
  echo "Local Nesso MCP server did not listen on port $PORT within 30 seconds." >&2
  exit 1
fi
echo "Local Nesso MCP server started: pid=$SERVER_PID port=$PORT"
echo "Resident GPU workers start lazily when batches are assigned to them."

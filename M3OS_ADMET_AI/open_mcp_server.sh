#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT=8000
PORT="${1:-${MCP_SERVER_PORT:-$DEFAULT_PORT}}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export M3OS_REINVENT_ROOT="${M3OS_REINVENT_ROOT:-$SCRIPT_DIR/../M3OS_REINVENT}"
export MCP_SERVER_PORT="$PORT"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

PID_FILE="$SCRIPT_DIR/admet_ai_mcp.pid"
LOG_FILE="$SCRIPT_DIR/admet_ai_mcp.log"
if [[ -f "$PID_FILE" ]]; then
  EXISTING_PID="$(<"$PID_FILE")"
  if [[ "$EXISTING_PID" =~ ^[0-9]+$ ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "ADMET-AI MCP server is already running: pid=$EXISTING_PID" >&2
    exit 1
  fi
fi

cd "$SCRIPT_DIR"
echo "Starting ADMET-AI MCP server on port $MCP_SERVER_PORT"
setsid "$SCRIPT_DIR/.venv/bin/python" \
  "$SCRIPT_DIR/mcp_server_improved_admet_ai.py" \
  > "$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID="$!"
printf '%s\n' "$SERVER_PID" > "$PID_FILE"

READY=0
for _attempt in {1..1200}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ADMET-AI MCP server exited during startup:" >&2
    tail -100 "$LOG_FILE" >&2
    exit 1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.5
done
if (( READY == 0 )); then
  echo "ADMET-AI MCP server did not listen on port $PORT within 600 seconds." >&2
  exit 1
fi
echo "ADMET-AI MCP server started: pid=$SERVER_PID port=$PORT"

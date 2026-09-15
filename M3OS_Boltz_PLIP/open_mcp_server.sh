#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT=8040
PORT="${1:-${MCP_SERVER_PORT:-$DEFAULT_PORT}}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MCP_SERVER_PORT="$PORT"

cd "$SCRIPT_DIR"
echo "Starting Boltz/PLIP MCP server on port $MCP_SERVER_PORT"
setsid "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/mcp_server_improved.py" > "$SCRIPT_DIR/boltz_mcp.log" 2>&1 < /dev/null &
echo "$!" > "$SCRIPT_DIR/boltz_mcp.pid"
echo "PID: $!"

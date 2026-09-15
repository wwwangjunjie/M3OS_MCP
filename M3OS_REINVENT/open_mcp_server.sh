#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT=8041
PORT="${1:-${MCP_SERVER_PORT:-$DEFAULT_PORT}}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export MCP_SERVER_PORT="$PORT"

if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  echo "MCP environment is missing; run uv sync --frozen in $SCRIPT_DIR" >&2
  exit 1
fi
if [[ ! -x "${REINVENT_PYTHON:-$SCRIPT_DIR/REINVENT4/.venv/bin/python}" ]]; then
  echo "REINVENT4 is not installed; follow README.md or set REINVENT_PYTHON" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
echo "Starting REINVENT MCP server on port $MCP_SERVER_PORT"
nohup "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/mcp_server_improved.py" > "$SCRIPT_DIR/reinvent_mcp.log" 2>&1 &
echo $! > "$SCRIPT_DIR/reinvent_mcp.pid"

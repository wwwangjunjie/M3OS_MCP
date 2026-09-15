#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PORT=8020
PORT="${1:-${MCP_SERVER_PORT:-$DEFAULT_PORT}}"
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export MCP_SERVER_PORT="$PORT"

cd "$SCRIPT_DIR"
echo "Starting IUPAC MCP server on port $MCP_SERVER_PORT"
nohup "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/mcp_server_improved_iupac_generate.py" > "$SCRIPT_DIR/iupac_mcp.log" 2>&1 &

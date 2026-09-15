#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-${MCP_SERVER_PORT:-8057}}"
PID_FILE="$SCRIPT_DIR/tamgen_mcp.pid"
LOG_FILE="$SCRIPT_DIR/tamgen_mcp.log"

if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Invalid port: $PORT" >&2
  exit 1
fi
if [[ ! -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
  echo "TamGen environment is missing; run ./setup_tamgen.sh" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_DIR/checkpoints/crossdock_pdb_A10/checkpoint_best.pt" ]]; then
  echo "TamGen checkpoint is missing" >&2
  exit 1
fi
if [[ ! -f "$SCRIPT_DIR/gpt_model/checkpoint_best.pt" ]]; then
  echo "TamGen GPT checkpoint is missing" >&2
  exit 1
fi

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export MCP_SERVER_PORT="$PORT"
export CUDA_VISIBLE_DEVICES="${TAMGEN_GPU_ID:-0}"
export TAMGEN_DEVICE="${TAMGEN_DEVICE:-cuda:0}"
export TAMGEN_PRELOAD="${TAMGEN_PRELOAD:-1}"
export TAMGEN_FP16="${TAMGEN_FP16:-0}"
export PYTHONPATH="$SCRIPT_DIR/source:$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "TamGen MCP server is already running (PID $(cat "$PID_FILE"))"
  exit 0
fi

cd "$SCRIPT_DIR"
setsid "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/mcp_server_tamgen.py" \
  >>"$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID=$!
echo "$SERVER_PID" >"$PID_FILE"

READY=0
for _attempt in {1..600}; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "TamGen MCP server exited during startup:" >&2
    tail -80 "$LOG_FILE" >&2
    exit 1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    READY=1
    break
  fi
  sleep 0.5
done
if (( READY == 0 )); then
  echo "TamGen MCP server did not listen on port $PORT within 300 seconds." >&2
  exit 1
fi
echo "TamGen MCP server started on port $PORT (PID $SERVER_PID)"


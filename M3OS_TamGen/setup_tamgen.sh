#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$SCRIPT_DIR/source"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY='*'
export no_proxy='*'
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required; set UV_BIN to its executable path" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_DIR/setup.py" ]]; then
  echo "TamGen source is missing at $SOURCE_DIR" >&2
  echo "Follow the Source and model setup section in README.md." >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  "$UV_BIN" venv --python 3.10 "$SCRIPT_DIR/.venv"
fi

"$UV_BIN" pip install --python "$PYTHON_BIN" \
  --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.3.0'

"$UV_BIN" pip install --python "$PYTHON_BIN" \
  --find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html \
  'torch-cluster==1.6.3'

"$UV_BIN" pip install --python "$PYTHON_BIN" \
  'biopython==1.84' \
  'einops>=0.8,<0.9' \
  'mcp==1.14.0' \
  'numpy==1.26.4' \
  'pandas==2.2.3' \
  'pyyaml==6.0.2' \
  'rdkit==2024.3.1' \
  'scipy==1.14.1' \
  'tensorboardx>=2.6,<3' \
  'torch-geometric==2.6.1' \
  'tqdm>=4.67,<5' \
  'pytest>=8,<9'

"$UV_BIN" pip install --python "$PYTHON_BIN" -e "${SOURCE_DIR}[chem]"

"$PYTHON_BIN" -c 'import torch, torch_cluster, rdkit, Bio, mcp, fairseq; print(torch.__version__, torch.version.cuda)'

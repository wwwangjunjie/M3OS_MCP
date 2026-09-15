#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$PROJECT_DIR/vendor/nesso"
NESSO_VENV="${NESSO_LOCAL_VENV:-$PROJECT_DIR/.nesso_venv}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [[ -z "$UV_BIN" ]]; then
  echo "uv is required; install it or set UV_BIN" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_DIR/pyproject.toml" ]]; then
  echo "Nesso source is missing: $SOURCE_DIR" >&2
  echo "Follow the Source and model setup section in README.md." >&2
  exit 1
fi

cd "$PROJECT_DIR"
"$UV_BIN" sync --frozen

if [[ ! -x "$NESSO_VENV/bin/python" ]]; then
  if [[ -n "${NESSO_BASE_PYTHON:-}" ]]; then
    if [[ ! -x "$NESSO_BASE_PYTHON" ]]; then
      echo "NESSO_BASE_PYTHON is not executable: $NESSO_BASE_PYTHON" >&2
      exit 1
    fi
    "$NESSO_BASE_PYTHON" -m venv --system-site-packages "$NESSO_VENV"
  else
    "$UV_BIN" venv --python "${NESSO_PYTHON_VERSION:-3.11}" "$NESSO_VENV"
  fi
fi

source_spec="$SOURCE_DIR"
if [[ "${NESSO_INSTALL_KERNELS:-0}" == "1" ]]; then
  source_spec="${SOURCE_DIR}[kernels]"
fi
"$UV_BIN" pip install --python "$NESSO_VENV/bin/python" -e "$source_spec"

"$NESSO_VENV/bin/python" - <<'PY'
import nesso
import torch

print("Nesso:", nesso.__file__)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

echo "Local Nesso environment is ready: $NESSO_VENV"
echo "Official model files are downloaded by Nesso on first use."

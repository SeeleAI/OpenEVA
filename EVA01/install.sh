#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; installing uv into the current user environment..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation did not put uv on PATH. Add ~/.local/bin to PATH and rerun." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-180}"
if [ -x "${VENV_DIR}/bin/python" ] && [ "${UV_VENV_CLEAR:-0}" != "1" ]; then
  echo "Using existing virtual environment at ${VENV_DIR}"
else
  uv venv "${VENV_DIR}" --python "${PYTHON_BIN}"
fi

TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
if [ -z "${TORCH_INDEX_URL}" ] && command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
fi

if [ -n "${TORCH_INDEX_URL}" ]; then
  uv pip install --python "${VENV_DIR}/bin/python" torch --index-url "${TORCH_INDEX_URL}"
fi

uv pip install --python "${VENV_DIR}/bin/python" -e "${ROOT_DIR}"

cat <<EOF
EVA01 environment is ready.

Activate it with:
  source ${VENV_DIR}/bin/activate

Try inference with:
  python ${ROOT_DIR}/infer.py --mesh ${ROOT_DIR}/assets/examples/<sample>.glb
EOF

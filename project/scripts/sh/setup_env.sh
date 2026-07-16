#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"

export PATH="${HOME}/.local/bin:${PATH}"
export UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"
export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.uv-python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv or add it to PATH." >&2
  exit 1
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_SPEC="${PYTHON_BIN}"
elif [[ -x /usr/bin/python3 ]] && [[ "$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.10" ]]; then
  PYTHON_SPEC="/usr/bin/python3"
else
  PYTHON_SPEC="3.10"
fi

uv venv --python "${PYTHON_SPEC}" --system-site-packages
uv sync --group dev "$@"

source .venv/bin/activate
python - <<'PY'
import sys

version = f"{sys.version_info.major}.{sys.version_info.minor}"
if version != "3.10":
    raise SystemExit(f"Expected Python 3.10 to match the remote GPU/HPC server, got {version}")

print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
except Exception as exc:
    print("torch MISSING", exc)

import h5py
print("h5py", h5py.__version__)
PY

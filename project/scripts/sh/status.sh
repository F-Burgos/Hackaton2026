#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT_DIR}"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

echo "== Git =="
git status --short --branch
git log --oneline --decorate -3

echo
echo "== Data =="
if [[ -d hackaton ]]; then
  find hackaton -maxdepth 3 -type f -printf '%s\t%P\n' | sort -nr | head -20
else
  echo "hackaton/ not found"
fi

echo
echo "== Python =="
python - <<'PY'
import importlib

core = [
    "h5py",
    "hydra",
    "matplotlib",
    "numpy",
    "omegaconf",
    "pandas",
    "pypdf",
    "sklearn",
    "torch",
    "yaml",
]
optional = [
    "umap",
]

print("Core dependencies:")
for name in core:
    try:
        module = importlib.import_module(name)
        print(f"{name}: OK {getattr(module, '__version__', '')}")
    except Exception as exc:
        print(f"{name}: MISSING ({exc})")

print("\nOptional dependencies:")
for name in optional:
    try:
        module = importlib.import_module(name)
        print(f"{name}: OK {getattr(module, '__version__', '')}")
    except Exception as exc:
        print(f"{name}: MISSING ({exc})")
PY

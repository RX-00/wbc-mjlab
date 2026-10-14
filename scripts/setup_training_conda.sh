#!/usr/bin/env bash
# Install this repo's locked CUDA dependencies into its own conda environment.
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_BASE=${CONDA_BASE:-"$HOME/miniconda3"}
ENV_PREFIX="$CONDA_BASE/envs/wbc-mjlab"
cd "$REPO"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_BASE/bin/conda" create -y -p "$ENV_PREFIX" python=3.12 pip
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

# Pylock retains the wheel URLs from uv.lock, including CUDA 12.8 PyTorch.
# Explicit --python installs into conda without creating a project .venv.
REQ_DIR=$(mktemp -d)
trap 'rm -rf -- "$REQ_DIR"' EXIT
uv export --quiet --frozen --extra cu128 --no-dev --no-emit-project \
  --format pylock.toml --output-file "$REQ_DIR/pylock.toml"
uv pip install --python "$ENV_PREFIX/bin/python" -r "$REQ_DIR/pylock.toml"
uv pip install --python "$ENV_PREFIX/bin/python" --no-deps -e .
uv pip check --python "$ENV_PREFIX/bin/python"
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable in the new environment'
x = torch.ones((64, 64), device='cuda')
y = x @ x
torch.cuda.synchronize()
print('GPU:', torch.cuda.get_device_name(0))
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
PY
printf 'Ready: conda activate %s\n' "$ENV_PREFIX"

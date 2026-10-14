#!/usr/bin/env bash
# One independent GPU policy per job. Resource/time overrides belong to sbatch.
#SBATCH --partition=debug
#SBATCH --exclude=a2r-main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=7-00:00:00
set -euo pipefail
TASK=${1:?task}
PHASE=${2:?phase}
ITERATIONS=${3:?iterations}
REPO=${4:?repository}
RUN_ROOT=${5:?output root}
IMAGE=${6:?SIF image}
: "${SLURM_JOB_ID:?Run through sbatch}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
[[ -n "$CUDA_VISIBLE_DEVICES" ]]
cd "$REPO"
sha256sum -c SNAPSHOT.sha256 > "$RUN_ROOT/console/snapshot_${SLURM_JOB_ID}.log"
(cd "$(dirname -- "$IMAGE")" && sha256sum -c IMAGE.sha256)
printf 'Job: %s Task: %s Phase: %s Node: %s CUDA_VISIBLE_DEVICES: %s\n' \
  "$SLURM_JOB_ID" "$TASK" "$PHASE" "$(hostname)" "$CUDA_VISIBLE_DEVICES"
nvidia-smi
# Persistent node-local JIT cache, isolated by user and image build.
CACHE_ROOT="/tmp/wbc-mjlab-$(id -u)-$(basename -- "$(dirname -- "$IMAGE")")"
mkdir -p "$CACHE_ROOT"
container=(apptainer exec --nv --cleanenv --no-home
  --bind "$REPO:/workspace/wbc-mjlab:ro" --bind "$RUN_ROOT:/runs:rw"
  --bind "$CACHE_ROOT:/cache:rw" --pwd /workspace/wbc-mjlab
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4
  --env XDG_CACHE_HOME=/cache --env WARP_CACHE_PATH=/cache/warp
  --env MPLCONFIGDIR=/cache/matplotlib
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1
  "$IMAGE")
# The batch step already has Slurm's CPU/RAM/GPU allocation. A2R workers do
# not install the client-side srun executable; use their existing job pattern.
"${container[@]}" python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable inside allocated container'
assert torch.cuda.device_count() == 1, 'Expected exactly one visible allocated GPU'
x = torch.ones((64, 64), device='cuda')
assert (x @ x).sum().item() == 64 ** 3
print('GPU preflight passed:', torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
"${container[@]}" python scripts/cluster/run_checked.py \
  --task "$TASK" --phase "$PHASE" --iterations "$ITERATIONS" \
  --job-id "$SLURM_JOB_ID"

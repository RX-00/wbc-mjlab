#!/usr/bin/env bash
# Allocated GPU checks; preserve a separate immutable snapshot for every run.
#SBATCH --partition=debug
#SBATCH --nodelist=a2r-wolf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx5090:1
#SBATCH --time=01:00:00
set -euo pipefail
REPO=${1:?snapshot}
RUN_ROOT=${2:?output root}
IMAGE=${3:?image}
shift 3
: "${SLURM_JOB_ID:?Run through sbatch}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must allocate a GPU}"
cd "$REPO"
sha256sum -c SNAPSHOT.sha256 > "$RUN_ROOT/snapshot_${SLURM_JOB_ID}.log"
(cd "$(dirname -- "$IMAGE")" && sha256sum -c IMAGE.sha256)
nvidia-smi
CACHE_ROOT="/tmp/wbc-mjlab-$(id -u)-$(basename -- "$(dirname -- "$IMAGE")")"
mkdir -p "$CACHE_ROOT"
apptainer exec --nv --cleanenv --no-home \
  --bind "$REPO:/workspace/wbc-mjlab:ro" --bind "$RUN_ROOT:/runs:rw" \
  --bind "$CACHE_ROOT:/cache:rw" --pwd /workspace/wbc-mjlab \
  --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --env OMP_NUM_THREADS=4 --env MKL_NUM_THREADS=4 \
  --env XDG_CACHE_HOME=/cache --env WARP_CACHE_PATH=/cache/warp \
  --env MPLCONFIGDIR=/cache/matplotlib --env "MUJOCO_GL=${WBC_MUJOCO_GL:-disable}" \
  --env PYTHONUNBUFFERED=1 --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONFAULTHANDLER=1 \
  "$IMAGE" python "$@"

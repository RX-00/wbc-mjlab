#!/usr/bin/env bash
# Run inside tmux. Smoke-test every task at the requested concurrency, then train.
# Overrides: PARALLEL=1 NUM_ENVS=2048 MAX_ITERS=200000 DATASET=samples
# SMOKE_ONLY=1 runs only the short tests. Requires Bash >= 5.1 (wait -n -p).
set -euo pipefail

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONDA_BASE=${CONDA_BASE:-"$HOME/miniconda3"}
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_BASE/envs/wbc-mjlab"
cd "$REPO"

PARALLEL=${PARALLEL:-2}
NUM_ENVS=${NUM_ENVS:-2048}
MAX_ITERS=${MAX_ITERS:-200000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
DATASET=${DATASET:-samples}
SEED=${SEED:-42}
SMOKE_ONLY=${SMOKE_ONLY:-0}
for value in "$PARALLEL" "$NUM_ENVS" "$MAX_ITERS" "$SAVE_INTERVAL"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Counts must be positive integers.' >&2; exit 1; }
done
(( PARALLEL <= 2 )) || { echo 'Use PARALLEL=1 or 2 on this single-GPU machine.' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

STAMP=$(date +%Y-%m-%d_%H-%M-%S)
RUN_ROOT=${RUN_ROOT:-"$REPO/logs/six_$STAMP"}
mkdir -p "$RUN_ROOT/console"
printf 'Logs: %s\n' "$RUN_ROOT"
if ! compgen -G "data/g1/$DATASET/npz/*.npz" >/dev/null; then
  wbc-mjlab-data-to-npz --robot g1 --dataset "$DATASET" --batch-size 8
fi

tasks=(Wbc-G1-Zest Wbc-G1-Zest-SE Wbc-G1 Wbc-G1-SE Wbc-G1-EE Wbc-G1-EE-SE)
pids=()
cleanup() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_phase() {
  local phase=$1 iterations=$2
  local start offset task log completed pid status
  local -a remaining
  for ((start=0; start<${#tasks[@]}; start+=PARALLEL)); do
    pids=()
    for ((offset=0; offset<PARALLEL && start+offset<${#tasks[@]}; offset++)); do
      task=${tasks[start+offset]}
      log="$RUN_ROOT/console/${phase}_${task}.log"
      printf 'Starting %s: %s (%s environments); log: %s\n' "$phase" "$task" "$NUM_ENVS" "$log"
      wbc-mjlab-train --task "$task" --dataset "$DATASET" \
        --env.scene.num-envs "$NUM_ENVS" \
        --agent.seed "$SEED" --agent.max-iterations "$iterations" \
        --agent.save-interval "$SAVE_INTERVAL" --agent.logger tensorboard \
        --agent.run-name "${phase}_seed${SEED}" \
        --log-root "$RUN_ROOT/$phase" >"$log" 2>&1 &
      pids+=("$!")
    done
    while ((${#pids[@]})); do
      if wait -n -p completed "${pids[@]}"; then
        remaining=()
        for pid in "${pids[@]}"; do
          [[ "$pid" == "$completed" ]] || remaining+=("$pid")
        done
        pids=("${remaining[@]}")
        printf 'Finished %s job, PID %s\n' "$phase" "$completed"
      else
        status=$?
        printf '%s failed (exit %s). Stopping; inspect %s/console/.\n' "$phase" "$status" "$RUN_ROOT" >&2
        return "$status"
      fi
    done
  done
}

# Same environment count and concurrency as full training, including PPO updates.
run_phase smoke 10
echo 'All six smoke tests passed.'
if [[ "$SMOKE_ONLY" != 1 ]]; then
  run_phase train "$MAX_ITERS"
  printf 'All six training runs finished. Checkpoints: %s/train/\n' "$RUN_ROOT"
fi

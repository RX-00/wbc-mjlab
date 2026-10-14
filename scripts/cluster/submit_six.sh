#!/usr/bin/env bash
# Submit six independent, fresh jobs; one policy failure never cancels another.
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=$(realpath -- "${1:?SIF image}")
RUN_ROOT=$(realpath -m -- "${2:?output root}")
PHASE=${3:-train}
case "$PHASE" in
  smoke) ITERATIONS=10; TIME=01:00:00 ;;
  train) ITERATIONS=200000; TIME=7-00:00:00 ;;
  *) echo 'Phase must be smoke or train.' >&2; exit 1 ;;
esac
mkdir -p "$RUN_ROOT/console" "$RUN_ROOT/status"
JOBS="$RUN_ROOT/jobs_${PHASE}.tsv"
[[ ! -e "$JOBS" ]] || { echo "Already submitted: $JOBS" >&2; exit 1; }
touch "$JOBS"
for task in Wbc-G1 Wbc-G1-SE Wbc-G1-Zest Wbc-G1-Zest-SE Wbc-G1-EE Wbc-G1-EE-SE; do
  job=$(sbatch --parsable --time="$TIME" --job-name="wbc_${PHASE}_${task}" \
    --output="$RUN_ROOT/console/%x-%j.out" --error="$RUN_ROOT/console/%x-%j.err" \
    "$REPO/scripts/cluster/train.sbatch" "$task" "$PHASE" "$ITERATIONS" \
    "$REPO" "$RUN_ROOT" "$IMAGE")
  printf '%s\t%s\n' "$task" "$job" | tee -a "$JOBS"
done
printf 'Jobs and outputs: %s\n' "$RUN_ROOT"

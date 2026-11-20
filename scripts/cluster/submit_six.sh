#!/usr/bin/env bash
# Submit fresh policies in priority order; a failed job blocks the remaining jobs.
# Usage: submit_six.sh IMAGE RUN_ROOT [PHASE] [run_checked.py options...]
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
IMAGE=$(realpath -- "${1:?SIF image}")
RUN_ROOT=$(realpath -m -- "${2:?output root}")
PHASE=${3:-train}
if (( $# >= 3 )); then shift 3; else shift 2; fi
case "$PHASE" in
  smoke) ITERATIONS=10; TIME=01:00:00 ;;
  stability) ITERATIONS=1000; TIME=02:00:00 ;;
  train) ITERATIONS=200000; TIME=7-00:00:00 ;;
  *) echo 'Phase must be smoke, stability, or train.' >&2; exit 1 ;;
esac
mkdir -p "$RUN_ROOT/console" "$RUN_ROOT/status"
JOBS="$RUN_ROOT/jobs_${PHASE}.tsv"
[[ ! -e "$JOBS" ]] || { echo "Already submitted: $JOBS" >&2; exit 1; }
touch "$JOBS"
previous=""
for task in Wbc-G1-SE Wbc-G1-Zest Wbc-G1 Wbc-G1-Zest-SE Wbc-G1-EE Wbc-G1-EE-SE; do
  dependency=()
  if [[ -n "$previous" ]]; then dependency=(--dependency="afterok:$previous"); fi
  job=$(sbatch --parsable "${dependency[@]}" --time="$TIME" --job-name="wbc_${PHASE}_${task}" \
    --output="$RUN_ROOT/console/%x-%j.out" --error="$RUN_ROOT/console/%x-%j.err" \
    "$REPO/scripts/cluster/train.sbatch" "$task" "$PHASE" "$ITERATIONS" \
    "$REPO" "$RUN_ROOT" "$IMAGE" "$@")
  printf '%s\t%s\n' "$task" "$job" | tee -a "$JOBS"
  previous="$job"
done
printf 'Jobs and outputs: %s\n' "$RUN_ROOT"

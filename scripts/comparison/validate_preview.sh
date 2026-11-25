#!/usr/bin/env bash
# Follow one sealed Slurm training run and validate immutable checkpoints locally.
set -euo pipefail
sm=$(realpath "${1:?SuperMimic checkout}")
seed=${2:?training seed}
job=${3:?Slurm job ID}
host=${4:?user@Slurm login}
port=${5:?SSH port}
wbc=$(cd "$(dirname "$0")/../.." && pwd)
art="$sm/artifacts/pipeline-comparison"
base="$wbc/logs/stability-final-v12/completed/Wbc-G1-20236"
store="$art/wbc-matched/seed-$seed"
out="$sm/logs/pipeline-comparison/wbc-matched/seed-$seed"
parent_job=$job
motion="$art/wbc-bones/validation/npz/crouch_turn_270_R_002__A244.npz"
py=${WBC_PYTHON:?WBC environment python executable}
mkdir -p "$store" "$out"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
export PYTHONPATH="$art/py-extra${PYTHONPATH:+:$PYTHONPATH}"
remote=(ssh -o BatchMode=yes -o ConnectTimeout=10 -p "$port" "$host")
for step in 5000 25000 45000 75000 100000 150000 175000 199999; do
  dest=$(printf "%s/iteration-%06d" "$out" "$step")
  if [[ -f "$dest/finished" ]]; then continue; fi
  mkdir -p "$dest"
  checkpoint="model_${step}.pt"
  job=$parent_job
  prior_seconds=0
  closed_parent=false
  if [[ -f "$out/recovery.json" ]]; then
    jq -e --arg parent "$parent_job" '.parent_job == $parent' "$out/recovery.json" >/dev/null
    boundary=$(jq -er .parent_iteration "$out/recovery.json")
    if (( step <= boundary )); then
      closed_parent=true
    else
      job=$(jq -er .successor_job "$out/recovery.json")
      prior_seconds=$(jq -er .parent_allocated_seconds "$out/recovery.json")
    fi
  fi
  run="wbc-mjlab-runs/comparison-matched-20261120/train/wbc_g1_preview/*seed${seed}_job${job}"
  # A later checkpoint or completed job establishes the writer has closed this file.
  while true; do
    state=$("${remote[@]}" "sacct -X -n -P -j $job -o State")
    case "$state" in
      COMPLETED*) break ;;
      FAILED*)
        if [[ "$closed_parent" == true ]] && "${remote[@]}" "test -s $run/$checkpoint"; then break; fi
        echo "Training job $job failed before validation step $step" >&2
        exit 1 ;;
      RUNNING*|PENDING*|CONFIGURING*|COMPLETING*)
        if (( step < 199999 )) && "${remote[@]}" "test -s $run/model_$((step+5000)).pt"; then break; fi ;;
      *) echo "Unexpected Slurm state for $job: $state" >&2; exit 1 ;;
    esac
    sleep 120
  done
  remote_record=$("${remote[@]}" "sha256sum $run/$checkpoint && stat -c %Y $run/$checkpoint && sacct -X -n -P -j $job -o Start")
  printf '%s\n' "$remote_record" > "$dest/cluster-checkpoint.txt"
  if (( prior_seconds > 0 )); then printf '%s\n' "$prior_seconds" > "$dest/prior-allocation-seconds.txt"; fi
  remote_hash=${remote_record%% *}
  if [[ "$closed_parent" == true && "$step" == "$boundary" ]]; then
    [[ "$remote_hash" == "$(jq -er .parent_sha256 "$out/recovery.json")" ]]
  fi
  if [[ ! -f "$store/$checkpoint" ]]; then
    rsync -at -e "ssh -o BatchMode=yes -p $port" "$host:$run/$checkpoint" "$store/"
  fi
  printf '%s  %s\n' "$remote_hash" "$store/$checkpoint" | sha256sum --check --status
  if [[ ! -f "$dest/evaluation/results.json" ]]; then
    staging=$(mktemp -d "$dest/export.XXXXXX")
    "$py" -u -m wbc_mjlab.scripts.evaluate --task Wbc-G1-Preview --checkpoint "$store/$checkpoint" \
      --motion-file "$motion" --output "$staging/evaluation" --physics-dt 0.0025 --device cuda:0 >"$staging/export.log" 2>&1
    mv "$staging/evaluation" "$dest/evaluation"
  fi
  jq -e --arg hash "$remote_hash" '.checkpoint_sha256 == $hash and .task == "Wbc-G1-Preview"' "$dest/evaluation/results.json" >/dev/null
  "$py" "$wbc/scripts/comparison/parity.py" "$base" "$dest" "$(dirname "$motion")" "$dest/adapter-parity.json"
  "$py" "$wbc/scripts/comparison/sweep.py" "$base" "$sm/docs/pipeline-comparison/bones-motions.csv" "$art/wbc-bones" "$dest/bones" \
    --actor "$dest" --split validation --rates 5 --horizons 15 45 --revisions none --workers 2
  "$py" "$wbc/scripts/comparison/sweep.py" "$base" "$art/lafan/manifest.csv" "$art/wbc-lafan" "$dest/lafan" \
    --actor "$dest" --split validation --rates 2 5 10 --horizons 5 15 45 --revisions none slow --workers 3
  date -Iseconds > "$dest/finished"
done

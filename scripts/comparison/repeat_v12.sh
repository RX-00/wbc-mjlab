#!/usr/bin/env bash
# Run on the Slurm login node. Supply a fresh path for this evaluation campaign.
set -euo pipefail
out=$(realpath -m -- "${1:?new absolute output directory}")
src="$HOME/wbc-mjlab-revisions/stability-20261028-final-v12"
image="$HOME/wbc-mjlab-images/2026-10-14_cbc0cf3/wbc-mjlab-cu128.sif"
previous="$HOME/wbc-mjlab-runs/2026-10-26_stability/final-v12-2p5ms"
mkdir "$out"
mkdir "$out/console"
# Bind previously completed training runs read-only into the evaluator.
export APPTAINER_BINDPATH="$previous/train:/training:ro"
node=a2r-wolf
for spec in Wbc-G1:20236 Wbc-G1-SE:19791 Wbc-G1-Zest:19792 Wbc-G1-Zest-SE:19794 Wbc-G1-EE:19795 Wbc-G1-EE-SE:19796; do
  task=${spec%:*}
  job=${spec#*:}
  ckpt=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$previous/status/train_${task}_job${job}.json")
  test -s "$previous/${ckpt#/runs/}"
  ckpt="/training/${ckpt#/runs/train/}"
  sbatch --parsable --nodelist="$node" --nodes=1 --time=03:00:00 \
    --job-name="repeat-$task" --output="$out/console/%x-%j.log" \
    "$src/scripts/cluster/check.sbatch" "$src" "$out" "$image" \
    -m wbc_mjlab.scripts.evaluate --task "$task" --checkpoint "$ckpt" \
    --onnx-file "$(dirname "$ckpt")/params/policy.onnx" --physics-dt 0.0025 \
    --motion-file /workspace/wbc-mjlab/data/g1/samples --output "/runs/$task"
  if [[ "$node" == a2r-wolf ]]; then node=a2r-fox; else node=a2r-wolf; fi
done

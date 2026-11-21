#!/usr/bin/env bash
# From a2r-main, using a checksummed source snapshot and the shared train split.
set -euo pipefail
src=$(realpath "${1:?source snapshot}")
out=$(realpath "${2:?campaign directory}")
image=$(realpath "${3:?v12 Apptainer image}")
(cd "$out/data/train" && sha256sum --quiet -c SHA256SUMS)
submit() {
  local phase=$1 seed=$2 node=$3 steps=$4
  shift 4
  sbatch --parsable --nodelist="$node" --job-name="preview-${phase}-${seed}" \
    --output="$out/console/%x-%j.log" "$@" "$src/scripts/cluster/train.sbatch" \
    Wbc-G1-Preview "$phase" "$steps" "$src" "$out" "$image" --physics-dt 0.0025 \
    --seed "$seed" --dataset-path /runs/data/train
}
ledger="$out/preview-jobs.tsv"
(set -o noclobber; : > "$ledger")
smoke=$(submit smoke 42 a2r-wolf 10 --time=00:45:00)
printf 'smoke\t42\t%s\n' "$smoke" >> "$ledger"
for spec in 42:a2r-wolf 43:a2r-fox 44:a2r-wolf; do
  seed=${spec%:*}
  dep=$smoke
  if [[ $seed == 44 ]]; then dep=$first; fi
  job=$(submit train "$seed" "${spec#*:}" 200000 --dependency="afterok:$dep" --kill-on-invalid-dep=yes)
  if [[ $seed == 42 ]]; then first=$job; fi
  printf 'train\t%s\t%s\n' "$seed" "$job" >> "$ledger"
done
cat "$ledger"

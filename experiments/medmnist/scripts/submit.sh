#!/bin/bash
# One-liner convenience: compute the array size from the config, then submit
# run.sh with the right partition. A plain wrapper — no self-submission.
# Cluster/interpreter settings come from env.sh (copy env.sh.example first).
#
#   ./submit.sh planecycle/lp
#   ./submit.sh planecycle/convnext_lp --cycle-order "HW DW DH"   # narrow to one setting
#   ./submit.sh baselines/convnext_lp  --block-type ACS --seed 42
# Args after the config are launch.py overrides (--arch/--block-type/--pool/
# --cycle-order/--seed/--family); they shrink the grid and the array size
# follows automatically.
set -euo pipefail

mode="${1:?usage: ./submit.sh <dir/lp|dir/ft> [launch overrides]}"
shift
here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$here/configs/env.sh" ] || { echo "Missing env.sh — cp configs/env.sh.example configs/env.sh and edit" >&2; exit 1; }
source "$here/configs/env.sh"

mkdir -p "$here/logs"
range=$("$PY" "$here/launch.py" --config "$mode" "$@" --array)
echo "Submitting '$mode' as array $range on $PARTITION ${*:+(overrides: $*)}"
sbatch --partition="$PARTITION" --array="$range" -J "${mode//\//_}" \
  --export="ALL,PC_SCRIPTS=$here" \
  "$here/run.sh" "$mode" "$@"

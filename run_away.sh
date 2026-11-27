#!/bin/bash
# USAGE:
#   ./run.sh [N]
#   N = how many commands to run at the same time (default: 1)
#   nohup ./run_away.sh >& run.log &
#   tail -f run.log
# ---- Write your commands here (one per line) ----

# Queue executions: provar augmentation amb més epochs, a veure si segueix aprenent, mirar corbes

COMMANDS=(
  #"python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --dataset scope95_full --inference /localdata-ssd/arnau/scope/scope95_fam_circular_permutation/scope95_fam_cp/ --inference-model sorted_results/full_try/big_one_dropout --output sorted_results/inference/cp_bigone"
  "python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --dataset scope95_full --hidden-dim 652 --seed 42 --model dicey --scale --ema --families-to-compare 25 --domains-per-family 4 --epochs 500 --dropout 0.2 --output sorted_results/full_try/big_one_long_run"
  #"python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --dataset scope95_full --hidden-dim 652 --seed 42 --model dicey --scale --outliers --augmentation --aug-mode all --ema --dropout 0.2 --families-to-compare 25 --domains-per-family 4 --output sorted_results/full_try/big_one_all"
  #"python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --dataset scope95_full --hidden-dim 652 --seed 42 --model dicey --scale --outliers --ema --dropout 0.2 --families-to-compare 25 --domains-per-family 4 --output sorted_results/full_try/big_one_outliers"
  #"python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --dataset scope95_full --hidden-dim 652 --seed 42 --model dicey --scale --augmentation --aug-mode all --ema --dropout 0.2 --families-to-compare 25 --domains-per-family 4 --output sorted_results/full_try/big_one_augmentation"
)
# --------------------------------------------------

N="${1:-1}"
if ! [[ "$N" =~ ^[1-9][0-9]*$ ]]; then
  echo "N must be a positive integer (got: $N)" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1   # so Python output shows up live

run() {  # $1 = job number, $2 = command
  bash -c "$2" 2>&1 | while IFS= read -r line; do echo "[$1] $line"; done
  return "${PIPESTATUS[0]}"
}

i=0
pids=()
for cmd in "${COMMANDS[@]}"; do
  i=$((i + 1))
  # wait until fewer than N jobs are running
  while (( $(jobs -rp | wc -l) >= N )); do sleep 0.2; done
  run "$i" "$cmd" &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit $failed

#!/bin/bash
# Usage: ./run.sh [N]
#   N = how many commands to run at the same time (default: 1)
#   ./run.sh      -> one at a time
#   ./run.sh 3    -> up to 3 at a time

# ---- Write your commands here (one per line) ----
COMMANDS=(
  "python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --representation sse --dataset scope95_small --epochs 120 --operator gine --seed 77 --output results/gine_sse/77"
  "python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --representation sse --dataset scope95_small --epochs 120 --operator gine --seed 420 --output results/gine_sse/420"
  "python graph_to_pytorch.py --annotation /localdata-ssd/arnau/scope/scope95_fam_annotated/ --representation sse --dataset scope95_small --epochs 120 --operator gine --seed 666 --output results/gine_sse/666"
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

#!/bin/bash
#SBATCH --job-name=fig3_steer
#SBATCH --partition=gpu-long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=titanrtx
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@300
#SBATCH --output=/mnt/scratch/r/runjia/fig3-workspace/logs/%x_%j.out

# Usage: sbatch --export=ALL,VARIANT=unit fig3/sbatch/steer.sh   (or VARIANT=white)
set -uo pipefail
: "${VARIANT:?set VARIANT=unit or VARIANT=white}"
WS=/mnt/scratch/r/runjia/fig3-workspace
source "$WS/venv/bin/activate"
export HF_HOME=$WS/hf_home HF_HUB_OFFLINE=1
REPO=$HOME/sae-for-vlm

python "$REPO/fig3/steer.py" \
    --variant "$VARIANT" \
    --sae_path "$WS/checkpoints/ae_200000.pt" \
    --white_image "$REPO/images/white.png" \
    --alpha 50 \
    --batch_size 16 \
    --out_dir "$WS/results" &
PID=$!
# 300 s before walltime Slurm sends USR1 to this script; forward it so steer.py
# checkpoints and exits 99, then resubmit ourselves to continue.
trap 'kill -USR1 $PID 2>/dev/null' USR1
rc=1
while true; do
    wait $PID
    rc=$?
    kill -0 $PID 2>/dev/null || break   # loop only if wait was interrupted by the trap
done
echo "steer.py exit code: $rc"
if [ "$rc" -eq 99 ] && [ "${RESUBMIT_COUNT:-0}" -lt 5 ]; then
    echo "resubmitting (count $((${RESUBMIT_COUNT:-0} + 1)))"
    sbatch --export=ALL,VARIANT=$VARIANT,RESUBMIT_COUNT=$((${RESUBMIT_COUNT:-0} + 1)) "$0"
fi
exit $rc

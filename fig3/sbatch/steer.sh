#!/bin/bash
#SBATCH --job-name=fig3_steer
#SBATCH --partition=gpu-long
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=titanrtx
#SBATCH --exclude=xgpe2,xgpe6
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/mnt/scratch/r/runjia/fig3-workspace/logs/%x_%j.out

# Usage: sbatch --export=ALL,VARIANT=unit fig3/sbatch/steer.sh   (or VARIANT=white)
set -uo pipefail
: "${VARIANT:?set VARIANT=unit or VARIANT=white}"
WS=/mnt/scratch/r/runjia/fig3-workspace
source "$WS/venv/bin/activate"
export HF_HOME=$WS/hf_home HF_HUB_OFFLINE=1
REPO=$HOME/sae-for-vlm

echo "=== start $(date) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} variant $VARIANT ==="
python "$REPO/fig3/steer.py" \
    --variant "$VARIANT" \
    --sae_path "$WS/checkpoints/ae_200000.pt" \
    --white_image "$REPO/images/white.png" \
    --alpha 50 \
    --batch_size 16 \
    --out_dir "$WS/results" &
PID=$!
# 300 s before walltime Slurm sends USR1 to this script; forward it so steer.py
# checkpoints and exits 99.
trap 'kill -USR1 $PID 2>/dev/null' USR1
while true; do
    wait $PID
    rc=$?
    if [ "$rc" -le 128 ]; then break; fi          # real exit code, not interrupted wait
    if ! kill -0 $PID 2>/dev/null; then
        wait $PID; rc=$?                           # child already gone: collect real rc
        break
    fi
done
echo "steer.py exit code: $rc"

# Requeue (same job id) so submit_all.sh's afterok edge to finalize survives the
# checkpoint/resume cycle. rc>128 = killed without USR1; the periodic checkpoint
# still allows resume. SLURM_RESTART_COUNT is maintained by Slurm across requeues.
if { [ "$rc" -eq 99 ] || [ "$rc" -gt 128 ]; } && [ "${SLURM_RESTART_COUNT:-0}" -lt 5 ]; then
    echo "requeueing job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0})"
    if ! scontrol requeue "$SLURM_JOB_ID"; then
        echo "WARNING: scontrol requeue failed; falling back to sbatch (this breaks"
        echo "the finalize dependency -- resubmit fig3/sbatch/finalize.sh manually"
        echo "with --dependency=afterok:<new ids> once steering finishes)"
        sbatch --export=ALL,VARIANT=$VARIANT "$0"
    fi
fi
exit $rc

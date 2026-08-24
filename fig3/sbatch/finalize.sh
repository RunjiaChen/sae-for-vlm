#!/bin/bash
#SBATCH --job-name=fig3_finalize
#SBATCH --partition=gpu
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=titanrtx
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/scratch/r/runjia/fig3-workspace/logs/%x_%j.out

set -uo pipefail
WS=/mnt/scratch/r/runjia/fig3-workspace
source "$WS/venv/bin/activate"
export HF_HOME=$WS/hf_home HF_HUB_OFFLINE=1
REPO=$HOME/sae-for-vlm

for VARIANT in unit white; do
    python "$REPO/fig3/sbert_score.py" \
        --steer_outputs "$WS/results/steer_outputs_${VARIANT}.json" \
        --interp_scores "$WS/results/interp_scores.pt" \
        --out_path "$WS/results/steer_scores_${VARIANT}.pt"
done

python "$REPO/fig3/plot_fig3.py" \
    --interp_scores "$WS/results/interp_scores.pt" \
    --steer_scores "$WS/results/steer_scores_unit.pt" "$WS/results/steer_scores_white.pt" \
    --out_dir "$WS/results"

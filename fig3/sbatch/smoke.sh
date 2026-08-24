#!/bin/bash
#SBATCH --job-name=fig3_smoke
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
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

python "$REPO/fig3/verify_equivalence.py" \
    --sae_path "$WS/checkpoints/ae_200000.pt" \
    --white_image "$REPO/images/white.png"

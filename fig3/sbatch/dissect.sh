#!/bin/bash
#SBATCH --job-name=fig3_dissect
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=titanrtx
#SBATCH --exclude=xgpe0,xgpe2,xgpe6
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/mnt/scratch/r/runjia/fig3-workspace/logs/%x_%j.out

set -uo pipefail
WS=/mnt/scratch/r/runjia/fig3-workspace
source "$WS/venv/bin/activate"
export HF_HOME=$WS/hf_home HF_HUB_OFFLINE=1
REPO=$HOME/sae-for-vlm

python "$REPO/fig3/dissect.py" \
    --acts_dir "$WS/results/activations" \
    --concepts_path "$WS/data/broden_labels_clean.txt" \
    --out_dir "$WS/results"

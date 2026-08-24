#!/bin/bash
#SBATCH --job-name=fig3_extract
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nv:1
#SBATCH --constraint=titanrtx
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/mnt/scratch/r/runjia/fig3-workspace/logs/%x_%j.out

set -uo pipefail
WS=/mnt/scratch/r/runjia/fig3-workspace
source "$WS/venv/bin/activate"
export HF_HOME=$WS/hf_home HF_HUB_OFFLINE=1
REPO=$HOME/sae-for-vlm

# probing images go to node-local disk, not the shared FS
IMG_DIR=${SLURM_TMPDIR:-/tmp}/fig3_imgs_$SLURM_JOB_ID
trap 'rm -rf "$IMG_DIR"' EXIT

python "$REPO/fig3/extract_activations.py" \
    --wds_dir "$WS/data/imagenet_val_wds" \
    --manifest_path "$WS/data/probing_manifest.json" \
    --img_dir "$IMG_DIR" \
    --sae_path "$WS/checkpoints/ae_200000.pt" \
    --out_dir "$WS/results/activations"

#!/bin/bash
# Environment setup for the Fig 3 replication (run once on xlogin1; network I/O only).
# Creates the venv and downloads all models + data into the scratch workspace.
set -euo pipefail

WS=/mnt/scratch/r/runjia/fig3-workspace
export HF_HOME=$WS/hf_home

# ---- 1. venv (system python 3.12.3; Pach pins torch 2.1.2 which has no cp312
#         wheels, so torch 2.5.1 is used; transformers stays at Pach's 4.46.3) ----
if [ ! -x "$WS/venv/bin/python" ]; then
    python3 -m venv "$WS/venv"
fi
source "$WS/venv/bin/activate"
pip install --upgrade pip
pip install \
    torch==2.5.1 torchvision==0.20.1 \
    transformers==4.46.3 accelerate==1.1.1 tokenizers==0.20.3 \
    sentencepiece==0.2.0 protobuf \
    sentence-transformers==3.2.1 \
    numpy==1.26.4 einops==0.8.0 matplotlib==3.9.3 pillow==11.0.0 tqdm==4.67.1 \
    "huggingface_hub[cli]"

# ---- 2. SAE checkpoint (ae_200000.pt -- ae_100000.pt does NOT exist on the hub) ----
SAE_REMOTE="matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_200000.pt"
if [ ! -f "$WS/checkpoints/ae_200000.pt" ]; then
    huggingface-cli download mateuszpach/sae-for-vlm "$SAE_REMOTE" --local-dir "$WS/checkpoints/hub"
    ln -sf "$WS/checkpoints/hub/$SAE_REMOTE" "$WS/checkpoints/ae_200000.pt"
fi

# ---- 3. Models into HF cache ----
huggingface-cli download llava-hf/llava-1.5-7b-hf
huggingface-cli download openai/clip-vit-large-patch14-336
huggingface-cli download sentence-transformers/all-mpnet-base-v2

# ---- 4. ImageNet-1K val (50k) as webdataset shards (ungated mirror) ----
mkdir -p "$WS/data/imagenet_val_wds"
for i in 0 1 2 3 4 5 6; do
    f="$WS/data/imagenet_val_wds/$i.tar"
    if [ ! -s "$f" ]; then
        wget -c -q -O "$f" "https://huggingface.co/datasets/clip-benchmark/wds_imagenet1k/resolve/main/test/$i.tar"
        echo "downloaded shard $i"
    fi
done

# ---- 5. Broden concept list from CLIP-dissect ----
if [ ! -s "$WS/data/broden_labels_clean.txt" ]; then
    wget -q -O "$WS/data/broden_labels_clean.txt" \
        "https://raw.githubusercontent.com/Trustworthy-ML-Lab/CLIP-dissect/master/data/broden_labels_clean.txt"
fi

echo "SETUP_DONE"

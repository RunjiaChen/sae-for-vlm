"""One GPU pass over the 50k probing images computing BOTH:

  Q: SAE latents at CLIP-ViT-L/14-336 layer 22 post_mlp_residual, mean-pooled over all
     577 tokens (Pach's --mean_pool convention)  -> (50000, 65536) fp16, chunked .pt
  A: clean (un-reconstructed) projected CLIP image embeddings -> (50000, 768) fp32

A must NOT go through the SAE, so instead of Pach's layer-replacing surgery we register
a non-destructive forward hook on layer 22 that reads its output, runs sae.encode on a
copy, and lets the forward pass continue untouched.
"""

import argparse
import os
import sys

import torch
import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, CLIPVisionModelWithProjection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE
from fig3.probing_dataset import ProbingDataset, build_manifest, extract_images


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--wds_dir', required=True)
    p.add_argument('--manifest_path', required=True)
    p.add_argument('--img_dir', required=True, help='node-local extraction dir')
    p.add_argument('--sae_path', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model_name', default='openai/clip-vit-large-patch14-336')
    p.add_argument('--layer', type=int, default=22)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--chunk_rows', type=int, default=10000, help='rows per saved Q chunk')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    done_flag = os.path.join(args.out_dir, '.extract_activations_done')
    if os.path.exists(done_flag):
        print('Already done, exiting.')
        return

    manifest = build_manifest(args.wds_dir, args.manifest_path)
    print(f'Manifest: {len(manifest)} images')
    assert len(manifest) == 50000, f'expected 50000 probing images, got {len(manifest)}'
    extract_images(args.wds_dir, manifest, args.img_dir)
    print('Images extracted')

    model = CLIPVisionModelWithProjection.from_pretrained(args.model_name).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name)
    sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path, device=args.device)
    sae.eval()
    dict_size = sae.dict_size
    print(f'SAE dict_size={dict_size} k={sae.k.item()} threshold={sae.threshold.item():.4f}')

    captured = {}

    def hook(module, inputs, output):
        # CLIPEncoderLayer returns a tuple; [0] is hidden_states (B, 577, 1024)
        captured['h'] = output[0]

    model.vision_model.encoder.layers[args.layer].register_forward_hook(hook)

    ds = ProbingDataset(args.img_dir, manifest)

    def collate(batch):
        return processor(images=batch, return_tensors='pt')

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate)

    n = len(ds)
    q_all = torch.empty(n, dict_size, dtype=torch.float16)
    a_all = torch.empty(n, model.config.projection_dim, dtype=torch.float32)

    row = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dl, desc='extract'):
            out = model(pixel_values=batch['pixel_values'].to(args.device))
            bsz = out.image_embeds.shape[0]
            a_all[row:row + bsz] = out.image_embeds.float().cpu()
            z = sae.encode(captured['h'].float())          # (B, 577, dict_size)
            q_all[row:row + bsz] = z.mean(dim=1).half().cpu()
            row += bsz
    assert row == n

    torch.save(a_all, os.path.join(args.out_dir, 'clip_image_embeds.pt'))
    nonzero = (q_all != 0).any(dim=0).sum().item()
    print(f'Neurons firing at least once on probing set: {nonzero}/{dict_size}')
    for start in range(0, n, args.chunk_rows):
        chunk = q_all[start:start + args.chunk_rows].clone()
        torch.save(chunk, os.path.join(args.out_dir, f'sae_q_rows{start:06d}.pt'))
    with open(done_flag, 'w') as f:
        f.write(f'rows={n} nonzero_neurons={nonzero}')
    print('DONE')


if __name__ == '__main__':
    main()

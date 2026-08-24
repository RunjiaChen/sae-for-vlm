"""One GPU pass over the 50k probing images computing BOTH:

  Q: SAE latents at CLIP-ViT-L/14-336 layer 22 post_mlp_residual, mean-pooled over all
     577 tokens (Pach's --mean_pool convention)  -> (50000, 65536) fp16, chunked .pt
  A: clean (un-reconstructed) projected CLIP image embeddings -> (50000, 768) fp32

A must NOT go through the SAE, so instead of Pach's layer-replacing surgery we register
a non-destructive forward hook on layer 22 that reads its output, runs sae.encode on a
copy, and lets the forward pass continue untouched.
"""

import argparse
import glob
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
    n = len(ds)

    def collate(batch):
        return processor(images=batch, return_tensors='pt')

    # Process in chunks of chunk_rows; each chunk is saved atomically as
    # sae_q_rows*.pt + a_rows*.pt so an interrupted job resumes at chunk level.
    nonzero_any = torch.zeros(dict_size, dtype=torch.bool)
    for start in range(0, n, args.chunk_rows):
        end = min(start + args.chunk_rows, n)
        q_path = os.path.join(args.out_dir, f'sae_q_rows{start:06d}.pt')
        a_path = os.path.join(args.out_dir, f'a_rows{start:06d}.pt')
        if os.path.exists(q_path) and os.path.exists(a_path):
            nonzero_any |= (torch.load(q_path, map_location='cpu') != 0).any(dim=0)
            print(f'chunk {start}-{end}: already done, skipping')
            continue
        subset = torch.utils.data.Subset(ds, list(range(start, end)))
        dl = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate)
        q_chunk = torch.empty(end - start, dict_size, dtype=torch.float16)
        a_chunk = torch.empty(end - start, model.config.projection_dim,
                              dtype=torch.float32)
        row = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(dl, desc=f'extract {start}-{end}'):
                out = model(pixel_values=batch['pixel_values'].to(args.device))
                bsz = out.image_embeds.shape[0]
                a_chunk[row:row + bsz] = out.image_embeds.float().cpu()
                z = sae.encode(captured['h'].float())      # (B, 577, dict_size)
                q_chunk[row:row + bsz] = z.mean(dim=1).half().cpu()
                row += bsz
        assert row == end - start
        for path, tensor in [(q_path, q_chunk), (a_path, a_chunk)]:
            torch.save(tensor, path + '.tmp')
            os.replace(path + '.tmp', path)
        nonzero_any |= (q_chunk != 0).any(dim=0)

    a_files = sorted(glob.glob(os.path.join(args.out_dir, 'a_rows*.pt')))
    a_all = torch.cat([torch.load(f, map_location='cpu') for f in a_files], dim=0)
    assert a_all.shape[0] == n
    torch.save(a_all, os.path.join(args.out_dir, 'clip_image_embeds.pt'))
    nonzero = nonzero_any.sum().item()
    print(f'Neurons firing at least once on probing set: {nonzero}/{dict_size}')
    with open(done_flag, 'w') as f:
        f.write(f'rows={n} nonzero_neurons={nonzero}')
    print('DONE')


if __name__ == '__main__':
    main()

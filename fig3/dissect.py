"""CLIP-Dissect over precomputed SAE activations.

Ports the exact recipe from Trustworthy-ML-Lab/CLIP-dissect (describe_neurons.py +
utils.get_similarity_from_activations + similarity.soft_wpmi with its default
hyperparameters top_k=100, a=10, lam=1) but consumes:
  - Q: precomputed SAE mean-pooled latents (50000 x 65536) from extract_activations.py
  - A: precomputed CLIP ViT-L/14-336 image embeddings (50000 x 768)
  - Broden concept list (1197 words), encoded as the raw word, no prompt template.

soft-WPMI is vectorized over neuron chunks (their reference loops one neuron at a
time); the math is unchanged, including the logsumexp p(d) term taken over ALL
neurons of the dissected layer.

Outputs in --out_dir:
  interp_scores.pt : {'scores': (65536,) fp32, 'concept_ids': (65536,) int64,
                      'concepts': list[str] (1197), 'nonzero_mask': (65536,) bool}
"""

import argparse
import glob
import os

import torch
import tqdm
from transformers import AutoTokenizer, CLIPTextModelWithProjection


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--acts_dir', required=True, help='output dir of extract_activations.py')
    p.add_argument('--concepts_path', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--model_name', default='openai/clip-vit-large-patch14-336')
    p.add_argument('--top_k', type=int, default=100)
    p.add_argument('--a', type=float, default=10.0)
    p.add_argument('--lam', type=float, default=1.0)
    p.add_argument('--min_prob', type=float, default=1e-7)
    p.add_argument('--p_start', type=float, default=0.998)
    p.add_argument('--p_end', type=float, default=0.97)
    p.add_argument('--neuron_chunk', type=int, default=2048)
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def load_q(acts_dir):
    files = sorted(glob.glob(os.path.join(acts_dir, 'sae_q_rows*.pt')))
    assert files, f'no Q chunks in {acts_dir}'
    return torch.cat([torch.load(f, map_location='cpu') for f in files], dim=0)


def encode_concepts(words, model_name, device):
    model = CLIPTextModelWithProjection.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    feats = []
    with torch.no_grad():
        for i in range(0, len(words), 256):
            inputs = tokenizer(words[i:i + 256], padding=True, return_tensors='pt').to(device)
            feats.append(model(**inputs).text_embeds.float().cpu())
    return torch.cat(feats, dim=0)


def soft_wpmi_chunked(clip_feats, q, top_k, a, lam, min_prob, p_start, p_end,
                      neuron_chunk, device):
    """clip_feats: (N_img, M) fp32 on GPU (pre-normalized product). q: (N_img, D) fp16 cpu.
    Returns mutual_info (D, M) fp32 on cpu. Same math as CLIP-dissect similarity.soft_wpmi."""
    n_img, n_concepts = clip_feats.shape
    dict_size = q.shape[1]
    clip_soft = torch.nn.functional.softmax(a * clip_feats, dim=1)  # (N, M)

    p_in = p_start - (torch.arange(top_k, device=device) / top_k * (p_start - p_end))
    p_in = p_in.view(1, top_k, 1)  # (1, K, 1)

    log_p_d_given_e = torch.empty(dict_size, n_concepts, dtype=torch.float32)
    for start in tqdm.tqdm(range(0, dict_size, neuron_chunk), desc='soft-wpmi'):
        qc = q[:, start:start + neuron_chunk].float().to(device)      # (N, C)
        inds = torch.topk(qc, k=top_k, dim=0)[1]                      # (K, C)
        feats = clip_soft[inds.T]                                     # (C, K, M)
        p_d_given_e = 1 + p_in * (feats - 1)                          # (C, K, M)
        log_p = torch.log(p_d_given_e + min_prob).sum(dim=1)          # (C, M)
        log_p_d_given_e[start:start + qc.shape[1]] = log_p.cpu()
        del qc, inds, feats, p_d_given_e, log_p

    lp = log_p_d_given_e.to(device)
    log_p_d = torch.logsumexp(lp, dim=0, keepdim=True) - torch.log(
        torch.tensor([float(dict_size)], device=device))
    mutual_info = lp - lam * log_p_d
    return mutual_info.cpu()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.concepts_path) as f:
        words = [w for w in f.read().split('\n') if w != '']
    print(f'{len(words)} concepts')
    assert len(words) == 1197, f'expected 1197 Broden concepts, got {len(words)}'

    image_feats = torch.load(os.path.join(args.acts_dir, 'clip_image_embeds.pt'),
                             map_location='cpu').float()
    text_feats = encode_concepts(words, args.model_name, args.device)
    image_feats /= image_feats.norm(dim=-1, keepdim=True)
    text_feats /= text_feats.norm(dim=-1, keepdim=True)
    clip_feats = (image_feats.to(args.device) @ text_feats.T.to(args.device))  # (N, M)
    print(f'P matrix: {tuple(clip_feats.shape)}')

    q = load_q(args.acts_dir)
    print(f'Q matrix: {tuple(q.shape)}')
    nonzero_mask = (q != 0).any(dim=0)
    print(f'neurons firing on probing set: {nonzero_mask.sum().item()}/{q.shape[1]}')

    sim = soft_wpmi_chunked(clip_feats, q, args.top_k, args.a, args.lam,
                            args.min_prob, args.p_start, args.p_end,
                            args.neuron_chunk, args.device)
    scores, concept_ids = torch.max(sim, dim=1)
    print(f'interpretability: mean={scores.mean():.4f} (paper ~0.154), '
          f'mean over firing neurons={scores[nonzero_mask].mean():.4f}')
    torch.save({'scores': scores, 'concept_ids': concept_ids, 'concepts': words,
                'nonzero_mask': nonzero_mask},
               os.path.join(args.out_dir, 'interp_scores.pt'))
    print('DONE')


if __name__ == '__main__':
    main()

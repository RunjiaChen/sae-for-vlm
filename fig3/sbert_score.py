"""Steerability scores: Sentence-BERT cosine between each neuron's steered one-word
LLaVA output and its CLIP-Dissect-assigned concept.

NOTE: this deliberately replaces Pach's steering_score.py metric (CLIP similarity
against top-activating-image embeddings), which is not the CB-SAE Fig 3 metric.

Words are stripped of surrounding whitespace/punctuation before embedding; the raw
strings stay untouched on disk (steer_outputs_*.json).
"""

import argparse
import json
import os
import string

import torch
from sentence_transformers import SentenceTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--steer_outputs', required=True, help='steer_outputs_{variant}.json')
    p.add_argument('--interp_scores', required=True, help='interp_scores.pt from dissect.py')
    p.add_argument('--out_path', required=True)
    p.add_argument('--model_name', default='sentence-transformers/all-mpnet-base-v2')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def clean(word):
    return word.strip().strip(string.punctuation + string.whitespace)


def main():
    args = parse_args()
    with open(args.steer_outputs) as f:
        steer = json.load(f)
    interp = torch.load(args.interp_scores, map_location='cpu', weights_only=False)
    words = [clean(w) for w in steer['words']]
    concepts = [interp['concepts'][i] for i in interp['concept_ids'].tolist()]
    assert len(words) == len(concepts), (len(words), len(concepts))

    model = SentenceTransformer(args.model_name, device=args.device)
    uniq = sorted(set(words) | set(concepts))
    emb = model.encode(uniq, batch_size=256, convert_to_tensor=True,
                       normalize_embeddings=True, show_progress_bar=True)
    idx = {w: i for i, w in enumerate(uniq)}
    w_emb = emb[torch.tensor([idx[w] for w in words])]
    c_emb = emb[torch.tensor([idx[c] for c in concepts])]
    scores = (w_emb * c_emb).sum(dim=1).cpu()

    empty = sum(1 for w in words if w == '')
    print(f'{len(words)} neurons, {len(set(words))} unique output words, '
          f'{empty} empty outputs')
    print(f'steerability ({steer["variant"]}): mean={scores.mean():.4f} '
          f'(paper Fig3 threshold 0.232; Table 1: 0.198 unit / 0.203 white)')
    torch.save({'variant': steer['variant'], 'scores': scores, 'words': words,
                'concepts': concepts}, args.out_path)
    print('DONE')


if __name__ == '__main__':
    main()

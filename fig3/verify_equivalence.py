"""Smoke test gating the full 65k steering runs.

Checks, for a sample of neurons and both variants:
  1. the rank-1-update reconstruction equals sae.decode() on the explicitly modified
     latents (allclose);
  2. the end-to-end batched path (fig3/steer.py machinery) produces the SAME generated
     string as Pach's reference path (models/llava.py attach_and_fix + llava.prompt,
     batch size 1).

Exits non-zero on any mismatch, which holds back the dependent steer jobs.
"""

import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE
from models.llava import Llava
from fig3.steer import PROMPT, build_prompt_embeds, precompute, steered_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sae_path', required=True)
    p.add_argument('--white_image', required=True)
    p.add_argument('--alpha', type=float, default=50.0)
    p.add_argument('--neurons', type=int, nargs='+',
                   default=[0, 1, 7, 100, 4095, 4096, 12288, 30000, 47646, 60000, 65535])
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def main():
    args = parse_args()
    args.variant = None  # filled per run
    args.layer = 22

    llava = Llava(args.device)
    sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path, device=args.device)
    sae.eval()

    model = llava.model
    processor = llava.processor
    prefix_emb, suffix_emb = build_prompt_embeds(model, processor, args.device)
    image = Image.open(args.white_image)

    failures = 0
    for variant, pre_zero in [('unit', True), ('white', False)]:
        args.variant = variant
        # precompute() must see the UNMODIFIED vision tower
        llava.model.vision_tower.vision_model.encoder.layers[llava.layer] = \
            llava.base_CLIPEncoderLayerPostMlpResidual
        _, z_white, base_recon = precompute(model, processor, sae, args)

        # --- check 1: rank-1 update vs explicit decode ---
        neurons = torch.tensor(args.neurons, device=args.device)
        fast = steered_features(neurons, z_white, base_recon, sae, args)
        for i, j in enumerate(args.neurons):
            z = torch.zeros_like(z_white) if pre_zero else z_white.clone()
            z[:, j] = args.alpha
            ref = sae.decode(z).to(torch.float16)[1:, :]
            if not torch.allclose(fast[i].float(), ref.float(), atol=2e-2, rtol=1e-3):
                diff = (fast[i].float() - ref.float()).abs().max().item()
                print(f'[FAIL] {variant} neuron {j}: recon max diff {diff}')
                failures += 1
        print(f'[ok] {variant}: rank-1 reconstruction matches explicit decode')

        # --- check 2: end-to-end strings vs Pach's reference path (batch 1) ---
        ref_words = {}
        for j in args.neurons:
            llava.attach_and_fix(sae=sae, neurons_to_fix={j: args.alpha}, pre_zero=pre_zero)
            ref_out = llava.prompt(PROMPT, image, max_tokens=5)[0].strip()
            ref_word = ref_out.split(' ')[0] if ref_out else ''

            with torch.no_grad():
                feats = steered_features(torch.tensor([j], device=args.device),
                                         z_white, base_recon, sae, args)
                img_emb = model.multi_modal_projector(feats)
                inputs_embeds = torch.cat([prefix_emb, img_emb, suffix_emb], dim=1)
                attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                  device=args.device)
                out = model.language_model.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attn,
                    max_new_tokens=5, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id
                    or processor.tokenizer.eos_token_id)
            mine = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
            my_word = mine.split(' ')[0] if mine else ''
            status = 'ok' if my_word == ref_word else 'FAIL'
            if status == 'FAIL':
                failures += 1
            ref_words[j] = ref_word
            print(f'[{status}] {variant} neuron {j}: ref="{ref_out}" mine="{mine}"')

        # restore the unmodified layer for the next variant
        llava.model.vision_tower.vision_model.encoder.layers[llava.layer] = \
            llava.base_CLIPEncoderLayerPostMlpResidual

        # --- check 3: the PRODUCTION batched configuration (all sample neurons in
        # one generate call). fp16 batching can flip borderline greedy tokens; a few
        # flips are tolerated, many mean something is wrong. ---
        with torch.no_grad():
            neurons_t = torch.tensor(args.neurons, device=args.device)
            feats = steered_features(neurons_t, z_white, base_recon, sae, args)
            img_emb = model.multi_modal_projector(feats)
            bsz = img_emb.shape[0]
            inputs_embeds = torch.cat([prefix_emb.expand(bsz, -1, -1), img_emb,
                                       suffix_emb.expand(bsz, -1, -1)], dim=1)
            attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                              device=args.device)
            out = model.language_model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attn,
                max_new_tokens=5, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id
                or processor.tokenizer.eos_token_id)
        decoded = processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        mismatches = 0
        for j, s in zip(args.neurons, decoded):
            w = s.strip().split(' ')[0] if s.strip() else ''
            if w != ref_words[j]:
                mismatches += 1
                print(f'[warn] {variant} neuron {j}: batched="{w}" bs1="{ref_words[j]}"')
        print(f'[{"ok" if mismatches <= len(args.neurons) // 5 else "FAIL"}] {variant}: '
              f'batched vs bs=1 mismatches {mismatches}/{len(args.neurons)}')
        if mismatches > len(args.neurons) // 5:
            failures += 1

    if failures:
        print(f'{failures} FAILURES')
        sys.exit(1)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

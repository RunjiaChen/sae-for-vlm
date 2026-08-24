"""Batched steering of all 65,536 SAE neurons through LLaVA-1.5-7B.

Semantics identical to Pach's models/llava.py attach_and_fix path (verified by
fig3/verify_equivalence.py) but ~2 orders of magnitude faster:

LLaVA-1.5 consumes vision_tower hidden_states[vision_feature_layer=-2], which is the
output of layer 22 -- exactly the SAE reconstruction after Pach's surgery. Layer 23 and
the final layernorm never touch it. So for the fixed white image we compute the
layer-22 pre-SAE hidden states h once, the SAE latents z = encode(h) once, and per
neuron j obtain the steered reconstruction by a rank-1 update:

  white variant (pre_zero=False): decode(z with z_j := alpha)
                                  = decode(z) + (alpha - z_j) * W_dec[j]
  unit  variant (pre_zero=True):  decode(alpha * e_j) = alpha * W_dec[j] + b_dec
                                  (every token identical)

The reconstruction is cast to fp16 (as SAEWrapper.decode does), CLS dropped, projected,
and spliced between the prompt's text embeddings; batches of neurons then go through
language_model.generate directly.

Checkpoints (scores computed so far, neuron index, RNG state) are written every
--checkpoint_every neurons, and on SIGUSR1 (sent by Slurm 300 s before walltime) the
current state is saved and the process exits with code 99 so the sbatch script can
resubmit itself.
"""

import argparse
import json
import os
import signal
import sys

import torch
import tqdm
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dictionary_learning.trainers import MatroyshkaBatchTopKSAE

PROMPT = "What is shown in this image? Use exactly one word!"

interrupted = False


def on_usr1(signum, frame):
    global interrupted
    interrupted = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=['unit', 'white'], required=True,
                   help='unit = pre_zero=True, white = pre_zero=False')
    p.add_argument('--sae_path', required=True)
    p.add_argument('--white_image', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--alpha', type=float, default=50.0)
    p.add_argument('--layer', type=int, default=22)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--checkpoint_every', type=int, default=512)
    p.add_argument('--max_new_tokens', type=int, default=5)
    p.add_argument('--limit', type=int, default=None, help='debug: only first N neurons')
    p.add_argument('--device', default='cuda')
    return p.parse_args()


def build_prompt_embeds(model, processor, device):
    """Returns (prefix_embeds, suffix_embeds) around the image slot, fp16 (1, T, 4096)."""
    conversation = [{
        'role': 'user',
        'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPT}],
    }]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    ids = processor.tokenizer(prompt, return_tensors='pt').input_ids[0]
    image_token = model.config.image_token_index
    img_pos = (ids == image_token).nonzero(as_tuple=True)[0]
    assert len(img_pos) >= 1, 'no <image> token in prompt'
    prefix_ids = ids[:img_pos[0]].unsqueeze(0).to(device)
    suffix_ids = ids[img_pos[-1] + 1:].unsqueeze(0).to(device)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        return embed(prefix_ids), embed(suffix_ids)


def precompute(model, processor, sae, args):
    """h (577,1024 fp16), z_white (577,dict fp32), base_recon (577,1024 fp32)."""
    n_layers = model.config.vision_config.num_hidden_layers
    vfl = model.config.vision_feature_layer
    assert n_layers == 24 and vfl in (-2, n_layers - 1), \
        f'shortcut assumes hidden_states[-2] == layer-22 output; got layers={n_layers} vfl={vfl}'
    assert model.config.vision_feature_select_strategy == 'default'

    image = Image.open(args.white_image)
    # LlavaProcessor in transformers 4.46.3 crashes on images-only calls; go straight
    # to the underlying CLIPImageProcessor (identical pixel_values).
    pixel_values = processor.image_processor(images=[image], return_tensors='pt')['pixel_values']
    pixel_values = pixel_values.to(args.device, torch.float16)
    with torch.no_grad():
        vt_out = model.vision_tower(pixel_values, output_hidden_states=True)
        h = vt_out.hidden_states[vfl][0]              # (577, 1024) fp16, layer-22 output
        z_white = sae.encode(h.float())               # (577, dict_size) fp32
        base_recon = sae.decode(z_white)              # (577, 1024) fp32
    return h, z_white, base_recon


def steered_features(neurons, z_white, base_recon, sae, args):
    """(B, 576, 4096-projector-input) fp16 reconstruction for a batch of neuron ids."""
    w = sae.W_dec[neurons]                            # (B, 1024) fp32
    if args.variant == 'unit':
        recon = args.alpha * w + sae.b_dec            # (B, 1024)
        recon = recon.unsqueeze(1).expand(-1, z_white.shape[0], -1)  # (B, 577, 1024)
    else:
        coeff = (args.alpha - z_white[:, neurons]).T.unsqueeze(-1)   # (B, 577, 1)
        recon = base_recon.unsqueeze(0) + coeff * w.unsqueeze(1)     # (B, 577, 1024)
    return recon.to(torch.float16)[:, 1:, :]          # drop CLS, fp16 like SAEWrapper.decode


def save_checkpoint(path, words, raw, next_index, args):
    state = {
        'variant': args.variant,
        'alpha': args.alpha,
        'words': words,
        'raw': raw,
        'next_index': next_index,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all(),
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    signal.signal(signal.SIGUSR1, on_usr1)
    os.makedirs(args.out_dir, exist_ok=True)
    # debug runs with --limit must never masquerade as the real 65k output
    suffix = f'_limit{args.limit}' if args.limit else ''
    ckpt_path = os.path.join(args.out_dir, f'steer_ckpt_{args.variant}{suffix}.pt')
    final_path = os.path.join(args.out_dir, f'steer_outputs_{args.variant}{suffix}.json')
    if os.path.exists(final_path):
        print('Final output already exists, exiting.')
        return

    model = LlavaForConditionalGeneration.from_pretrained(
        'llava-hf/llava-1.5-7b-hf', torch_dtype=torch.float16, device_map=args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained('llava-hf/llava-1.5-7b-hf')
    sae = MatroyshkaBatchTopKSAE.from_pretrained(args.sae_path, device=args.device)
    sae.eval()
    dict_size = sae.dict_size

    prefix_emb, suffix_emb = build_prompt_embeds(model, processor, args.device)
    _, z_white, base_recon = precompute(model, processor, sae, args)
    print(f'white-image active latents (mean-pooled != 0): '
          f'{(z_white != 0).any(dim=0).sum().item()}')

    total = args.limit if args.limit else dict_size
    words, raw = [], []
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        assert state['variant'] == args.variant and state['alpha'] == args.alpha
        words, raw = state['words'], state['raw']
        torch.set_rng_state(state['torch_rng'])
        print(f'Resumed at neuron {len(words)}')

    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    batch_size = args.batch_size
    pbar = tqdm.tqdm(total=total, initial=len(words), desc=f'steer-{args.variant}')

    oom_at_min_batch = 0
    while len(words) < total:
        if interrupted:
            save_checkpoint(ckpt_path, words, raw, len(words), args)
            print(f'SIGUSR1: checkpointed at neuron {len(words)}, exiting 99')
            sys.exit(99)
        start = len(words)
        neurons = torch.arange(start, min(start + batch_size, total), device=args.device)
        try:
            with torch.no_grad():
                feats = steered_features(neurons, z_white, base_recon, sae, args)
                img_emb = model.multi_modal_projector(feats)         # (B, 576, 4096)
                bsz = img_emb.shape[0]
                inputs_embeds = torch.cat([
                    prefix_emb.expand(bsz, -1, -1),
                    img_emb,
                    suffix_emb.expand(bsz, -1, -1),
                ], dim=1)
                attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                                  device=args.device)
                out = model.language_model.generate(
                    inputs_embeds=inputs_embeds, attention_mask=attn,
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=pad_id)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch_size == 1:
                oom_at_min_batch += 1
                if oom_at_min_batch >= 3:
                    save_checkpoint(ckpt_path, words, raw, len(words), args)
                    print(f'repeated OOM at batch_size=1, neuron {len(words)}; giving up')
                    sys.exit(1)
            batch_size = max(1, batch_size // 2)
            print(f'OOM -> batch_size={batch_size}')
            continue
        oom_at_min_batch = 0
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        for s in decoded:
            s = s.strip()
            raw.append(s)
            words.append(s.split(' ')[0] if s else '')
        pbar.update(len(decoded))

        if interrupted or len(words) % args.checkpoint_every < batch_size:
            save_checkpoint(ckpt_path, words, raw, len(words), args)
        if interrupted:
            print(f'SIGUSR1: checkpointed at neuron {len(words)}, exiting 99')
            sys.exit(99)

    save_checkpoint(ckpt_path, words, raw, len(words), args)
    with open(final_path + '.tmp', 'w') as f:
        json.dump({'variant': args.variant, 'alpha': args.alpha,
                   'words': words, 'raw': raw}, f)
    os.replace(final_path + '.tmp', final_path)
    print('DONE')


if __name__ == '__main__':
    main()

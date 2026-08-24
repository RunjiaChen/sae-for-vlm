# Replicating CB-SAE Figure 3 (Kulkarni et al., CVPR 2026, arXiv:2512.10805)

Scatter of all 65,536 SAE neurons: x = interpretability (CLIP-Dissect soft-WPMI score),
y = steerability (Sentence-BERT cosine between steered LLaVA one-word output and the
neuron's CLIP-Dissect concept). SAE = Pach et al. released Matryoshka Batch-Top-k (k=20, x64)
on CLIP-ViT-L/14-336 layer 22 `post_mlp_residual` (`mateuszpach/sae-for-vlm`).

Working branch: `replicate-fig3` in `~/sae-for-vlm` on SoC cluster (xlogin1).
All heavy compute via `sbatch` on `gpu`/`gpu-long`, target `--gres=gpu:nv:1 --constraint=titanrtx`.
Big files live in `/mnt/scratch/r/runjia/fig3-workspace/` (58 TB free), NOT in the repo.

## Status (newest first)

- 2026-08-24: Setup job 753702 died at the download step with "Stale file handle" —
  I rsynced the review-fixed setup_env.sh over the script WHILE the job's bash was
  executing it from NFS. Lesson: never edit a script a running job is executing.
  venv was fully installed before the abort (torch 2.5.1+cu124, transformers 4.46.3,
  tokenizers 0.20.3, sentence-transformers 3.2.1, numpy 1.26.4 — all pins correct).
  Resubmitted as job 753803 (idempotent script: pip no-ops, then downloads).

- 2026-08-24: Adversarial code review (13-agent workflow) found and fixed before first run:
  (1) CRITICAL: LlavaProcessor in transformers 4.46.3 crashes on images-only calls ->
  use processor.image_processor directly in steer.py precompute();
  (2) CRITICAL: `from dictionary_learning.trainers import ...` pulls in buffer.py ->
  nnsight (not installed); guarded the buffer import in dictionary_learning/__init__.py;
  (3) MAJOR: steer.sh's sbatch-self-resubmit on walltime broke finalize's afterok chain ->
  switched to `scontrol requeue` (same job id) + SLURM_RESTART_COUNT guard + open-mode=append;
  (4) MAJOR: partial-download guards ([ ! -s ]) accepted truncated wds shards -> .part+mv +
  exact size verification; (5) MAJOR: non-atomic image extraction could poison a requeued
  job -> atomic writes + wipe-on-missing-done-flag + restart-count-keyed tmp dir;
  plus: chunk-level resume in extract_activations.py, interrupted/OOM handling order in
  steer.py, --limit output suffixing, empty-word steer score floor 0.0, finalize set -e,
  batched-vs-bs1 check in the smoke gate, mkdir -p logs in submit_all.sh.
- 2026-08-24: Recon done. Branch created. Setup job 753702 (normal partition) installs venv
  + downloads. Compute nodes DO have internet (verified, job 753696).
- wds shard layout verified by HTTP range request: members are sNNNNNNN.cls/.jpg pairs
  in key order; 7 shards, 6.8 GB total.

## Key design decisions & assumptions (read before resuming)

1. **ImageNet-1K val source**: HF `clip-benchmark/wds_imagenet1k`, `test/` split = the 50k
   ILSVRC2012 val images as 7 webdataset tar shards (6.8 GB, ungated, plain HTTPS).
   No HF token exists on the cluster and `ILSVRC/imagenet-1k` is gated, so torchvision
   `ImageNet` is not used. **Probing-set order = shard order 0.tar..6.tar, members in
   stored tar order.** Both the SAE-activation pass and the CLIP-embedding pass use the
   same loader (`fig3/probing_dataset.py`) so row i always refers to the same image.
2. **Concept set**: `data/broden_labels_clean.txt` from Trustworthy-ML-Lab/CLIP-dissect —
   exactly 1197 concepts (file has no trailing newline; `wc -l` says 1196).
3. **CLIP-Dissect recipe** (ported from their `describe_neurons.py`/`utils.py`/`similarity.py`):
   P = (image_feats/||.||) @ (text_feats/||.||).T with concepts encoded as the raw word,
   no prompt template; similarity = `soft_wpmi(P, Q)` with defaults top_k=100, a=10, lam=1;
   interpretability score of neuron j = max_k similarity[j,k]; concept = argmax.
   Eval CLIP = ViT-L/14-336 (HF `openai/clip-vit-large-patch14-336`, equivalent weights to
   OpenAI clip pkg). Expected mean score ≈ 0.154.
4. **Neuron activation q_j on the probing set** = SAE encoder latents (post threshold, as in
   `MatroyshkaBatchTopKSAE.encode` default) **mean-pooled over all 577 tokens** (incl. CLS),
   matching Pach's `--mean_pool` convention used for HAIs. ASSUMPTION: CB-SAE does not state
   the token-pooling rule; mean-pool is CLIP-Dissect's `avg` pool analogue and Pach's own
   convention.
5. **SAE checkpoint**: `ae_200000.pt` (Pach's `scripts/mllm_steering.sh` references
   `ae_100000.pt` which does NOT exist on the hub — known bug, path fixed here).
6. **Steering**: LLaVA-1.5-7B fp16, white image `images/white.png`, prompt exactly
   `What is shown in this image? Use exactly one word!`, alpha = 50 (paper) not Pach's 100,
   greedy decoding, max_new_tokens=5, first word of the reply.
   Both variants are computed (per task spec, ambiguity #1):
   - `unit`: pre_zero=True — all latents zeroed, neuron j := 50
   - `white`: pre_zero=False — white-image latents kept, neuron j := 50
7. **Batched steering shortcut (the big speedup)**: LLaVA-1.5 consumes
   `vision_tower.hidden_states[-2]` = the *output of layer 22*, which after Pach's surgery is
   exactly the SAE reconstruction. Layer 23 + post-layernorm never touch it. So steered visual
   features are computed WITHOUT running the vision tower per neuron:
   h = layer-22 output on the white image (once); z = sae.encode(h) (once);
   per neuron j: decoded_j = decode(z with col j := 50) = base_recon + (50 - z[:,j]) * W_dec[j]
   (rank-1 update; for `unit` variant decoded_j = 50*W_dec[j] + b_dec on every token).
   Then drop CLS, multi_modal_projector, splice into prompt embeds, batch B neurons through
   `language_model.generate(inputs_embeds=...)`. Verified equivalent to Pach's
   `attach_and_fix` path on a sample of neurons before the big run (see smoke test).
8. **Steerability score** = cosine in Sentence-BERT (`all-mpnet-base-v2`) space between the
   steered one-word output and the neuron's CLIP-Dissect concept. Pach's `steering_score.py`
   (CLIP sim vs top-activating-image embeddings) computes a different metric and is not used.
   Raw model outputs are saved to disk so scoring can be redone without re-running GPUs.
9. Ambiguity #2 (recorded, per task): CB-SAE may have trained their own SAE; neuron indices
   won't match theirs. Target = distribution + quadrant percentages, not specific neurons.
   Paper targets: means interp≈0.154 / steer=0.232; quadrants 36.26 / 25.03 / 19.87 / 18.84 %.

## Cluster facts (verified 2026-08-24)

- `ssh soc` → xlogin1. Bastion needs `ssh -fN stujump` password once per 8 h (user-side).
- Partitions: gpu (3h), gpu-long (3d), normal (3h, CPU), long (3d, CPU). gpu DefaultTime=15min
  → ALWAYS pass `--time`.
- Many `nv` (titanrtx) nodes idle right now; also xgph11/12/14 (a100-40 ×2) idle.
  Sticking to titanrtx per task spec.
- System python 3.12.3, VENV module works, pip 24.0. No module/conda/containers.
- No HF token on cluster. Git identity was unset → set repo-locally to
  Runjia Chen <runjia@u.nus.edu>.
- `/mnt/scratch/r/runjia`: 39 MB used, FS has 58 TB free.

## Environment (exact commands)

See `fig3/setup_env.sh` (committed) — creates `/mnt/scratch/r/runjia/fig3-workspace/venv`
(python3.12) and installs pinned deps. Pach pins torch 2.1.2 (no cp312 wheels), so we use
torch 2.5.1 + transformers 4.46.3 (Pach's pin, kept for identical LLaVA behavior).
HF cache: `HF_HOME=/mnt/scratch/r/runjia/fig3-workspace/hf_home`.

## Pipeline / files (all under `fig3/` in the repo)

- `probing_dataset.py` — deterministic loader over the 7 wds shards (tarfile, no wds dep).
- `extract_activations.py` — one pass over the 50k probing images with CLIP-ViT-L/14-336:
  saves BOTH the SAE mean-pooled latents Q (50000 x 65536, fp16, chunked .pt) AND the
  projected CLIP image embeddings A (50000 x 768) in the same loop.
- `dissect.py` — Broden text embeddings, P = A_n @ T_n.T, soft-WPMI (ported), per-neuron
  concept + interpretability score → `interp_scores.pt`.
- `steer.py` — batched steering (see decision 7), checkpoint/resume, SIGUSR1-safe,
  `--variant unit|white` → raw one-word outputs per neuron (`steer_outputs_{variant}.json`).
- `sbert_score.py` — Sentence-BERT cosine(output word, assigned concept) → steer scores.
- `plot_fig3.py` — scatter + mean lines + quadrant percentages → `figure3_{variant}.png/pdf`.
- `sbatch/` — job scripts. All long jobs: gpu-long, titanrtx, `--signal=B:USR1@300` + trap.
- `verify_equivalence.py` — smoke test: my batched path vs Pach's attach_and_fix on N neurons.

## Data & downloads (all under /mnt/scratch/r/runjia/fig3-workspace/)

- `data/imagenet_val_wds/{0..6}.tar` — from HF clip-benchmark/wds_imagenet1k test/ (6.8 GB)
- `data/broden_labels_clean.txt` — from CLIP-dissect repo
- `checkpoints/ae_200000.pt` — 537 MB from HF mateuszpach/sae-for-vlm,
  `matroyshka_batch_top_k_20_x64/random_k_2/imagenet_train_activations_clip-vit-large-patch14-336_22_post_mlp_residual_matroyshka_batch_top_k_20_x64/trainer_0/checkpoints/ae_200000.pt`
- `hf_home/` — llava-hf/llava-1.5-7b-hf, openai/clip-vit-large-patch14-336,
  sentence-transformers/all-mpnet-base-v2

## Job log

(job IDs, nodes, log paths appended here as they are submitted)

## Open questions / gotchas

- Compute-node internet access: tested below (job `net_test`).
- soft-WPMI on dead neurons (never fire on probing set): topk over an all-zero q_j picks
  arbitrary images; score becomes a degenerate near-constant. Counted and reported, kept in
  the scatter (CB-SAE plotted all 65,536 points).
- LLaVA one-word replies may carry trailing punctuation ("Dog."); raw strings are saved,
  punctuation/whitespace stripped only inside sbert_score.py.

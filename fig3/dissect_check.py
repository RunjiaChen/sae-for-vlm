"""Scale investigation for the interpretability score.

Motivation: our vectorized soft-WPMI port gives mean max-similarity 2.71 on the
65,536 SAE neurons, while CB-SAE reports ~0.154 and CLIP-Dissects released CSVs
also show 0.1-0.3. This script, on our exact P and Q:
  1. runs the VERBATIM CLIP-Dissect similarity.soft_wpmi (per-neuron loop) on a
     neuron subset and on the full dict -> checks our port is numerically faithful;
  2. runs CLIP-Dissects cos_similarity on the same data -> checks whether that
     reproduces the papers ~0.154 scale.
"""
import glob, math, os, sys
import torch, tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fig3.dissect import load_q, encode_concepts, soft_wpmi_chunked

WS = "/mnt/scratch/r/runjia/fig3-workspace"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- verbatim from Trustworthy-ML-Lab/CLIP-dissect similarity.py (soft_wpmi) ----
def soft_wpmi_reference(clip_feats, target_feats, top_k=100, a=10, lam=1,
                        device=DEV, min_prob=1e-7, p_start=0.998, p_end=0.97):
    with torch.no_grad():
        clip_feats = torch.nn.functional.softmax(a*clip_feats, dim=1)
        inds = torch.topk(target_feats, dim=0, k=top_k)[1]
        prob_d_given_e = []
        p_in_examples = p_start-(torch.arange(start=0, end=top_k)/top_k*(p_start-p_end)).unsqueeze(1).to(device)
        for orig_id in tqdm.tqdm(range(target_feats.shape[1])):
            curr_clip_feats = clip_feats.gather(0, inds[:,orig_id:orig_id+1].expand(-1,clip_feats.shape[1])).to(device)
            curr_p_d_given_e = 1+p_in_examples*(curr_clip_feats-1)
            curr_p_d_given_e = torch.sum(torch.log(curr_p_d_given_e+min_prob), dim=0, keepdim=True)
            prob_d_given_e.append(curr_p_d_given_e)
        prob_d_given_e = torch.cat(prob_d_given_e, dim=0)
        prob_d = (torch.logsumexp(prob_d_given_e, dim=0, keepdim=True) -
                  torch.log(prob_d_given_e.shape[0]*torch.ones([1]).to(device)))
        mutual_info = prob_d_given_e - lam*prob_d
    return mutual_info

# ---- verbatim cos_similarity from the same file (batched matmul form) ----
def cos_similarity_reference(clip_feats, target_feats, device=DEV):
    with torch.no_grad():
        clip_feats = clip_feats / torch.norm(clip_feats, p=2, dim=0, keepdim=True)
        target_feats = target_feats / torch.norm(target_feats, p=2, dim=0, keepdim=True)
        batch_size = 4096
        sims = []
        for t_i in tqdm.tqdm(range(math.ceil(target_feats.shape[1]/batch_size))):
            curr = target_feats[:, t_i*batch_size:(t_i+1)*batch_size].to(device).T
            sims.append(curr @ clip_feats.to(device))
    return torch.cat(sims, dim=0)

with open(f"{WS}/data/broden_labels_clean.txt") as f:
    words = [w for w in f.read().split("\n") if w != ""]
image_feats = torch.load(f"{WS}/results/activations/clip_image_embeds.pt", map_location="cpu").float()
text_feats = encode_concepts(words, "openai/clip-vit-large-patch14-336", DEV)
image_feats /= image_feats.norm(dim=-1, keepdim=True)
text_feats /= text_feats.norm(dim=-1, keepdim=True)
P = image_feats.to(DEV) @ text_feats.T.to(DEV)
q = load_q(f"{WS}/results/activations")

stored = torch.load(f"{WS}/results/interp_scores.pt", map_location="cpu", weights_only=False)

# 1) reference soft-wpmi on the FULL dict (population matters for p(d))
mi_ref = soft_wpmi_reference(P.cpu() if DEV=="cpu" else P, q.float())
ref_scores, ref_ids = torch.max(mi_ref.cpu(), dim=1)
agree = (ref_ids == stored["concept_ids"]).float().mean().item()
diff = (ref_scores - stored["scores"]).abs().max().item()
print(f"[port check] argmax agreement {agree:.4f}, max |score diff| {diff:.6f}")
print(f"[soft-wpmi] mean={ref_scores.mean():.4f} median={ref_scores.median():.4f} "
      f"min={ref_scores.min():.4f} max={ref_scores.max():.4f}")

# 2) cos similarity variant
cs = cos_similarity_reference(P, q.float())
cs_scores, cs_ids = torch.max(cs.cpu(), dim=1)
print(f"[cos-sim]  mean={cs_scores.mean():.4f} median={cs_scores.median():.4f} "
      f"min={cs_scores.min():.4f} max={cs_scores.max():.4f} (paper mean ~0.154)")
overlap = (cs_ids == stored["concept_ids"]).float().mean().item()
print(f"[cos-sim] concept agreement with soft-wpmi assignment: {overlap:.4f}")
torch.save({"cos_scores": cs_scores, "cos_ids": cs_ids,
            "softwpmi_ref_scores": ref_scores},
           f"{WS}/results/dissect_check.pt")
print("DONE")

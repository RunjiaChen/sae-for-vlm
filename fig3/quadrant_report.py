"""Decide the interpretability-score definition empirically.

The paper says soft-WPMI, but raw soft-WPMI on our 65k population has mean 2.71 vs
the papers ~0.154 (cosine-like scale). For every available steering variant this
script builds the Fig-3 quadrant table under BOTH definitions:
  A) interp = raw soft-WPMI max (interp_scores.pt), concepts = its argmax
  B) interp = cos-similarity max (dissect_check.pt), concepts = its argmax
and compares quadrant percentages against the paper targets.
Steerability = SBERT cosine(steered word, assigned concept) in both cases.
"""
import json, os, string, sys
import torch
from sentence_transformers import SentenceTransformer

WS = "/mnt/scratch/r/runjia/fig3-workspace"
PAPER = {"ll": 36.26, "lh": 25.03, "hl": 19.87, "hh": 18.84}

def clean(w):
    return w.strip().strip(string.punctuation + string.whitespace)

interp = torch.load(f"{WS}/results/interp_scores.pt", map_location="cpu", weights_only=False)
check = torch.load(f"{WS}/results/dissect_check.pt", map_location="cpu", weights_only=False)
words_broden = interp["concepts"]

cos_scores = check["cos_scores"].clone().float()
cos_ids = check["cos_ids"].clone()
bad = ~torch.isfinite(cos_scores)
cos_scores[bad] = 0.0
cos_ids[bad] = 0
print(f"cos-sim: {int(bad.sum())} non-finite (dead) neurons floored to 0")
print(f"cos-sim: mean={cos_scores.mean():.4f} median={cos_scores.median():.4f} "
      f"min={cos_scores.min():.4f} max={cos_scores.max():.4f}  (paper ~0.154)")
print(f"soft-wpmi: mean={interp['scores'].mean():.4f}")

defs = {
    "softwpmi": (interp["scores"].float(), interp["concept_ids"]),
    "cossim": (cos_scores, cos_ids),
}

variants = {}
for v in ("unit", "white"):
    p = f"{WS}/results/steer_outputs_{v}.json"
    if os.path.exists(p):
        with open(p) as f:
            variants[v] = [clean(w) for w in json.load(f)["words"]]
print(f"variants available: {list(variants)}")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
allwords = set(words_broden)
for ws in variants.values():
    allwords |= set(ws)
uniq = sorted(allwords)
emb = model.encode(uniq, batch_size=256, convert_to_tensor=True,
                   normalize_embeddings=True, show_progress_bar=False)
idx = {w: i for i, w in enumerate(uniq)}

summary = {}
for dname, (iscores, ids) in defs.items():
    for v, steer_words in variants.items():
        concepts = [words_broden[i] for i in ids.tolist()]
        w_emb = emb[torch.tensor([idx[w] for w in steer_words])]
        c_emb = emb[torch.tensor([idx[c] for c in concepts])]
        s = (w_emb * c_emb).sum(dim=1).cpu().float()
        s[torch.tensor([w == "" for w in steer_words])] = 0.0
        x, y = iscores, s
        xt, yt = x.mean(), y.mean()
        n = len(x)
        q = {"ll": ((x < xt) & (y < yt)).sum().item() / n * 100,
             "lh": ((x < xt) & (y >= yt)).sum().item() / n * 100,
             "hl": ((x >= xt) & (y < yt)).sum().item() / n * 100,
             "hh": ((x >= xt) & (y >= yt)).sum().item() / n * 100}
        dev = sum(abs(q[k] - PAPER[k]) for k in q)
        print(f"\n=== {dname} x {v}: interp_mean={xt:.4f} steer_mean={yt:.4f} "
              f"(paper 0.154 / 0.232-ish) total|dev|={dev:.2f} ===")
        for k, label in [("ll","low/low"),("lh","low/high"),("hl","high/low"),("hh","high/high")]:
            print(f"  {label:9s}: {q[k]:6.2f}%   paper {PAPER[k]:6.2f}%")
        summary[f"{dname}_{v}"] = {"interp_mean": float(xt), "steer_mean": float(yt),
                                   "quadrants": q, "total_abs_dev": dev}
        torch.save({"variant": v, "assign": dname, "scores": s,
                    "words": steer_words, "concepts": concepts},
                   f"{WS}/results/steer_scores_{v}_{dname}.pt")

with open(f"{WS}/results/quadrant_report.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE")

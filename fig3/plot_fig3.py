"""Figure 3 replication: interpretability vs steerability scatter of all 65,536 SAE
neurons, quadrants split at each axis's mean, with quadrant percentages annotated.

Produces one figure per steering variant present, plus a text summary comparing
quadrant percentages to the paper's targets.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

PAPER = {  # quadrant -> (percent, count) from CB-SAE Fig. 3
    'low interp / low steer': (36.26, 23763),
    'low interp / high steer': (25.03, 16403),
    'high interp / low steer': (19.87, 13022),
    'high interp / high steer': (18.84, 12348),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--interp_scores', required=True)
    p.add_argument('--steer_scores', nargs='+', required=True,
                   help='steer_scores_{variant}.pt files')
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


def quadrants(x, y, x_thr, y_thr):
    return {
        'low interp / low steer': ((x < x_thr) & (y < y_thr)).sum().item(),
        'low interp / high steer': ((x < x_thr) & (y >= y_thr)).sum().item(),
        'high interp / low steer': ((x >= x_thr) & (y < y_thr)).sum().item(),
        'high interp / high steer': ((x >= x_thr) & (y >= y_thr)).sum().item(),
    }


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    interp = torch.load(args.interp_scores, map_location='cpu', weights_only=False)
    x = interp['scores'].float()
    summary = {'interp_mean': x.mean().item(), 'variants': {}}
    print(f'interpretability: mean={x.mean():.4f} (paper ~0.154)')

    for path in args.steer_scores:
        st = torch.load(path, map_location='cpu', weights_only=False)
        variant, y = st['variant'], st['scores'].float()
        x_thr, y_thr = x.mean().item(), y.mean().item()
        n = len(x)
        assert len(y) == n
        counts = quadrants(x, y, x_thr, y_thr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(x.numpy(), y.numpy(), s=1, alpha=0.05, color='#1f77b4',
                   rasterized=True, linewidths=0)
        ax.axvline(x_thr, color='gray', ls='--', lw=1)
        ax.axhline(y_thr, color='gray', ls='--', lw=1)
        for (qx, qy, key) in [(0.02, 0.02, 'low interp / low steer'),
                              (0.02, 0.98, 'low interp / high steer'),
                              (0.98, 0.02, 'high interp / low steer'),
                              (0.98, 0.98, 'high interp / high steer')]:
            pct = 100.0 * counts[key] / n
            ax.text(qx, qy, f'{pct:.2f}%', transform=ax.transAxes,
                    ha='left' if qx < 0.5 else 'right',
                    va='bottom' if qy < 0.5 else 'top', fontsize=11, fontweight='bold')
        ax.set_xlabel('Interpretability (CLIP-Dissect soft-WPMI)')
        ax.set_ylabel('Steerability (SBERT cosine, steered output vs concept)')
        ax.set_title(f'SAE neurons: interpretability vs steerability '
                     f'({variant} variant, alpha=50, n={n})')
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(os.path.join(args.out_dir, f'figure3_{variant}.{ext}'), dpi=200)
        plt.close(fig)

        print(f'\n=== variant {variant}: thresholds interp={x_thr:.4f} '
              f'(paper 0.154) steer={y_thr:.4f} (paper Fig3 0.232) ===')
        vs = {'steer_mean': y_thr, 'quadrants': {}}
        for key, cnt in counts.items():
            pct = 100.0 * cnt / n
            ppct, pcnt = PAPER[key]
            print(f'{key:26s}: {pct:6.2f}% ({cnt:5d})   paper: {ppct:.2f}% ({pcnt})')
            vs['quadrants'][key] = {'pct': pct, 'count': cnt,
                                    'paper_pct': ppct, 'paper_count': pcnt}
        summary['variants'][variant] = vs

    with open(os.path.join(args.out_dir, 'fig3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nDONE')


if __name__ == '__main__':
    main()

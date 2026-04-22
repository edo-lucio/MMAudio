"""Extract and visualize optimal GW couplings on an eval set.

Runs the given model on batches from ExtractedVGG_test, computes the GW
coupling T*, groups samples by class, and saves within-class / cross-class
heatmaps plus a ranking of strongest/weakest aligned classes.

Usage:
    python experiments/analyze_couplings.py \
        --weights output/gw_var_global/gw_var_global_ema_final.pth \
        --cfg exp_id=coupling_analysis \
        --out_dir analysis/couplings
"""
import argparse
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from mmaudio.data.extracted_vgg import ExtractedVGG
from mmaudio.model.gw_regularization import compute_gw_regularization
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.sequence_config import CONFIG_16K


def load_model(weights_path: str, cfg):
    empty_string_feat = torch.load('./ext_weights/empty_string.pth', weights_only=True)[0]
    net = get_my_mmaudio(cfg.model, empty_string_feat=empty_string_feat).cuda().eval()
    sd = torch.load(weights_path, map_location='cuda', weights_only=True)
    net.load_weights(sd)
    return net


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out_dir', default='analysis/couplings')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_batches', type=int, default=50)
    p.add_argument('--variant', default='global')
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with initialize(version_base='1.3.2', config_path='../config'):
        cfg = compose('train_config')

    seq_cfg = CONFIG_16K
    data_dim = dict(
        latent_seq_len=seq_cfg.latent_seq_len,
        clip_seq_len=seq_cfg.clip_seq_len,
        sync_seq_len=seq_cfg.sync_seq_len,
        text_seq_len=cfg.data_dim.text_seq_len,
        clip_dim=cfg.data_dim.clip_dim,
        sync_dim=cfg.data_dim.sync_dim,
        text_dim=cfg.data_dim.text_dim,
    )
    dset = ExtractedVGG(
        tsv_path=cfg.data.ExtractedVGG_test.tsv,
        premade_mmap_dir=cfg.data.ExtractedVGG_test.memmap_dir,
        data_dim=data_dim,
    )

    net = load_model(args.weights, cfg)

    # iterate batches; store per-sample labels and average couplings by class pair
    class_pair_T = defaultdict(list)
    class_counts = defaultdict(int)

    for bi in range(args.max_batches):
        start = bi * args.batch_size
        end = min(start + args.batch_size, len(dset))
        if start >= end:
            break
        rows = [dset[i] for i in range(start, end)]
        labels = [r['caption'] for r in rows]
        clip_f = torch.stack([r['clip_features'] for r in rows]).cuda()
        a_mean = torch.stack([r['a_mean'] for r in rows]).cuda()
        video_exist = torch.stack([r['video_exist'] for r in rows]).cuda()

        a_mean_norm = net.normalize(a_mean.clone())
        _, T = compute_gw_regularization(
            net,
            variant=args.variant,
            clip_f_raw=clip_f,
            x1=a_mean_norm,
            video_exist=video_exist,
            detach_video=True,
        )
        if T is None:
            continue
        T = T.cpu().numpy()

        for i, li in enumerate(labels):
            class_counts[li] += 1
            for j, lj in enumerate(labels):
                class_pair_T[(li, lj)].append(T[i, j])

    # per-class self-alignment (diagonal mass within a class group)
    self_scores = {}
    for c, _ in class_counts.items():
        vals = class_pair_T.get((c, c), [])
        if vals:
            self_scores[c] = float(np.mean(vals))

    df = pd.DataFrame(
        sorted(self_scores.items(), key=lambda x: -x[1]),
        columns=['class', 'mean_self_coupling'],
    )
    df.to_csv(out / 'class_self_alignment.csv', index=False)

    # heatmap: top-20 classes by count, show mean coupling between class pairs
    top = [c for c, _ in sorted(class_counts.items(), key=lambda x: -x[1])[:20]]
    H = np.zeros((len(top), len(top)))
    for i, ci in enumerate(top):
        for j, cj in enumerate(top):
            vals = class_pair_T.get((ci, cj), [])
            H[i, j] = float(np.mean(vals)) if vals else 0.0

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(H, cmap='viridis')
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title('Average GW coupling T*[i,j] between class pairs')
    fig.tight_layout()
    fig.savefig(out / 'class_pair_heatmap.png', dpi=150)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()

"""Motivating diagnostics: is MMAudio's embedding space relationally misaligned?

Runs the released MMAudio checkpoint (no training) on VGGSound test latents,
extracts paired (video, audio) representations with each of the four GW variants,
and computes four geometry probes:

  1. Pairwise-distance scatter + Pearson / Spearman correlation between
     D_V[i,k] and D_A[i,k] on the upper triangle.
  2. Raw entropic GW / FGW loss at init.
  3. kNN-graph agreement (Jaccard @ k in {5, 10, 20}).
  4. Effective rank of video- and audio-side representations.

Plus a class-pair coupling heatmap reusing compute_gw_regularization.

Usage:
    python experiments/diagnose_geometry.py \
        --weights weights/mmaudio_small_16k.pth \
        --out_dir analysis/motivation \
        --max_samples 2000

No training, no GPU-heavy inner loop; the whole run is O(minutes) on one GPU.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize
from scipy.stats import kendalltau, pearsonr, spearmanr

from mmaudio.data_mod.extracted_vgg import ExtractedVGG
from mmaudio.model.gw_regularization import (
    _extract_representations, _normalize_dist, compute_gw_regularization,
    entropic_gw_loss, fused_gw_loss, pairwise_distances,
)
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.sequence_config import CONFIG_16K
from mmaudio.utils.paths import repo_path

VARIANTS = ['global', 'projected', 'c_g', 'fused']


def load_net(weights_path, cfg):
    empty = torch.load(repo_path('ext_weights', 'empty_string.pth'), weights_only=True)[0]
    net = get_my_mmaudio(cfg.model, empty_string_feat=empty).cuda().eval()
    sd = torch.load(weights_path, map_location='cuda', weights_only=True)
    net.load_weights(sd)
    return net


def load_test_dset(cfg):
    seq = CONFIG_16K
    data_dim = dict(
        latent_seq_len=seq.latent_seq_len,
        clip_seq_len=seq.clip_seq_len,
        sync_seq_len=seq.sync_seq_len,
        text_seq_len=cfg.data_dim.text_seq_len,
        clip_dim=cfg.data_dim.clip_dim,
        sync_dim=cfg.data_dim.sync_dim,
        text_dim=cfg.data_dim.text_dim,
    )
    return ExtractedVGG(
        tsv_path=cfg.data.ExtractedVGG_test.tsv,
        premade_mmap_dir=cfg.data.ExtractedVGG_test.memmap_dir,
        data_dim=data_dim,
    )


@torch.no_grad()
def gather_reps(net, dset, variant, max_samples):
    """Pull (v, a) representations for up to max_samples paired test clips."""
    n = min(max_samples, len(dset))
    clip_chunks, x1_chunks, labels = [], [], []
    for i in range(n):
        r = dset[i]
        clip_chunks.append(r['clip_features'])
        x1_chunks.append(r['a_mean'])
        labels.append(r['caption'])
    clip_f = torch.stack(clip_chunks).cuda().float()
    x1 = torch.stack(x1_chunks).cuda().float()
    x1 = net.normalize(x1.clone())
    v, a = _extract_representations(net, variant, clip_f, x1)
    return v.cpu().numpy(), a.cpu().numpy(), labels


def upper_tri(M):
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def knn_jaccard(V, A, k):
    """Mean Jaccard overlap of k-NN sets on video vs audio sides."""
    DV = ((V[:, None] - V[None, :]) ** 2).sum(-1)
    DA = ((A[:, None] - A[None, :]) ** 2).sum(-1)
    np.fill_diagonal(DV, np.inf)
    np.fill_diagonal(DA, np.inf)
    nn_v = np.argsort(DV, axis=1)[:, :k]
    nn_a = np.argsort(DA, axis=1)[:, :k]
    jacc = []
    for i in range(len(V)):
        sv, sa = set(nn_v[i].tolist()), set(nn_a[i].tolist())
        u = len(sv | sa)
        jacc.append(len(sv & sa) / u if u else 0.0)
    return float(np.mean(jacc))


def effective_rank(X, eps=1e-12):
    S = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
    p = S / (S.sum() + eps)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def class_kendall_tau(V, A, labels):
    """Kendall-tau between class-pair mean distances on video vs audio sides."""
    by_class = defaultdict(list)
    for i, c in enumerate(labels):
        by_class[c].append(i)
    cls = [c for c, ids in by_class.items() if len(ids) >= 3]
    if len(cls) < 3:
        return float('nan')
    centroids_v = np.stack([V[by_class[c]].mean(0) for c in cls])
    centroids_a = np.stack([A[by_class[c]].mean(0) for c in cls])
    DV = ((centroids_v[:, None] - centroids_v[None, :]) ** 2).sum(-1)
    DA = ((centroids_a[:, None] - centroids_a[None, :]) ** 2).sum(-1)
    vec_v = upper_tri(DV)
    vec_a = upper_tri(DA)
    tau, _ = kendalltau(vec_v, vec_a)
    return float(tau)


@torch.no_grad()
def gw_value(V_np, A_np, fused=False, alpha=0.5, batch=64):
    """Mean entropic GW (or FGW) loss on random batches drawn from reps."""
    n = len(V_np)
    V = torch.from_numpy(V_np).cuda().float()
    A = torch.from_numpy(A_np).cuda().float()
    losses = []
    rng = np.random.default_rng(0)
    for _ in range(min(64, n // batch)):
        idx = rng.choice(n, size=batch, replace=False)
        v = V[idx]
        a = A[idx]
        DV = _normalize_dist(pairwise_distances(v))
        DA = _normalize_dist(pairwise_distances(a))
        if fused:
            vn = torch.nn.functional.normalize(v, dim=-1)
            an = torch.nn.functional.normalize(a, dim=-1)
            C = 1.0 - vn @ an.t()
            loss, _ = fused_gw_loss(DV, DA, C, alpha=alpha)
        else:
            loss, _ = entropic_gw_loss(DV, DA)
        losses.append(float(loss.item()))
    return float(np.mean(losses)), float(np.std(losses))


def plot_scatter(V, A, variant, out_path):
    DV = ((V[:, None] - V[None, :]) ** 2).sum(-1)
    DA = ((A[:, None] - A[None, :]) ** 2).sum(-1)
    dv = upper_tri(DV)
    da = upper_tri(DA)
    # subsample for plotting
    if len(dv) > 20000:
        idx = np.random.default_rng(0).choice(len(dv), 20000, replace=False)
        dv_s, da_s = dv[idx], da[idx]
    else:
        dv_s, da_s = dv, da
    r_pearson, _ = pearsonr(dv, da)
    r_spearman, _ = spearmanr(dv, da)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(dv_s, da_s, s=2, alpha=0.2, color='steelblue')
    ax.set_xlabel(r'$D_V[i,k]$ (video pairwise)')
    ax.set_ylabel(r'$D_A[i,k]$ (audio pairwise)')
    ax.set_title(f'{variant}\nPearson={r_pearson:.3f}, Spearman={r_spearman:.3f}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return r_pearson, r_spearman


def plot_knn_curve(results, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = results['ks']
    for v in VARIANTS:
        ax.plot(ks, [results['knn'][v][k] for k in ks], marker='o', label=v)
    ax.set_xlabel('k')
    ax.set_ylabel('Jaccard(kNN_video, kNN_audio)')
    ax.set_title('kNN-graph agreement across modalities (released MMAudio)')
    ax.axhline(1.0, ls='--', color='gray', alpha=0.5, label='perfect alignment')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_coupling_heatmap(net, dset, out_path, variant='global',
                          batch_size=64, max_batches=40, top_n=20):
    class_pair_T = defaultdict(list)
    class_counts = defaultdict(int)
    for bi in range(max_batches):
        start = bi * batch_size
        end = min(start + batch_size, len(dset))
        if start >= end:
            break
        rows = [dset[i] for i in range(start, end)]
        labels = [r['caption'] for r in rows]
        clip_f = torch.stack([r['clip_features'] for r in rows]).cuda()
        a_mean = torch.stack([r['a_mean'] for r in rows]).cuda()
        video_exist = torch.stack([r['video_exist'] for r in rows]).cuda()
        a_mean_norm = net.normalize(a_mean.clone())
        _, T, _ = compute_gw_regularization(
            net, variant=variant, clip_f_raw=clip_f, x1=a_mean_norm,
            video_exist=video_exist, detach_video=True,
        )
        if T is None:
            continue
        T = T.cpu().numpy()
        for i, li in enumerate(labels):
            class_counts[li] += 1
            for j, lj in enumerate(labels):
                class_pair_T[(li, lj)].append(T[i, j])

    top = [c for c, _ in sorted(class_counts.items(), key=lambda x: -x[1])[:top_n]]
    H = np.zeros((len(top), len(top)))
    for i, ci in enumerate(top):
        for j, cj in enumerate(top):
            vals = class_pair_T.get((ci, cj), [])
            H[i, j] = float(np.mean(vals)) if vals else 0.0
    diag_mass = float(np.trace(H) / H.sum()) if H.sum() > 0 else 0.0
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(H, cmap='viridis')
    ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'GW coupling T*[i,j] at init ({variant}); '
                 f'diag-mass={diag_mass:.3f}')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return diag_mass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out_dir', default='analysis/motivation')
    p.add_argument('--max_samples', type=int, default=2000)
    p.add_argument('--ks', type=int, nargs='+', default=[5, 10, 20])
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with initialize(version_base='1.3.2', config_path='../../config'):
        cfg = compose('train_config')

    dset = load_test_dset(cfg)
    net = load_net(args.weights, cfg)

    rows = []
    knn_by_variant = {}
    scatter_info = {}
    for v in VARIANTS:
        V, A, labels = gather_reps(net, dset, v, args.max_samples)
        r_p, r_s = plot_scatter(V, A, v, out / f'scatter_{v}.png')
        scatter_info[v] = (r_p, r_s)

        knn_by_variant[v] = {}
        for k in args.ks:
            knn_by_variant[v][k] = knn_jaccard(V, A, k)

        gw_mean, gw_std = gw_value(V, A, fused=(v == 'fused'))
        tau = class_kendall_tau(V, A, labels)
        er_v = effective_rank(V)
        er_a = effective_rank(A)
        rows.append({
            'variant': v,
            'pearson_DV_DA': r_p,
            'spearman_DV_DA': r_s,
            'gw_init_mean': gw_mean,
            'gw_init_std': gw_std,
            'class_kendall_tau': tau,
            'erank_video': er_v,
            'erank_audio': er_a,
            **{f'jaccard_knn@{k}': knn_by_variant[v][k] for k in args.ks},
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / 'metrics.csv', index=False)
    df.to_latex(out / 'metrics.tex', index=False, float_format='%.3f')

    plot_knn_curve({'ks': args.ks, 'knn': knn_by_variant}, out / 'knn.png')
    diag = plot_coupling_heatmap(net, dset, out / 'coupling.png', variant='global')

    print(df.to_string(index=False))
    print(f'Coupling diagonal mass (global): {diag:.3f}')
    print(f'Wrote diagnostics to {out}')


if __name__ == '__main__':
    main()

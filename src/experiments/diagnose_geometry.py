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
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

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


def linear_cka(V, A):
    """Centered Kernel Alignment with linear kernel — relational alignment in [0, 1]."""
    Vc = V - V.mean(0, keepdims=True)
    Ac = A - A.mean(0, keepdims=True)
    KV = Vc @ Vc.T
    KA = Ac @ Ac.T
    hsic = (KV * KA).sum()
    denom = np.sqrt((KV * KV).sum() * (KA * KA).sum()) + 1e-12
    return float(hsic / denom)


def procrustes_residual(V, A):
    """Best orthogonal alignment of V→A: returns relative residual ∈ [0, 1].

    A residual ≪ 1 means a single rotation already aligns the spaces — entropic
    GW is then redundant and can introduce bias.
    """
    Vc = V - V.mean(0, keepdims=True)
    Ac = A - A.mean(0, keepdims=True)
    if Vc.shape[1] != Ac.shape[1]:
        # pad shorter side with zeros so we can compare
        d = max(Vc.shape[1], Ac.shape[1])
        Vp = np.zeros((Vc.shape[0], d)); Vp[:, :Vc.shape[1]] = Vc
        Ap = np.zeros((Ac.shape[0], d)); Ap[:, :Ac.shape[1]] = Ac
        Vc, Ac = Vp, Ap
    M = Ac.T @ Vc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    resid = np.linalg.norm(Vc @ R.T - Ac, ord='fro') ** 2
    denom = np.linalg.norm(Ac, ord='fro') ** 2 + 1e-12
    return float(resid / denom)


def gw_permutation_test(V_np, A_np, n_perm=20, batch=64, fused=False, alpha=0.5):
    """GW(V, A) vs GW(V, shuffle(A)). Tight gap = no relational signal to align."""
    n = len(V_np)
    if n < batch * 2:
        return float('nan'), float('nan'), float('nan')
    V = torch.from_numpy(V_np).cuda().float()
    A = torch.from_numpy(A_np).cuda().float()
    rng = np.random.default_rng(0)

    def _gw(v, a):
        DV = _normalize_dist(pairwise_distances(v))
        DA = _normalize_dist(pairwise_distances(a))
        if fused:
            vn = torch.nn.functional.normalize(v, dim=-1)
            an = torch.nn.functional.normalize(a, dim=-1)
            C = 1.0 - vn @ an.t()
            loss, _ = fused_gw_loss(DV, DA, C, alpha=alpha)
        else:
            loss, _ = entropic_gw_loss(DV, DA)
        return float(loss.item())

    paired, shuffled = [], []
    for _ in range(n_perm):
        idx = rng.choice(n, size=batch, replace=False)
        v = V[idx]
        a = A[idx]
        paired.append(_gw(v, a))
        perm = rng.permutation(batch)
        shuffled.append(_gw(v, a[perm]))
    paired = np.array(paired); shuffled = np.array(shuffled)
    z = (shuffled.mean() - paired.mean()) / (paired.std() + shuffled.std() + 1e-12)
    return float(paired.mean()), float(shuffled.mean()), float(z)


def plot_svd_spectra(reps_by_variant, out_path):
    """Log-scale singular values for V and A across variants — exposes mode collapse."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, side in zip(axes, ['video', 'audio']):
        for v, (V, A) in reps_by_variant.items():
            X = V if side == 'video' else A
            S = np.linalg.svd(X - X.mean(0, keepdims=True), compute_uv=False)
            S = S / (S[0] + 1e-12)
            ax.semilogy(S, label=v, lw=1.5)
        ax.set_title(f'{side} singular spectrum (normalized)')
        ax.set_xlabel('index'); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'$\sigma_i / \sigma_0$')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_procrustes_cka_bar(rows, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    variants = [r['variant'] for r in rows]
    x = np.arange(len(variants))
    w = 0.35
    ax.bar(x - w/2, [r['linear_cka'] for r in rows], width=w, label='linear CKA')
    ax.bar(x + w/2, [r['procrustes_resid'] for r in rows], width=w, label='Procrustes residual')
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel('value'); ax.set_ylim(0, 1.05)
    ax.set_title('CKA (higher = more aligned) vs Procrustes residual (lower = more aligned)')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _2d_embedding(X, n_neighbors=15, min_dist=0.1, seed=0):
    """UMAP if available, otherwise PCA fallback. Returns (N, 2) float."""
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
            min_dist=min_dist,
            random_state=seed,
            metric='euclidean',
        )
        return reducer.fit_transform(X), 'UMAP'
    except Exception:
        Xc = X - X.mean(0, keepdims=True)
        try:
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd as scipy_svd
            U, S, _ = scipy_svd(Xc, full_matrices=False, lapack_driver='gesvd')
        return (U[:, :2] * S[:2]), 'PCA'


def plot_pairwise_dist_overlay(V, A, variant, out_path, bins=80):
    """Histogram overlay of mean-normalised pairwise squared L2 on the upper
    triangle. GW operates on these matrices -- if the marginals match shape,
    GW's optimisation is well-posed; mismatched shapes diagnose where the
    entropic relaxation has to do work.
    """
    DV = ((V[:, None] - V[None, :]) ** 2).sum(-1)
    DA = ((A[:, None] - A[None, :]) ** 2).sum(-1)
    dv = upper_tri(DV) / (upper_tri(DV).mean() + 1e-12)
    da = upper_tri(DA) / (upper_tri(DA).mean() + 1e-12)
    rng_hi = float(np.percentile(np.concatenate([dv, da]), 99.0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dv, bins=bins, range=(0, rng_hi), density=True,
            alpha=0.5, label='video', color='steelblue')
    ax.hist(da, bins=bins, range=(0, rng_hi), density=True,
            alpha=0.5, label='audio', color='darkorange')
    ax.set_xlabel('normalised pairwise squared L2')
    ax.set_ylabel('density')
    ax.set_title(f'{variant}: pairwise-distance distribution')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_shepard(V, A, variant, out_path, max_pairs=20000):
    """Distance-rank scatter. A monotone curve = order-preserving up to a
    monotone reparameterisation of the metric -- the invariance class
    quadratic GW operates within. The Spearman annotation summarises it.
    """
    DV = ((V[:, None] - V[None, :]) ** 2).sum(-1)
    DA = ((A[:, None] - A[None, :]) ** 2).sum(-1)
    dv = upper_tri(DV); da = upper_tri(DA)
    rv = rankdata(dv); ra = rankdata(da)
    rho, _ = spearmanr(dv, da)
    if len(rv) > max_pairs:
        idx = np.random.default_rng(0).choice(len(rv), max_pairs, replace=False)
        rv_s, ra_s = rv[idx], ra[idx]
    else:
        rv_s, ra_s = rv, ra
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(rv_s, ra_s, s=2, alpha=0.2, color='steelblue')
    ax.plot([1, len(rv)], [1, len(rv)], 'r--', lw=1, alpha=0.7,
            label='isotonic reference')
    ax.set_xlabel(r'rank of $D_V[i,j]$')
    ax.set_ylabel(r'rank of $D_A[i,j]$')
    ax.set_title(f'{variant}: Shepard plot   |   Spearman ρ = {rho:.3f}')
    ax.legend()
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_mantel_profile(V, A, variant, out_path,
                        quantiles=(0.05, 0.10, 0.25, 0.50, 1.00)):
    """Pearson(D_V, D_A) restricted to the closest q-fraction of pairs in V,
    plotted vs q. A *descending* curve (high local r, low global r) is the
    GW signature: rigid methods average over scales and miss this.
    """
    DV = ((V[:, None] - V[None, :]) ** 2).sum(-1)
    DA = ((A[:, None] - A[None, :]) ** 2).sum(-1)
    dv = upper_tri(DV); da = upper_tri(DA)
    n = len(dv)
    rv = rankdata(dv)
    rs = []
    for q in quantiles:
        cutoff = max(8, int(np.ceil(q * n)))
        sel = rv <= cutoff
        if sel.sum() < 8:
            rs.append(np.nan); continue
        r, _ = pearsonr(dv[sel], da[sel])
        rs.append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([100 * q for q in quantiles], rs, 'o-', color='darkgreen', lw=1.5)
    ax.axhline(0.0, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('quantile of $D_V$ retained (closest q% of pairs)')
    ax.set_ylabel(r'Pearson($D_V$, $D_A$) on selected pairs')
    ax.set_title(f'{variant}: Mantel correlation across distance scales')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def _procrustes_2d(target_2d: np.ndarray, source_2d: np.ndarray) -> np.ndarray:
    """Best similarity transform (rotation + scale + translation) of source
    onto target. Used to overlay UMAP(V) and UMAP(A) for the visual residual.
    """
    Xc = target_2d - target_2d.mean(0)
    Yc = source_2d - source_2d.mean(0)
    M = Yc.T @ Xc
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd
        U, _, Vt = scipy_svd(M, full_matrices=False, lapack_driver='gesvd')
    R = U @ Vt
    s = float((Xc * (Yc @ R)).sum() / ((Yc * Yc).sum() + 1e-12))
    return target_2d.mean(0) + s * (Yc @ R)


def plot_procrustes_umap_overlay(V, A, labels, variant, out_path,
                                 max_points=600, n_links=120,
                                 top_n_classes=12):
    """Overlay UMAP(V) and Procrustes-aligned UMAP(A). Pair-connectors visualise
    what's left after the *best rigid* alignment -- precisely what GW is meant
    to fix non-rigidly. Long residuals = GW has work to do; short residuals =
    a similarity transform sufficed and GW is overkill.
    """
    n = min(len(V), len(A), max_points)
    rng = np.random.default_rng(0)
    if len(V) > n:
        idx = rng.choice(len(V), n, replace=False)
        V = V[idx]; A = A[idx]; labels = [labels[i] for i in idx]

    V_2d, method = _2d_embedding(V)
    A_2d, _ = _2d_embedding(A)
    A_aligned = _procrustes_2d(V_2d, A_2d)

    counts = defaultdict(int)
    for c in labels:
        counts[c] += 1
    top = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n_classes]]
    cmap = plt.get_cmap('tab20', max(len(top), 1))
    color_of = {c: cmap(i) for i, c in enumerate(top)}

    fig, ax = plt.subplots(figsize=(8, 7))
    link_idx = rng.choice(n, size=min(n_links, n), replace=False)
    for i in link_idx:
        ax.plot([V_2d[i, 0], A_aligned[i, 0]],
                [V_2d[i, 1], A_aligned[i, 1]],
                color='gray', alpha=0.25, lw=0.5)
    for c in top:
        mask = np.array([labels[i] == c for i in range(n)])
        if not mask.any():
            continue
        ax.scatter(V_2d[mask, 0], V_2d[mask, 1],
                   s=14, c=[color_of[c]], marker='o', alpha=0.85,
                   edgecolors='none', label=f'V · {str(c)[:18]}')
        ax.scatter(A_aligned[mask, 0], A_aligned[mask, 1],
                   s=14, c=[color_of[c]], marker='^', alpha=0.85,
                   edgecolors='none')
    resid = float(np.linalg.norm(V_2d - A_aligned, ord='fro') ** 2 /
                  (np.linalg.norm(V_2d, ord='fro') ** 2 + 1e-12))
    ax.set_title(
        f'{variant}: Procrustes-aligned {method} overlay '
        f'(○ V, △ A; relative residual = {resid:.3f})'
    )
    ax.set_xticks([]); ax.set_yticks([])
    handles, lbls = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, lbls, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _betti0_curve(D: np.ndarray, n_grid: int = 80):
    """β₀(r) -- connected components of the Vietoris-Rips filtration on D
    via single-linkage / union-find. Pure numpy, no TDA dependencies.
    """
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    edges = D[iu]
    order = np.argsort(edges)
    rs = edges[order]
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    components = n
    rs_grid = np.linspace(0.0, float(rs[-1]) if rs.size else 1.0, n_grid)
    bettis = []
    j = 0
    src = iu[0][order]; dst = iu[1][order]
    for r in rs_grid:
        while j < len(rs) and rs[j] <= r:
            a, b = find(int(src[j])), find(int(dst[j]))
            if a != b:
                parent[a] = b
                components -= 1
            j += 1
        bettis.append(components)
    return rs_grid, np.array(bettis)


def plot_betti0_overlay(V, A, variant, out_path, max_points=400):
    """Overlay β₀(r) for V and A on a normalised filtration radius. Matching
    curves = matching merge dynamics = same global connectedness profile,
    a coarse but dependency-free topological agreement check.
    """
    n = min(len(V), len(A), max_points)
    if len(V) > n:
        idx = np.random.default_rng(0).choice(len(V), n, replace=False)
        V, A = V[idx], A[idx]
    DV = ((V[:, None] - V[None, :]) ** 2).sum(-1)
    DA = ((A[:, None] - A[None, :]) ** 2).sum(-1)
    DV = DV / (DV.mean() + 1e-12)
    DA = DA / (DA.mean() + 1e-12)
    rv, bv = _betti0_curve(DV)
    ra, ba = _betti0_curve(DA)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rv, bv, label='video', color='steelblue', lw=1.5)
    ax.plot(ra, ba, label='audio', color='darkorange', lw=1.5)
    ax.set_xlabel('filtration radius r (mean-normalised)')
    ax.set_ylabel(r'$\beta_0(r)$ -- connected components')
    ax.set_yscale('log'); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f'{variant}: Betti-0 curves (single-linkage Vietoris-Rips)')
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_paired_umap(V, A, labels, variant, out_path,
                     top_n_classes=12, max_points=2000, cka=None, proc=None):
    """Side-by-side labelled 2D embeddings of V and A.

    Visual isometry probe: matching cluster topologies across panels mean V
    and A share relational structure (GW has signal to align). Mismatched
    topologies mean GW is forcing alignment that isn't natively there --
    a signal that the chosen video/audio encoders don't share geometry and
    that swapping encoders may matter more than tuning GW.
    """
    # subsample for speed
    n = min(len(V), len(A), max_points)
    rng = np.random.default_rng(0)
    if len(V) > n:
        idx = rng.choice(len(V), n, replace=False)
        V = V[idx]; A = A[idx]
        labels = [labels[i] for i in idx]

    V_2d, method = _2d_embedding(V)
    A_2d, _ = _2d_embedding(A)

    # pick top-N frequent classes; everything else goes gray.
    counts = defaultdict(int)
    for c in labels:
        counts[c] += 1
    top = [c for c, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n_classes]]
    cmap = plt.get_cmap('tab20', len(top))
    color_of = {c: cmap(i) for i, c in enumerate(top)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, X2, side in zip(axes, [V_2d, A_2d], ['video', 'audio']):
        # gray background points (everything not in top-N)
        gray_mask = np.array([labels[i] not in color_of for i in range(len(labels))])
        if gray_mask.any():
            ax.scatter(X2[gray_mask, 0], X2[gray_mask, 1],
                       s=6, c='lightgray', alpha=0.4, linewidths=0)
        for c in top:
            mask = np.array([labels[i] == c for i in range(len(labels))])
            if not mask.any():
                continue
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       s=10, c=[color_of[c]], alpha=0.85, linewidths=0,
                       label=str(c))
        ax.set_title(f'{side} ({method})')
        ax.set_xticks([]); ax.set_yticks([])

    suptitle = f'{variant}: paired {method} of V and A'
    if cka is not None and proc is not None and not (np.isnan(cka) or np.isnan(proc)):
        suptitle += f'   |   CKA={cka:.3f}   Procrustes resid={proc:.3f}'
    fig.suptitle(suptitle, fontsize=11)

    # single legend across both axes
    handles, lbls = axes[0].get_legend_handles_labels()
    if handles:
        # truncate long labels for legibility
        short = [str(l)[:28] + ('…' if len(str(l)) > 28 else '') for l in lbls]
        fig.legend(handles, short, loc='lower center', ncol=min(6, len(handles)),
                   fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return method


def plot_gw_perm(rows, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    variants = [r['variant'] for r in rows]
    x = np.arange(len(variants))
    w = 0.35
    ax.bar(x - w/2, [r['gw_paired_mean'] for r in rows], width=w, label='paired')
    ax.bar(x + w/2, [r['gw_shuffled_mean'] for r in rows], width=w, label='shuffled')
    for i, r in enumerate(rows):
        ax.text(i, max(r['gw_paired_mean'], r['gw_shuffled_mean']),
                f"z={r['gw_perm_z']:.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(variants)
    ax.set_ylabel('GW loss')
    ax.set_title('GW(V, A) vs GW(V, shuffle(A)) — tight gap = no relational signal')
    ax.legend(); ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    reps_by_variant = {}
    for v in VARIANTS:
        V, A, labels = gather_reps(net, dset, v, args.max_samples)
        reps_by_variant[v] = (V, A)
        r_p, r_s = plot_scatter(V, A, v, out / f'scatter_{v}.png')
        scatter_info[v] = (r_p, r_s)

        knn_by_variant[v] = {}
        for k in args.ks:
            knn_by_variant[v][k] = knn_jaccard(V, A, k)

        gw_mean, gw_std = gw_value(V, A, fused=(v == 'fused'))
        gw_paired, gw_shuffled, gw_z = gw_permutation_test(V, A, fused=(v == 'fused'))
        tau = class_kendall_tau(V, A, labels)
        er_v = effective_rank(V)
        er_a = effective_rank(A)
        cka = linear_cka(V, A)
        proc = procrustes_residual(V, A)
        plot_paired_umap(
            V, A, labels, v, out / f'umap_{v}.png',
            cka=cka, proc=proc,
        )
        plot_pairwise_dist_overlay(V, A, v, out / f'dist_overlay_{v}.png')
        plot_shepard(V, A, v, out / f'shepard_{v}.png')
        plot_mantel_profile(V, A, v, out / f'mantel_{v}.png')
        plot_procrustes_umap_overlay(
            V, A, labels, v, out / f'procrustes_umap_{v}.png',
        )
        plot_betti0_overlay(V, A, v, out / f'betti0_{v}.png')
        rows.append({
            'variant': v,
            'pearson_DV_DA': r_p,
            'spearman_DV_DA': r_s,
            'gw_init_mean': gw_mean,
            'gw_init_std': gw_std,
            'gw_paired_mean': gw_paired,
            'gw_shuffled_mean': gw_shuffled,
            'gw_perm_z': gw_z,
            'linear_cka': cka,
            'procrustes_resid': proc,
            'class_kendall_tau': tau,
            'erank_video': er_v,
            'erank_audio': er_a,
            **{f'jaccard_knn@{k}': knn_by_variant[v][k] for k in args.ks},
        })

    df = pd.DataFrame(rows)
    df.to_csv(out / 'metrics.csv', index=False)
    df.to_latex(out / 'metrics.tex', index=False, float_format='%.3f')

    plot_knn_curve({'ks': args.ks, 'knn': knn_by_variant}, out / 'knn.png')
    plot_svd_spectra(reps_by_variant, out / 'svd_spectra.png')
    plot_procrustes_cka_bar(rows, out / 'cka_procrustes.png')
    plot_gw_perm(rows, out / 'gw_permutation.png')
    diag = plot_coupling_heatmap(net, dset, out / 'coupling.png', variant='global')

    print(df.to_string(index=False))
    print(f'Coupling diagonal mass (global): {diag:.3f}')
    print(f'Wrote diagnostics to {out}')


if __name__ == '__main__':
    main()

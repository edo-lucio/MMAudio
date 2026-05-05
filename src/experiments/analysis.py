"""Analysis utilities for GW experiments.

Subcommands:
  curves    - plot FM loss and GW loss curves from TensorBoard logs
  table     - LaTeX table of eval metrics across experiments
  perclass  - per-class IB-score comparison baseline vs GW (histogram)
  geometry  - effective rank, singular spectrum, t-SNE of reps
               for baseline vs GW model

Usage:
  python experiments/analysis.py curves  --runs output/gw_var_*
  python experiments/analysis.py table   --runs output/gw_var_* --out table.tex
  python experiments/analysis.py perclass --baseline output/gw_baseline \\
         --gw output/gw_var_global --out perclass.png
  python experiments/analysis.py geometry --baseline output/gw_baseline/gw_baseline_ema_final.pth \\
         --gw output/gw_var_global/gw_var_global_ema_final.pth
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------- shared helpers ----------

def read_scalars(run_dir: Path, tags: list[str]) -> dict[str, pd.DataFrame]:
    """Aggregate scalar events across every events.out.tfevents.* file in
    run_dir (and any subdirectory). Resumed / interrupted runs accumulate
    multiple event files in the same logdir; reading only the latest one
    silently drops the rest of the training history.

    For each requested tag we concatenate the scalar series from every file,
    sort by step, and drop duplicate (step, tag) entries keeping the most
    recently written value. Returns {tag: DataFrame[step, tag]}.
    """
    ev_files = sorted(run_dir.rglob('events.out.tfevents.*'))
    if not ev_files:
        return {}

    # Group event files by their parent directory and let EventAccumulator
    # merge files within each logdir. This handles the common case where all
    # event files sit directly under run_dir.
    logdirs = sorted({f.parent for f in ev_files})

    per_tag_frames: dict[str, list[pd.DataFrame]] = {tag: [] for tag in tags}
    for logdir in logdirs:
        ea = EventAccumulator(str(logdir))
        ea.Reload()
        scalar_tags = ea.Tags().get('scalars', [])
        for tag in tags:
            if tag in scalar_tags:
                rows = [(e.wall_time, e.step, e.value) for e in ea.Scalars(tag)]
                df = pd.DataFrame(rows, columns=['wall_time', 'step', tag])
                per_tag_frames[tag].append(df)

    out: dict[str, pd.DataFrame] = {}
    for tag, frames in per_tag_frames.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        # If the same step was written by two sessions (e.g. resume from
        # checkpoint replays a few steps), keep the latest by wall_time.
        df = df.sort_values('wall_time').drop_duplicates(
            subset='step', keep='last'
        ).sort_values('step').reset_index(drop=True)
        out[tag] = df[['step', tag]]
    return out


# ---------- curves ----------

def cmd_curves(args):
    out = Path(args.out or 'analysis/curves.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for run in args.runs:
        run = Path(run)
        sc = read_scalars(run, ['train/loss', 'train/gw_loss'])
        if 'train/loss' in sc:
            axes[0].plot(sc['train/loss']['step'], sc['train/loss']['train/loss'], label=run.name)
        if 'train/gw_loss' in sc:
            axes[1].plot(sc['train/gw_loss']['step'], sc['train/gw_loss']['train/gw_loss'],
                         label=run.name)
    axes[0].set_title('Flow matching loss'); axes[0].legend(fontsize=7); axes[0].set_xlabel('step')
    axes[1].set_title('GW loss'); axes[1].legend(fontsize=7); axes[1].set_xlabel('step')
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print(f'Wrote {out}')


# ---------- metrics table ----------

METRIC_TAGS = ['test/fd_passt', 'test/inception_score', 'test/ib_score', 'test/desync']


def cmd_table(args):
    rows = []
    for run in args.runs:
        run = Path(run)
        sc = read_scalars(run, METRIC_TAGS)
        row = {'run': run.name}
        for t in METRIC_TAGS:
            if t in sc:
                row[t.split('/')[-1]] = sc[t].iloc[-1][t]
        rows.append(row)
    df = pd.DataFrame(rows)
    if args.out:
        df.to_latex(args.out, index=False, float_format='%.3f')
        print(f'Wrote {args.out}')
    else:
        print(df.to_string(index=False))


# ---------- per-class IB-score comparison ----------

def cmd_perclass(args):
    """Expects per-class IB-score dumps in each run dir: perclass_ib.json = {class: score}."""
    base = json.loads((Path(args.baseline) / 'perclass_ib.json').read_text())
    gw = json.loads((Path(args.gw) / 'perclass_ib.json').read_text())
    classes = sorted(set(base) & set(gw))
    delta = [gw[c] - base[c] for c in classes]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(delta, bins=40, color='steelblue')
    ax.axvline(0, ls='--', color='k')
    ax.set_xlabel('IB-score(GW) - IB-score(baseline)')
    ax.set_ylabel('# classes')
    ax.set_title(f'Per-class IB improvement (mean={np.mean(delta):.3f})')
    fig.tight_layout()
    out = Path(args.out or 'analysis/perclass.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    # also dump ranked CSV
    df = pd.DataFrame({'class': classes, 'delta': delta}).sort_values('delta', ascending=False)
    df.to_csv(out.with_suffix('.csv'), index=False)
    print(f'Wrote {out} and {out.with_suffix(".csv")}')


# ---------- representation geometry ----------

def effective_rank(S: np.ndarray, eps: float = 1e-12) -> float:
    p = S / (S.sum() + eps)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def safe_svd(X: np.ndarray) -> np.ndarray:
    """SVD with NaN/Inf sanitisation and a gesdd→gesvd fallback.

    Collapsed reps after a failed GW run frequently produce non-finite values
    or near-degenerate matrices that trip the default gesdd LAPACK driver
    with "SVD did not converge". gesvd is slower but numerically robust.
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return np.linalg.svd(X, compute_uv=False)
    except np.linalg.LinAlgError:
        try:
            from scipy.linalg import svd as scipy_svd
            return scipy_svd(X, compute_uv=False, lapack_driver='gesvd')
        except Exception:
            # Last-resort: SVD of X^T X, take sqrt of eigenvalues.
            G = X.T @ X if X.shape[0] >= X.shape[1] else X @ X.T
            w = np.linalg.eigvalsh(G)
            w = np.clip(w, 0.0, None)
            return np.sqrt(w[::-1])


def _upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def _linear_cka(V: np.ndarray, A: np.ndarray) -> float:
    Vc = V - V.mean(0, keepdims=True)
    Ac = A - A.mean(0, keepdims=True)
    KV = Vc @ Vc.T
    KA = Ac @ Ac.T
    hsic = (KV * KA).sum()
    denom = np.sqrt((KV * KV).sum() * (KA * KA).sum()) + 1e-12
    return float(hsic / denom)


def _procrustes_residual(V: np.ndarray, A: np.ndarray) -> float:
    Vc = V - V.mean(0, keepdims=True)
    Ac = A - A.mean(0, keepdims=True)
    if Vc.shape[1] != Ac.shape[1]:
        d = max(Vc.shape[1], Ac.shape[1])
        Vp = np.zeros((Vc.shape[0], d)); Vp[:, :Vc.shape[1]] = Vc
        Ap = np.zeros((Ac.shape[0], d)); Ap[:, :Ac.shape[1]] = Ac
        Vc, Ac = Vp, Ap
    M = Ac.T @ Vc
    try:
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd
        U, _, Vt = scipy_svd(M, full_matrices=False, lapack_driver='gesvd')
    R = U @ Vt
    resid = np.linalg.norm(Vc @ R.T - Ac, ord='fro') ** 2
    denom = np.linalg.norm(Ac, ord='fro') ** 2 + 1e-12
    return float(resid / denom)


def _pairwise_sq_l2(X: np.ndarray) -> np.ndarray:
    sq = (X * X).sum(-1, keepdims=True)
    D = sq + sq.T - 2.0 * (X @ X.T)
    return np.clip(D, 0.0, None)


def _knn_jaccard(V: np.ndarray, A: np.ndarray, k: int) -> float:
    DV = _pairwise_sq_l2(V); DA = _pairwise_sq_l2(A)
    np.fill_diagonal(DV, np.inf); np.fill_diagonal(DA, np.inf)
    nn_v = np.argsort(DV, axis=1)[:, :k]
    nn_a = np.argsort(DA, axis=1)[:, :k]
    jacc = []
    for i in range(len(V)):
        sv, sa = set(nn_v[i].tolist()), set(nn_a[i].tolist())
        u = len(sv | sa)
        jacc.append(len(sv & sa) / u if u else 0.0)
    return float(np.mean(jacc))


def _gw_permutation_numpy(V: np.ndarray, A: np.ndarray,
                          n_perm: int = 20, batch: int = 64,
                          epsilon: float = 0.1, num_iter: int = 5):
    """Lightweight numpy GW permutation test.

    Mirrors the entropic GW objective from mmaudio.model.gw_regularization
    using a small-batch Sinkhorn loop so we don't drag the full training
    stack into post-hoc analysis.
    """
    n = len(V)
    if n < batch * 2:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(0)

    def _normalize(D):
        m = max(D.mean(), 1e-8)
        return D / m

    def _gw(v, a):
        DV = _normalize(_pairwise_sq_l2(v))
        DA = _normalize(_pairwise_sq_l2(a))
        B = DV.shape[0]
        p = np.full(B, 1.0 / B); q = np.full(B, 1.0 / B)
        T = np.outer(p, q)
        for _ in range(num_iter):
            C = -2.0 * DV @ T @ DA.T
            C = C - C.max()
            K = np.exp(-C / epsilon)
            u = np.ones(B); v_s = np.ones(B)
            for _ in range(20):
                u = p / (K @ v_s + 1e-30)
                v_s = q / (K.T @ u + 1e-30)
            T = u[:, None] * K * v_s[None, :]
        return _frob_quad(DV, DA, T)

    def _frob_quad(DV, DA, T):
        # <L(DV,DA) ⊗ T, T>_F  with L(a,b) = (a-b)^2.
        # Reduce to:  sum DV^2 p p^T + q^T DA^2 q  - 2 tr(DV T DA T^T)
        a = (DV * DV).sum(axis=1)
        b = (DA * DA).sum(axis=1)
        p = T.sum(axis=1); q = T.sum(axis=0)
        c = a @ p + b @ q - 2.0 * (DV @ T @ DA.T * T).sum()
        return float(c)

    paired, shuffled = [], []
    for _ in range(n_perm):
        idx = rng.choice(n, size=batch, replace=False)
        v = V[idx]; a = A[idx]
        paired.append(_gw(v, a))
        perm = rng.permutation(batch)
        shuffled.append(_gw(v, a[perm]))
    paired = np.array(paired); shuffled = np.array(shuffled)
    z = (shuffled.mean() - paired.mean()) / (paired.std() + shuffled.std() + 1e-12)
    return float(paired.mean()), float(shuffled.mean()), float(z)


def cmd_geometry(args):
    """Computes SVD-based metrics from saved-out projected features.
    Expects each run to have <run>/gw_features.pt = dict(video=(N,D), audio=(N,D), labels=list)."""
    out = Path(args.out or 'analysis/geometry.png')
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    summary = {}
    reps = {}  # tag -> {'video': V, 'audio': A}
    for tag, path in [('baseline', args.baseline), ('gw', args.gw)]:
        feats = torch.load(path, map_location='cpu', weights_only=True)
        reps[tag] = {}
        for mod, key in [('video', 'video'), ('audio', 'audio')]:
            X = feats[key].numpy().astype(np.float64, copy=False)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            reps[tag][mod] = X
            S = safe_svd(X)
            if S.size == 0 or S[0] <= 0.0:
                summary[f'{tag}_{mod}_erank'] = 0.0
                continue
            summary[f'{tag}_{mod}_erank'] = effective_rank(S)
            ax = axes[0 if mod == 'video' else 1]
            ax.semilogy(S / S[0], label=f'{tag}')
            ax.set_title(f'{mod} singular spectrum (normalized)')
            ax.set_xlabel('index')

    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    # ── post-hoc cross-modal geometry probes (CKA, Procrustes, GW perm, kNN) ──
    probes = {}
    for tag in ('baseline', 'gw'):
        V, A = reps[tag]['video'], reps[tag]['audio']
        n = min(len(V), len(A))
        V, A = V[:n], A[:n]
        if n < 8:
            continue
        try:
            cka = _linear_cka(V, A)
        except Exception:
            cka = float('nan')
        try:
            proc = _procrustes_residual(V, A)
        except Exception:
            proc = float('nan')
        try:
            knn10 = _knn_jaccard(V, A, k=min(10, n - 1))
        except Exception:
            knn10 = float('nan')
        try:
            gw_p, gw_s, gw_z = _gw_permutation_numpy(V, A, batch=min(64, n // 2))
        except Exception:
            gw_p = gw_s = gw_z = float('nan')
        probes[tag] = dict(
            linear_cka=cka, procrustes_resid=proc, knn_jaccard_at10=knn10,
            gw_paired=gw_p, gw_shuffled=gw_s, gw_perm_z=gw_z,
        )
        summary.update({f'{tag}_{k}': v for k, v in probes[tag].items()})

    if probes:
        # CKA + Procrustes bar chart (baseline vs gw)
        out_cka = out.parent / 'cka_procrustes.png'
        fig, ax = plt.subplots(figsize=(7, 4))
        tags = list(probes.keys())
        x = np.arange(len(tags)); w = 0.35
        ax.bar(x - w/2, [probes[t]['linear_cka'] for t in tags], width=w, label='linear CKA')
        ax.bar(x + w/2, [probes[t]['procrustes_resid'] for t in tags], width=w, label='Procrustes residual')
        ax.set_xticks(x); ax.set_xticklabels(tags)
        ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis='y')
        ax.set_title('CKA (↑ aligned) vs Procrustes residual (↓ aligned)')
        ax.legend()
        fig.tight_layout(); fig.savefig(out_cka, dpi=150); plt.close(fig)

        # GW permutation bar chart
        out_perm = out.parent / 'gw_permutation.png'
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w/2, [probes[t]['gw_paired'] for t in tags], width=w, label='paired')
        ax.bar(x + w/2, [probes[t]['gw_shuffled'] for t in tags], width=w, label='shuffled')
        for i, t in enumerate(tags):
            top = max(probes[t]['gw_paired'], probes[t]['gw_shuffled'])
            ax.text(i, top, f"z={probes[t]['gw_perm_z']:.2f}",
                    ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(tags); ax.grid(alpha=0.3, axis='y')
        ax.set_title('GW(V, A) vs GW(V, shuffle(A)) — tight gap = no relational signal')
        ax.legend()
        fig.tight_layout(); fig.savefig(out_perm, dpi=150); plt.close(fig)

    # JSON sidecar so the pipeline can pick up the numbers without re-parsing stdout.
    with open(out.parent / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f'Wrote {out}')
    if probes:
        print(f'Wrote {out.parent / "cka_procrustes.png"}, {out.parent / "gw_permutation.png"}')
    print(f'Wrote {out.parent / "metrics.json"}')


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    pc = sub.add_parser('curves'); pc.add_argument('--runs', nargs='+', required=True)
    pc.add_argument('--out', default=None)

    pt = sub.add_parser('table'); pt.add_argument('--runs', nargs='+', required=True)
    pt.add_argument('--out', default=None)

    pp = sub.add_parser('perclass')
    pp.add_argument('--baseline', required=True); pp.add_argument('--gw', required=True)
    pp.add_argument('--out', default=None)

    pg = sub.add_parser('geometry')
    pg.add_argument('--baseline', required=True); pg.add_argument('--gw', required=True)
    pg.add_argument('--out', default=None)

    args = p.parse_args()
    dict(curves=cmd_curves, table=cmd_table, perclass=cmd_perclass, geometry=cmd_geometry)[args.cmd](args)


if __name__ == '__main__':
    main()

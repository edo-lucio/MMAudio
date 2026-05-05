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


def cmd_geometry(args):
    """Computes SVD-based metrics from saved-out projected features.
    Expects each run to have <run>/gw_features.pt = dict(video=(N,D), audio=(N,D), labels=list)."""
    out = Path(args.out or 'analysis/geometry.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    summary = {}
    for tag, path in [('baseline', args.baseline), ('gw', args.gw)]:
        feats = torch.load(path, map_location='cpu', weights_only=True)
        for mod, key in [('video', 'video'), ('audio', 'audio')]:
            X = feats[key].numpy()
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

    print(json.dumps(summary, indent=2))
    print(f'Wrote {out}')


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

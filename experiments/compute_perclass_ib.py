"""Per-class similarity scores for a trained run, written to <run>/perclass_ib.json.

For each test clip, we already have:
    pred PANN features:  <run>/test-output/cache/pann_features.pth
    GT   PANN features:  <gt_cache>/pann_features.pth

This script computes per-clip cosine similarity in PANN-embedding space, groups
by VGGSound class label (read from the test tsv), means, and dumps:
    {class_label: mean_cosine_sim}

It's the proxy `cmd_perclass` reads — higher = predicted audio is more similar
to GT in the audio-tagging feature space, per class.

Usage:
    python experiments/compute_perclass_ib.py \\
        --run output/gw_var_global \\
        --gt_cache ./data/eval-cache/vggsound-test-eval-cache \\
        --tsv training/example_output/memmap/vgg-test.tsv
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def load_features(path: Path) -> dict[str, torch.Tensor]:
    """av_bench writes a dict mapping clip_id -> feature tensor."""
    return torch.load(path, map_location='cpu', weights_only=True)


def read_id_to_label(tsv_path: Path) -> dict[str, str]:
    """tsv columns: id\\tlabel\\t..."""
    out = {}
    with open(tsv_path) as f:
        header = f.readline().strip().split('\t')
        if 'id' not in header:
            raise ValueError(f'Expected `id` column in {tsv_path}, got {header}')
        id_idx = header.index('id')
        # try common label column names
        label_idx = None
        for cand in ('label', 'caption', 'class', 'category'):
            if cand in header:
                label_idx = header.index(cand)
                break
        if label_idx is None:
            label_idx = 1  # fall back to 2nd column
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(id_idx, label_idx):
                continue
            out[parts[id_idx]] = parts[label_idx]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, type=Path,
                   help='output/<exp_id> directory')
    p.add_argument('--gt_cache', required=True, type=Path,
                   help='Directory containing GT pann_features.pth')
    p.add_argument('--tsv', required=True, type=Path,
                   help='Test split tsv (for id -> class mapping)')
    p.add_argument('--out', type=Path, default=None,
                   help='Output JSON path (default: <run>/perclass_ib.json)')
    args = p.parse_args()

    out_path = args.out or (args.run / 'perclass_ib.json')

    # av_bench writes the predicted-side cache at <audio_dir>/cache where
    # audio_dir = <run>/<tag>-sampled (tag='test' for ExtractedVGG_test).
    pred_candidates = [
        args.run / 'test-sampled' / 'cache' / 'pann_features.pth',
        args.run / 'test-output' / 'cache' / 'pann_features.pth',
    ]
    pred_path = next((p for p in pred_candidates if p.is_file()), None)
    gt_path = args.gt_cache / 'pann_features.pth'
    if pred_path is None:
        raise FileNotFoundError(
            f'No predicted PANN features under {args.run}. '
            f'Looked at: {[str(p) for p in pred_candidates]}. '
            f'Run eval_run.py / eval_all.job first.')
    if not gt_path.is_file():
        raise FileNotFoundError(f'Missing GT PANN features at {gt_path}.')

    pred = load_features(pred_path)
    gt = load_features(gt_path)
    id_to_label = read_id_to_label(args.tsv)

    common_ids = sorted(set(pred) & set(gt) & set(id_to_label))
    if not common_ids:
        raise RuntimeError(
            f'No clip ids overlap between predictions ({len(pred)}), '
            f'GT cache ({len(gt)}), and tsv labels ({len(id_to_label)}). '
            f'Check that ids in {pred_path} match the GT cache and tsv.')

    per_class = defaultdict(list)
    for cid in common_ids:
        v_pred = pred[cid].flatten().float()
        v_gt = gt[cid].flatten().float()
        # cosine similarity in PANN-embedding space (proxy for IB-score)
        sim = F.cosine_similarity(v_pred.unsqueeze(0), v_gt.unsqueeze(0)).item()
        per_class[id_to_label[cid]].append(sim)

    summary = {cls: float(sum(v) / len(v)) for cls, v in per_class.items()}
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f'Wrote {out_path}  classes={len(summary)}  clips={len(common_ids)}')


if __name__ == '__main__':
    main()

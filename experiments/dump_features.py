"""Dump (video, audio) reps for a trained run, writing <out>/gw_features.pt.

Used by experiments/analysis.py geometry, which expects:
    <path>/gw_features.pt = dict(video=(N,D), audio=(N,D), labels=list[str])

Usage:
    python experiments/dump_features.py \\
        --weights output/gw_var_global/gw_var_global_ema_final.pth \\
        --variant global \\
        --out     output/gw_var_global/gw_features.pt
"""
import argparse
from pathlib import Path

import torch
from hydra import compose, initialize

from mmaudio.data.extracted_vgg import ExtractedVGG
from mmaudio.model.gw_regularization import _extract_representations
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
    p.add_argument('--variant', default='global',
                   choices=['global', 'projected', 'c_g', 'fused'])
    p.add_argument('--out', required=True,
                   help='Path to output gw_features.pt (e.g. output/<run>/gw_features.pt)')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--max_samples', type=int, default=4000)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    n = min(len(dset), args.max_samples)
    v_chunks, a_chunks, labels = [], [], []

    for start in range(0, n, args.batch_size):
        end = min(start + args.batch_size, n)
        rows = [dset[i] for i in range(start, end)]
        labels.extend([r['caption'] for r in rows])

        clip_f = torch.stack([r['clip_features'] for r in rows]).cuda().float()
        a_mean = torch.stack([r['a_mean'] for r in rows]).cuda().float()
        a_norm = net.normalize(a_mean)

        v, a = _extract_representations(net, args.variant, clip_f, a_norm)
        v_chunks.append(v.detach().cpu())
        a_chunks.append(a.detach().cpu())

    V = torch.cat(v_chunks, dim=0)
    A = torch.cat(a_chunks, dim=0)
    torch.save({'video': V, 'audio': A, 'labels': labels}, out_path)
    print(f'Wrote {out_path}  video={tuple(V.shape)}  audio={tuple(A.shape)}  N={len(labels)}')


if __name__ == '__main__':
    main()

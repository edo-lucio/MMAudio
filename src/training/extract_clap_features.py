"""Pre-extract CLAP audio embeddings for the V2A training corpus.

Why this script exists. The default audio side of the GW regulariser is
the VAE latent x_1 (a 250 x 20 tensor at 31.25 fps), which is shaped by
audio reconstruction only and has no text-aware semantic structure. CLIP
visual features, by contrast, were pretrained against text. The
hypothesis tested by `gw_regularization.audio_encoder = clap` is that
swapping the audio side of GW for an embedding produced by a *text-aware*
audio encoder (LAION-CLAP) lets the GW regulariser align two
representations that share the same anchor (natural-language semantics),
so the relational geometry of the two sides should be more easily put
into correspondence.

Output. One memmap shard per split, keyed by clip id, holding a single
512-d CLAP audio embedding per clip:

    output_dir/
        clap_features.memmap         # shape: (N, 512), dtype float32
        clap_features.tsv            # columns: idx, clip_id

The dataloader extension (in mmaudio.data_mod) will load these alongside
the existing CLIP / Synchformer / VAE-latent features when the CLAP
ablation is selected at training time.

Requires:  pip install laion-clap soundfile torchaudio

Usage example:

    torchrun --standalone --nproc_per_node=1 \
        src/training/extract_clap_features.py \
        --audio_dir   ./data/audiocaps/audio \
        --captions_tsv ./data/audiocaps/audiocaps-train.tsv \
        --output_dir  ./data/v1-16-memmap/audiocaps-train-clap
"""

import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

log = logging.getLogger(__name__)

# CLAP defaults: 48 kHz mono input, 10 s window, 512-d output embedding.
CLAP_SR = 48_000
CLAP_WINDOW_SEC = 10
CLAP_EMBED_DIM = 512


def _load_clap(model_id: str, ckpt_path: str | None, device: torch.device):
    """Load LAION-CLAP and put it in eval mode.

    Two ways to obtain CLAP weights, in order of preference:
      1. Local checkpoint (`ckpt_path` argument). Useful on offline clusters.
      2. Hugging Face / laion_clap auto-download (may need internet on the
         compute node).
    """
    try:
        import laion_clap
    except ImportError as e:
        raise RuntimeError(
            "laion-clap is not installed. Run `pip install laion-clap` in "
            "the training environment, then resubmit."
        ) from e

    # 'HTSAT-base' is the audio backbone used by the public CLAP releases;
    # 'roberta' is the text backbone but we never call the text tower here.
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    if ckpt_path is not None and Path(ckpt_path).is_file():
        log.info(f"loading CLAP weights from local: {ckpt_path}")
        model.load_ckpt(ckpt_path)
    else:
        log.info(f"loading CLAP weights via laion_clap auto-download: {model_id}")
        model.load_ckpt(model_id=model_id)
    model.eval()
    model.to(device)
    return model


def _read_clip(audio_path: Path) -> torch.Tensor:
    """Load a wav, downmix to mono, resample to CLAP_SR, pad/truncate to 10 s."""
    wav, sr = torchaudio.load(str(audio_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != CLAP_SR:
        wav = torchaudio.functional.resample(wav, sr, CLAP_SR)
    target = CLAP_SR * CLAP_WINDOW_SEC
    if wav.shape[1] < target:
        wav = torch.nn.functional.pad(wav, (0, target - wav.shape[1]))
    elif wav.shape[1] > target:
        wav = wav[:, :target]
    return wav.squeeze(0).contiguous()                        # (N,)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument('--audio_dir',    type=Path, required=True,
                        help='dir containing {clip_id}.wav files')
    parser.add_argument('--captions_tsv', type=Path, required=True,
                        help='manifest with `id` column (other columns ignored)')
    parser.add_argument('--output_dir',   type=Path, required=True,
                        help='dest dir for clap_features.memmap + tsv')
    parser.add_argument('--clap_model_id', type=str,
                        default='630k-audioset-best.pt',
                        help='CLAP checkpoint for laion_clap.load_ckpt()')
    parser.add_argument('--clap_ckpt_path', type=str, default=None,
                        help='local path to CLAP weights; if set, takes '
                             'precedence over --clap_model_id')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = _load_clap(args.clap_model_id, args.clap_ckpt_path, device)

    df = pd.read_csv(args.captions_tsv, sep='\t')
    if 'id' not in df.columns:
        raise SystemExit(f'captions_tsv {args.captions_tsv} has no `id` column')
    clip_ids = df['id'].tolist()
    log.info(f"manifest: {len(clip_ids)} clips")

    # Pre-allocate memmap.
    memmap_path = args.output_dir / 'clap_features.memmap'
    arr = np.memmap(memmap_path, dtype=np.float32, mode='w+',
                    shape=(len(clip_ids), CLAP_EMBED_DIM))

    valid_rows: list[tuple[int, str]] = []
    failed: list[str] = []
    cursor = 0

    pbar = tqdm(total=len(clip_ids), desc='extract', dynamic_ncols=True)
    while cursor < len(clip_ids):
        batch_ids = clip_ids[cursor:cursor + args.batch_size]
        wavs: list[torch.Tensor] = []
        kept_ids: list[str] = []
        for cid in batch_ids:
            path = args.audio_dir / f'{cid}.wav'
            if not path.exists():
                failed.append(cid)
                continue
            try:
                wavs.append(_read_clip(path))
                kept_ids.append(cid)
            except Exception as exc:                                  # noqa
                log.warning(f"skip {cid}: {exc}")
                failed.append(cid)

        if wavs:
            wav_batch = torch.stack(wavs).to(device)                  # (B, N)
            with torch.inference_mode():
                emb = model.get_audio_embedding_from_data(
                    wav_batch, use_tensor=True
                )                                                      # (B, 512)
            emb_np = emb.detach().float().cpu().numpy()
            for cid, vec in zip(kept_ids, emb_np):
                arr[len(valid_rows)] = vec
                valid_rows.append((len(valid_rows), cid))

        cursor += args.batch_size
        pbar.update(len(batch_ids))
    pbar.close()

    # Trim memmap to the rows we actually wrote.
    arr.flush()
    n_valid = len(valid_rows)
    if n_valid < len(clip_ids):
        log.info(f"trimming memmap from {len(clip_ids)} → {n_valid} rows")
        trimmed_path = memmap_path.with_suffix('.trimmed')
        trimmed = np.memmap(trimmed_path, dtype=np.float32, mode='w+',
                            shape=(n_valid, CLAP_EMBED_DIM))
        trimmed[:] = arr[:n_valid]
        trimmed.flush()
        del trimmed, arr
        os.replace(trimmed_path, memmap_path)

    # Sidecar TSV: row index ↔ clip id.
    side_tsv = args.output_dir / 'clap_features.tsv'
    pd.DataFrame(valid_rows, columns=['idx', 'id']).to_csv(
        side_tsv, sep='\t', index=False
    )

    log.info(f"wrote {n_valid} embeddings → {memmap_path}")
    log.info(f"sidecar manifest          → {side_tsv}")
    if failed:
        log.warning(f"skipped {len(failed)} clips (missing or unreadable). "
                    f"first 5: {failed[:5]}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

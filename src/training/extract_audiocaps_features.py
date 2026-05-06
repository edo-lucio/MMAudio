"""Pre-extract CLAP audio + CLIP-text caption embeddings for AudioCaps.

Why this script exists. The cross-dataset FGW variant (Reading B) needs
AudioCaps clips to live in the same CLAP-audio space as VGGSound clips and
the same CLIP-text space as VGGSound class labels. This script does both
in one pass and writes a single TensorDict memmap that the
ExtractedAudioCaps dataset class loads at training time.

Output layout (one shard per split):

    output_dir/
        meta.memmap/                          # TensorDict mmap directory
            clap_features    : (N, 512)       float32
            text_features    : (N, T_seq, T_dim)  float32
        ids.tsv                               # idx<TAB>id<TAB>caption

The text encoder used here MUST be identical to the one used to extract
VGGSound's `text_features` (CLIP-text via FeaturesUtils.encode_text); the
class-label-vs-caption cosine in the FGW cross-cost is only meaningful if
the two text axes live in the same space.

Usage:

    python src/training/extract_audiocaps_features.py \
        --audio_dir   ./data/audiocaps/audio \
        --captions_tsv ./data/audiocaps/audiocaps-train.tsv \
        --output_dir  ./data/audiocaps/extracted/train \
        --vae_ckpt    ext_weights/v1-16.pth \
        --vocoder_ckpt ext_weights/best_netG.pt \
        --synchformer_ckpt ext_weights/synchformer_state_dict.pth \
        --clap_ckpt   ext_weights/laion_clap.pt
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tensordict import TensorDict
from tqdm import tqdm

from mmaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger(__name__)

CLAP_SR = 48_000
CLAP_WINDOW_SEC = 10
CLAP_EMBED_DIM = 512


def _load_clap(ckpt_path: str | None, model_id: str, device: torch.device):
    try:
        import laion_clap
    except ImportError as e:
        raise RuntimeError(
            "laion-clap not installed in this env."
        ) from e
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    if ckpt_path and Path(ckpt_path).is_file():
        model.load_ckpt(ckpt_path)
    else:
        model.load_ckpt(model_id=model_id)
    model.eval().to(device)
    return model


def _read_clip(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != CLAP_SR:
        wav = torchaudio.functional.resample(wav, sr, CLAP_SR)
    target = CLAP_SR * CLAP_WINDOW_SEC
    if wav.shape[1] < target:
        wav = torch.nn.functional.pad(wav, (0, target - wav.shape[1]))
    elif wav.shape[1] > target:
        wav = wav[:, :target]
    return wav.squeeze(0).contiguous()


def main() -> int:
    p = ArgumentParser()
    p.add_argument('--audio_dir',     type=Path, required=True)
    p.add_argument('--captions_tsv',  type=Path, required=True)
    p.add_argument('--output_dir',    type=Path, required=True)
    p.add_argument('--vae_ckpt',      type=str, required=True)
    p.add_argument('--vocoder_ckpt',  type=str, required=True)
    p.add_argument('--synchformer_ckpt', type=str, required=True)
    p.add_argument('--clap_ckpt',     type=str, default=None)
    p.add_argument('--clap_model_id', type=str,
                   default='630k-audioset-best.pt')
    p.add_argument('--mode',          type=str, default='16k',
                   choices=['16k', '44k'])
    p.add_argument('--audio_batch',   type=int, default=16)
    p.add_argument('--text_batch',    type=int, default=64)
    p.add_argument('--device',        type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    df = pd.read_csv(args.captions_tsv, sep='\t')
    if 'id' not in df.columns or 'caption' not in df.columns:
        raise SystemExit(
            f'{args.captions_tsv} must have id and caption columns'
        )
    rows = df.to_dict('records')
    log.info(f'manifest: {len(rows)} clips')

    # Filter to clips actually on disk (download may have stopped early).
    on_disk = []
    for r in rows:
        if (args.audio_dir / f'{r["id"]}.wav').is_file():
            on_disk.append(r)
        else:
            log.warning(f'missing wav: {r["id"]}')
    log.info(f'on disk: {len(on_disk)}')
    if not on_disk:
        raise SystemExit('no audio files found, nothing to extract')

    # ── CLAP audio embeddings ────────────────────────────────────────────────
    clap = _load_clap(args.clap_ckpt, args.clap_model_id, device)
    audio_emb = torch.zeros((len(on_disk), CLAP_EMBED_DIM), dtype=torch.float32)
    pbar = tqdm(total=len(on_disk), desc='clap audio', dynamic_ncols=True)
    for i in range(0, len(on_disk), args.audio_batch):
        batch_rows = on_disk[i:i + args.audio_batch]
        wavs = []
        idxs = []
        for j, r in enumerate(batch_rows):
            try:
                wavs.append(_read_clip(args.audio_dir / f'{r["id"]}.wav'))
                idxs.append(i + j)
            except Exception as exc:                                  # noqa
                log.warning(f'audio decode skip {r["id"]}: {exc}')
        if wavs:
            wav_batch = torch.stack(wavs).to(device)
            with torch.inference_mode():
                emb = clap.get_audio_embedding_from_data(
                    wav_batch, use_tensor=True
                ).detach().float().cpu()
            for k, j in enumerate(idxs):
                audio_emb[j] = emb[k]
        pbar.update(len(batch_rows))
    pbar.close()
    del clap
    torch.cuda.empty_cache()

    # ── CLIP-text caption embeddings (must match VGGSound text_features) ────
    feats = FeaturesUtils(
        tod_vae_ckpt=args.vae_ckpt,
        bigvgan_vocoder_ckpt=args.vocoder_ckpt,
        synchformer_ckpt=args.synchformer_ckpt,
        enable_conditions=True,
        mode=args.mode,
        need_vae_encoder=False,
    ).to(device).eval()

    captions = [str(r['caption']) for r in on_disk]
    text_emb_chunks = []
    pbar = tqdm(total=len(captions), desc='clip-text', dynamic_ncols=True)
    for i in range(0, len(captions), args.text_batch):
        chunk = captions[i:i + args.text_batch]
        with torch.inference_mode():
            t = feats.encode_text(chunk).detach().float().cpu()
        text_emb_chunks.append(t)
        pbar.update(len(chunk))
    pbar.close()
    text_emb = torch.cat(text_emb_chunks, dim=0)
    if text_emb.shape[0] != len(on_disk):
        raise RuntimeError(f'text_emb {text_emb.shape} vs {len(on_disk)}')
    log.info(f'audio_emb={tuple(audio_emb.shape)} '
             f'text_emb={tuple(text_emb.shape)}')

    # ── Persist as TensorDict memmap (matches ExtractedVGG layout) ───────────
    td = TensorDict(
        {'clap_features': audio_emb, 'text_features': text_emb},
        batch_size=[len(on_disk)],
    )
    mmap_path = args.output_dir / 'meta.memmap'
    td.memmap_(str(mmap_path))

    sidecar = args.output_dir / 'ids.tsv'
    pd.DataFrame(
        [{'idx': i, 'id': r['id'], 'caption': r['caption']}
         for i, r in enumerate(on_disk)]
    ).to_csv(sidecar, sep='\t', index=False)

    log.info(f'wrote {len(on_disk)} rows → {mmap_path}')
    log.info(f'sidecar              → {sidecar}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

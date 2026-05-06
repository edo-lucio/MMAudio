"""ExtractedAudioCaps: companion dataset for the cross-dataset FGW variant.

This dataset feeds the *audio side* of the Reading-B GW objective: each
sample carries CLAP audio embedding + CLIP-text caption embedding (no
video, no VAE latents). Used only as an auxiliary loader; the main VGGSound
loader still drives flow-matching.

Memmap layout produced by `src/training/extract_audiocaps_features.py`:
  meta.memmap/
    clap_features:  (N, 512)
    text_features:  (N, T_seq, T_dim)
"""
import logging
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data.dataset import Dataset

from mmaudio.utils.dist_utils import local_rank

log = logging.getLogger()


class ExtractedAudioCaps(Dataset):

    def __init__(
        self,
        *,
        memmap_dir: Union[str, Path],
        ids_tsv: Union[str, Path] | None = None,
    ):
        super().__init__()
        memmap_dir = Path(memmap_dir)
        td = TensorDict.load_memmap(memmap_dir / 'meta.memmap')
        self.clap_features = td['clap_features']
        self.text_features = td['text_features']

        if ids_tsv is not None and Path(ids_tsv).is_file():
            self.df_list = pd.read_csv(ids_tsv, sep='\t').to_dict('records')
        else:
            # Fall back to numeric ids if the sidecar is missing.
            self.df_list = [
                {'id': str(i), 'caption': ''}
                for i in range(len(self.clap_features))
            ]
        if local_rank == 0:
            log.info(f'AudioCaps: loaded {len(self)} samples '
                     f'from {memmap_dir}')
            log.info(f'  clap_features={tuple(self.clap_features.shape)}')
            log.info(f'  text_features={tuple(self.text_features.shape)}')

        self.audio_exist = torch.tensor(1, dtype=torch.bool)

    def __len__(self) -> int:
        return self.clap_features.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            'id': self.df_list[idx]['id'],
            'audio_clap': self.clap_features[idx],          # (512,)
            'text_features': self.text_features[idx],        # (T_seq, T_dim)
            'audio_exist': self.audio_exist,
        }

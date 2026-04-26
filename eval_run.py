"""Post-training evaluation: synthesize EMA + generate test audio + compute metrics.

Equivalent to the tail of `train.py` (the EMA-synth + `sample()` block at the end),
but standalone. Run on any `output/<exp_id>/` that has accumulated EMA buffers,
even if training was interrupted before reaching the final iteration.

For exp_id=<run_id>, this:
  1. Synthesizes output/<run_id>/<run_id>_ema_final.pth from the EMA buffers
     under output/<run_id>/ema_ckpts/ (if it doesn't already exist).
  2. Calls `sample()` from mmaudio.sample — generates audio for the VGGSound
     test set, runs av_bench.extract + evaluate on it, and writes metrics to
     output/<run_id>/test-output_metrics.json.

Usage:
    torchrun --standalone --nproc_per_node=1 eval_run.py exp_id=<run_id>

Example:
    torchrun --standalone --nproc_per_node=1 eval_run.py \\
        exp_id=gw_baseline model=small_16k duration_s=8.0 batch_size=32

Hydra overrides (composed from eval_config.yaml -> base_config.yaml -> data/base.yaml):
  exp_id=<id>          # required — points at output/<id>/
  model=small_16k      # must match what was trained
  duration_s=8.0       # match the training audio length
  batch_size=32        # per-GPU eval batch size
  weights=path.pth     # explicit override; otherwise auto-finds <exp_id>_ema_final.pth
"""
import logging
from datetime import timedelta
from pathlib import Path

import hydra
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from mmaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from mmaudio.sample import sample
from mmaudio.utils.dist_utils import local_rank
from mmaudio.utils.synthesize_ema import synthesize_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_config.yaml')
def main(cfg: DictConfig):
    distributed.init_process_group(backend='nccl', timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)

    run_dir = HydraConfig.get().run.dir

    # Patch data_dim from sequence_config to match the model variant.
    if cfg.model.endswith('16k'):
        seq_cfg = CONFIG_16K
    elif cfg.model.endswith('44k'):
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Unknown model: {cfg.model}')
    with open_dict(cfg):
        if 'data_dim' in cfg:
            cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
            cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
            cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # Synthesize EMA if not already present.
    ema_path = Path(run_dir) / f'{cfg.exp_id}_ema_final.pth'
    if local_rank == 0 and not ema_path.is_file():
        log.info(f'No {ema_path.name} on disk — synthesizing EMA '
                 f'with sigma={cfg.ema.default_output_sigma}')
        state_dict = synthesize_ema(cfg, cfg.ema.default_output_sigma, step=None)
        torch.save(state_dict, ema_path)
        log.info(f'Synthesized EMA saved to {ema_path}')
    distributed.barrier()

    log.info(f'Running test-set inference + metric extraction for exp_id={cfg.exp_id}')
    sample(cfg)

    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    main()

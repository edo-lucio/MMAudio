# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MMAudio (CVPR 2025) — flow-matching video/text → audio synthesis. This fork adds **Gromov-Wasserstein (GW) regularization** to the training loss to align the geometry of the video conditioning space with the audio representation space. See `mmaudio/model/gw_regularization.py`, the `gw_regularization` block in `config/base_config.yaml`, and `experiments/`.

## Install

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -e .
# for evaluation during training:
# pip install git+https://github.com/hkchengrex/av-benchmark
```

Place VAE / vocoder / Synchformer / `empty_string.pth` checkpoints in `ext_weights/` per `docs/TRAINING.md`.

## Common commands

Inference (CLI / gradio):
```bash
python demo.py --duration=8 --video=<path> --prompt "..."
python gradio_demo.py    # port 7860
```

Feature extraction (precompute before training):
```bash
torchrun --standalone --nproc_per_node=<n> training/extract_video_training_latents.py
torchrun --standalone --nproc_per_node=<n> training/extract_audio_training_latents.py
# audio also needs a prior partitioning step:
python training/partition_clips.py
```

Train:
```bash
# sanity (no compile, example data, single GPU)
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 train.py \
    exp_id=debug compile=False debug=True example_train=True batch_size=1
# full run
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py \
    exp_id=exp_1 model=small_16k
# GW-regularized run (override via Hydra)
torchrun --standalone --nproc_per_node=8 train.py exp_id=gw_global \
    gw_regularization.enabled=true gw_regularization.variant=global
```
Outputs land in `output/<exp_id>/`.

Batch evaluation:
```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 batch_eval.py \
    duration_s=8 dataset=vggsound model=small_16k num_workers=8
```

GW experiments:
```bash
bash experiments/run_gw_experiments.sh baseline    # or: variants|lambda|detach|schedule|ood|all
python experiments/make_ood_split.py --train_tsv sets/vgg3-train.tsv \
    --test_tsv sets/vgg3-test.tsv --n_holdout 30 --out_dir sets/
python experiments/analyze_couplings.py --weights output/<run>/<run>_ema_final.pth
python experiments/analysis.py curves --runs output/gw_*
```

No linter, formatter, or test suite is configured in this repo.

## Architecture

### Training fast-path: precomputed TensorDicts, not raw video

Training **does not** run CLIP / Synchformer / VAE encoders at step time. Everything is precomputed to memory-mapped `TensorDict`s (see `docs/TRAINING.md`). `mmaudio/data/extracted_vgg.py` and `extracted_audio.py` load these and yield per-sample dicts with keys `a_mean`, `a_std`, `clip_features`, `sync_features`, `text_features`, `video_exist`, `text_exist`, `id`, `caption`. `MultiModalDataset` (`mmaudio/data/mm_dataset.py`) concats video-bearing and audio-only datasets — hence the `video_exist`/`text_exist` flags. Audio-only samples have empty video features, and vice versa.

### Three sources of modality masking

Be careful when touching conditioning tensors — they get mutated in three places:

1. **Presence masking** (`runner.py` `train_pass`): `clip_f[~video_exist] = empty_clip_feat` etc. — samples without a modality get a learnable null token.
2. **CFG masking** (`runner.py` `train_fn`, in-place): a fraction (`null_condition_probability`) of samples have video or text replaced with the null token for classifier-free guidance training.
3. **Network internals**: `MMAudio.preprocess_conditions` in `mmaudio/model/networks.py` projects and averages features and must be called with real conditioning; see step (2) — `clip_f` is mutated in-place inside `train_fn`, so if you need pre-CFG-null features (e.g. for GW), **clone before calling `train_fn`**.

### Flow-matching network (`mmaudio/model/networks.py`)

`MMAudio.forward(latent, clip_f, sync_f, text_f, t)` splits into two halves:

1. `preprocess_conditions` — projects raw features to `hidden_dim` and builds `clip_f_c`, `text_f_c` (both `(B, hidden_dim)`, avg-pooled then conditional-projection). These feed the **global** conditioning `c_g = global_cond_mlp(clip_f_c + text_f_c) + t_embed(t)`. Cached across ODE steps at inference.
2. `predict_flow` — runs the audio latent through `audio_input_proj` → N1 `JointBlock`s (video/text/audio joint-attention, adaLN from `c_g`) → N2 `MMDitSingleBlock`s (audio-only, adaLN from `c_g + sync_f`) → `FinalBlock`. Sync features are injected as a per-token adaLN signal rather than through attention.

Sequence lengths are fixed per model: `CONFIG_16K`/`CONFIG_44K` in `mmaudio/model/sequence_config.py` (latent 250/345, CLIP 64, sync 192). Changing duration requires calling `MMAudio.update_seq_lengths` so RoPE rotations get recomputed.

Network factories: `small_16k` / `small_44k` / `medium_44k` / `large_44k` / `large_44k_v2` in `networks.py`. `v2` variants differ in activation (SiLU vs SELU) and timestep-embedder range; **the training script does not support `_v2` training.**

### Runner (`mmaudio/runner.py`)

`Runner` owns DDP wrapping, EMA (via `nitrous_ema.PostHocEMA`, `sigma_rels=[0.05, 0.1]`), the optimizer, the `FeaturesUtils` (VAE + BigVGAN + Synchformer — used only for decoding/logging at train time and for full eval), TensorBoard logging, and checkpointing.

Key methods: `train_fn` (torch.compiled hot path), `train_pass` (non-compiled wrapper: dataloading, masking, logging, backward, EMA update), `validation_pass`, `inference_pass` (runs the ODE via `FlowMatching.to_data` with CFG wrapping), `eval` (uses `av_bench` to compute FD_PaSST / IS / IB-score / DeSync).

### Flow matching (`mmaudio/model/flow_matching.py`)

Minimal class: `get_x0_xt_c` samples `x0 ~ N(0, I)` and interpolates `xt`; `loss` is MSE against `x1 - (1 - min_sigma) * x0`; `to_data` runs Euler or adaptive ODE via `torchdiffeq`.

### Config (Hydra)

Entry configs: `config/train_config.yaml` (extends `base_config.yaml`) and `config/eval_config.yaml`. Data paths live in `config/data/base.yaml` and `config/eval_data/base.yaml` — update before training. Any field is overridable on the CLI: `key=value` or dotted paths like `gw_regularization.lambda_gw=0.05`.

### GW regularization (this fork's addition)

`mmaudio/model/gw_regularization.py` implements entropic GW via mirror-descent with an inner Sinkhorn projection (Peyré et al.), plus a fused-GW variant. All numerics run inside `torch.cuda.amp.autocast(enabled=False)` on fp32 — the outer training loop uses bf16 autocast and Sinkhorn is not stable in bf16.

Four variants (selected via `gw_regularization.variant`):
- `global` — pairwise dists on raw CLIP avg-pool vs normalized `a_mean` avg-pool
- `projected` — same but after `clip_input_proj` / `audio_input_proj`
- `c_g` — uses `clip_cond_proj(clip_input_proj(clip_f).mean(1))` (the video contribution to `c_g`)
- `fused` — FGW with cosine cross-domain cost on projected features

The GW loss is computed in `train_pass` **after** `train_fn` returns, over only the `video_exist=True` subset. `clip_f` is cloned **before** `train_fn` so CFG-nulling doesn't leak in. `a_mean` (deterministic VAE posterior mean), not a noisy sample, is used as the audio representation — it's more stable and doesn't require bouncing intermediate tensors out of the compiled path. `detach_video=true` by default: the audio space is shaped to the video geometry, not the other way around.

`lambda_schedule` provides `constant` / `linear_rampup` / `cosine_anneal`; configure warmup via `gw_regularization.warmup_steps`.

## Gotchas

- `MMAudio.normalize` / `unnormalize` and `a_mean.sub_().div_()` are **in-place** — clone if you need the original.
- `torch.compile` is on by default (`compile=True`). Disable for debugging: `compile=False`.
- GradScaler is disabled by default (`enable_grad_scaler: False`) — see README update log (stability regression fix, 2025-02-27).
- Checkpoints follow two shapes: `*_last.pth` (weights only) and `*_ckpt_last.pth` (weights + optimizer + scheduler + EMA). EMA weights are synthesized post-training into `*_ema_final.pth`.
- Validation uses `eval_rng_clone` for reproducibility; don't draw from `trainer.rng` inside val/eval without snapshotting.
- Hydra run directory is `./output/${exp_id}` — re-running the same `exp_id` auto-resumes from `<exp_id>_ckpt_last.pth`.

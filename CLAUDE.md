# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo purpose

Research fork of **MMAudio** (CVPR 2025, Cheng et al.) that adds a **Gromov-Wasserstein (GW) regularizer** to the flow-matching V2A training objective. The fork doubles as the codebase and the thesis write-up:

- `train.py`, `mmaudio/`, `config/`, `experiments/`, `jobs/` — training, eval, and cluster scripts.
- `thesis/` — LaTeX thesis (chapters, figures, `refs.bib`).
- `SETUP.md` is the authoritative end-to-end setup/experiment guide; prefer it over re-deriving install or data-prep steps.

## Code architecture (big picture)

The training pipeline is a Hydra-configured DDP flow-matching trainer that **never extracts features at step time** — CLIP, Synchformer, and VAE latents are all precomputed into memmapped TensorDicts. This is the most important fact about the repo: the hot training loop only does flow-matching (+ optional GW) on cached tensors.

Call graph:

```
train.py  (Hydra entry; distributed_setup; seeds; dataset + loader plumbing)
  └─ mmaudio/runner.py :: Runner
        ├─ mmaudio/model/networks.py :: get_my_mmaudio     # DiT; DDP-wrapped
        ├─ mmaudio/model/flow_matching.py :: FlowMatching  # LFM loss + Euler sampler
        ├─ mmaudio/model/gw_regularization.py              # LGW / LFGW (see below)
        └─ nitrous_ema.PostHocEMA                          # post-hoc EMA buffers
```

`Runner.train_pass` is where GW is injected: `train_fn` (compiled) returns the FM loss + features; GW is computed **after** in fp32 over the `video_exist=True` subset, then added with `lambda_schedule`. `train_fn` and `val_fn` are compiled **separately** on purpose — merging them destroys performance. `torch.compile` is on by default (`compile=False` to disable when debugging).

GW variants in `mmaudio/model/gw_regularization.py::compute_gw_regularization`:
- `global` — pairwise dists on raw CLIP avg-pool vs normalized `a_mean` avg-pool
- `projected` — same but after `clip_input_proj` / `audio_input_proj`
- `c_g` — uses the video contribution to the global conditioning `c_g`
- `fused` — FGW with a cosine cross-domain cost

Data flow: `mmaudio/data/extracted_vgg.py` + `extracted_audio.py` are memmap-backed datasets; `mm_dataset.py` mixes them. Extraction scripts live in `training/` (video: `extract_video_training_latents.py`, audio: `extract_audio_training_latents.py`).

External encoders/decoders live under `mmaudio/ext/` (autoencoder/VAE, BigVGAN vocoder, Synchformer). Their weights are loaded from `ext_weights/` — see `config/base_config.yaml` for the paths.

## Configuration (Hydra)

Config composes `config/base_config.yaml` ← `config/train_config.yaml` (with `data: base` from `config/data/base.yaml`). Override anything on the CLI:

```bash
torchrun --standalone --nproc_per_node=N train.py \
    exp_id=my_run \
    model=small_16k \
    gw_regularization.enabled=true \
    gw_regularization.variant=global \
    gw_regularization.lambda_gw=0.01 \
    batch_size=64 compile=False
```

- `exp_id` is the single knob that controls the run directory (`./output/${exp_id}`) and **auto-resume**: re-running with the same `exp_id` picks up `<exp_id>_ckpt_last.pth`. Change `exp_id` to start fresh.
- `model` must end with `16k` or `44k` (routes to `CONFIG_16K` / `CONFIG_44K` in `mmaudio/model/sequence_config.py`).
- `_v2` networks (e.g. `mmaudio_large_44k_v2.pth`) are **inference-only** — they cannot be trained with this script.

## Common commands

All training/eval commands assume you've run `pip install -e .` and have `ext_weights/` + precomputed memmaps in place. See `SETUP.md` for the slow one-time data prep; the commands below are the ones you run repeatedly.

```bash
# Smoke test (1 GPU, no learning, just confirms the pipeline runs):
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 train.py \
    exp_id=debug compile=False debug=True example_train=True batch_size=1

# Baseline training (GW off), 2 GPUs:
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py \
    exp_id=baseline_small_16k model=small_16k

# Full GW experiment sweep driver (baseline, variants, lambda, detach, schedule, ood, gwxsync, all):
GPUS=8 ITERS=300000 MODEL=small_16k bash experiments/run_gw_experiments.sh <group>

# Feature extraction (edit the constants at the top of the script first):
torchrun --standalone --nproc_per_node=N training/extract_video_training_latents.py
python  training/partition_clips.py
torchrun --standalone --nproc_per_node=N training/extract_audio_training_latents.py

# Batch eval (FD / IS / IB / DeSync) on a trained checkpoint:
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 batch_eval.py \
    duration_s=8 dataset=vggsound model=small_16k num_workers=8 \
    weights=output/<exp_id>/<exp_id>_ema_final.pth

# Demo (text + optional video):
python demo.py --duration=8 --video=<path> --prompt "..."
```

### Checkpoint flavors

Produced in `output/<exp_id>/`:
- `<exp_id>_last.pth` — weights only.
- `<exp_id>_ckpt_last.pth` — weights + optimizer + scheduler + EMA (used for resuming).
- `<exp_id>_ema_final.pth` — synthesized **post-training** from EMA buffers via `mmaudio/utils/synthesize_ema.py`. Use this for eval/demo.

### GW analysis scripts (`experiments/`)

- `analyze_couplings.py` — inspect learned T\*; requires the run was launched with `gw_regularization.save_couplings=true`.
- `analysis.py curves --runs output/gw_*` — compare training curves across the sweep.
- `diagnose_geometry.py` — pairwise-distance diagnostics.
- `make_ood_split.py` — produces `sets/vgg3-{train,test}-ood.tsv` for the OOD group.

## Cluster / SLURM workflow

**Always** use the templates in `jobs/` for SLURM submissions — they encode non-obvious cluster constraints:

- `jobs/_header.sh` and `jobs/_common.sh` load CUDA/NCCL/Miniconda modules, activate the env, `cd` to the repo, cap thread pools, and alias `sets/vgg-*.tsv` → `sets/vgg3-*.tsv`. Source `_header.sh` from every job after the `#SBATCH` block.
- **RLIMIT_NPROC on this cluster is ~12.** `_common.sh` therefore pins `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=2`, `MKL_NUM_THREADS=2`, `NUMEXPR_NUM_THREADS=2`, `VECLIB_MAXIMUM_THREADS=2`. Keep `#SBATCH --cpus-per-task` ≤ ~12 and `num_workers × GPUs` within that budget, or BLAS will either crash or silently hang during `import torch` (symptom: empty stdout, because Python never finishes importing).
- Config mentions `sets/vgg3-*.tsv` but the repo ships `sets/vgg-*.tsv` — `_common.sh` copies them on first run. Don't delete the copies.

Existing job scripts (`train_baseline.job`, `train_variants.job`, `train_lambda.job`, `train_detach.job`, `train_schedule.job`, `train_ood.job`, `train_gwxsync.job`, `eval.job`, `eval_all.job`, `extract_features.job`, `analysis.job`, `diagnose_geometry.job`) are the canonical entry points — prefer parameterizing them (`EXP_ID=… MODEL=… ITERS=… GPUS=… sbatch ...`) over writing new ones.

## Gotchas

These bite on long runs. In priority order:

- **Sinkhorn is fp32-only.** `compute_gw_regularization` casts explicitly; bf16 produces NaNs. Don't "optimize" this.
- **`GradScaler` is disabled** (`enable_grad_scaler: False`). This is deliberate (Feb 2025 stability fix, see README changelog), not an oversight.
- **In-place mutations in the model:** `MMAudio.normalize` / `unnormalize` and `a_mean.sub_().div_()` modify their inputs. `clip_f` is also mutated in-place inside `train_fn` for CFG null-masking. `runner.py::train_pass` clones before calling the GW regularizer — preserve that if you edit it.
- **Validation RNG:** `eval_rng_clone` is used so val/eval is reproducible across runs. Don't draw from `trainer.rng` inside val/eval without snapshotting.
- **Same `exp_id` = resume**, not overwrite. Change `exp_id` (or delete `output/<exp_id>/<exp_id>_ckpt_last.pth`) to start over.
- **`num_workers` is per-GPU**, multiplied by `--nproc_per_node`. Easy to exceed the nproc budget.

## Thesis conventions

- Class: `article`, 11pt, `natbib` with `(round, authoryear)`.
- Math macros (defined in `thesis/thesis.tex` preamble): `\LFM, \LGW, \LFGW, \DV, \DA, \Tstar, \R, \E`. Always use them; never redefine or inline.
- Figures: `\includegraphics{figures/name}`; tables use `booktabs`.
- Bibliography: `thesis/refs.bib` — ~40 entries covering OT foundations, flow matching, V2A, audio-visual alignment, multimodal backbones. Add new keys there; don't duplicate them into this file.

### Chapter map (`thesis/chapters/`)

- `intro.tex` — motivation, pointwise vs relational gap, contributions.
- `background.tex` — flow matching, GW/FGW, existing V2A models (MMAudio, Synchformer).
- `method.tex` — formalization, FGW, training objective.
- `design.tex` — D1–D5 design axes and justifications.
- `experiments.tex` — H1–H5 hypotheses, ablation grid, metrics.
- `results.tex` — filled in after experiments run.
- `conclusion.tex` — scope/limitations, expected contributions.

### Thesis argument (one paragraph)

Pointwise objectives (contrastive, attention, adaLN) align instances but not relational geometry. We add a GW penalty between intra-video and intra-audio pairwise distance matrices, solved via entropic relaxation + Sinkhorn. Tested across 5 design axes (D1–D5) against FAD / PaSST / ImageBind / DeSync metrics.

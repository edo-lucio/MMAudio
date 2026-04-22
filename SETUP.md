# MMAudio + GW Regularization — Setup & Experiment Guide

End-to-end checklist for going from a fresh clone to running the full GW-regularization sweep. Work through the sections top-to-bottom; each one assumes the previous is done.

Reference docs in the repo: `docs/TRAINING.md`, `docs/MODELS.md`, `docs/EVAL.md`, `CLAUDE.md`.

---

## 0. Hardware & storage sanity-check

Taken from `docs/TRAINING.md` — these are the upstream author's recommendations, not hard minimums.

- Single node; multi-node training is not implemented.
- GPUs: 2× 80GB H100 for `small_16k`; 8× 80GB H100 for `large_44k`. GW experiments default to 8 GPUs (see `experiments/run_gw_experiments.sh`).
- System RAM: 600GB+ for 16kHz, 700GB+ for 44.1kHz (the memmapped TensorDicts get hot-cached by the OS, which is what unlocks 3–5 GB/s random reads).
- Storage: >2TB fast NVMe. If RAM is large enough, caching compensates for slower disks.

If you have less, shrink `batch_size` and `num_workers` in `config/base_config.yaml` and expect slower steps — the code will still run.

---

## 1. Environment

- [ ] **Python 3.9+** in a fresh env (miniforge recommended).
- [ ] **Install PyTorch** matching your CUDA. Example for CUDA 11.8:
  ```bash
  pip install torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu118 --upgrade
  ```
  PyTorch **2.5.1+** is required.
- [ ] **ffmpeg < 7** (torchaudio imposes this upper bound):
  ```bash
  conda install -c conda-forge 'ffmpeg<7'
  ```
- [ ] **Install MMAudio in editable mode** from the repo root:
  ```bash
  pip install -e .
  ```
- [ ] **Install `av-benchmark`** (needed by `Runner.eval` for validation-during-training and full test-set eval):
  ```bash
  pip install git+https://github.com/hkchengrex/av-benchmark
  ```
- [ ] Verify GPUs are visible: `python -c "import torch; print(torch.cuda.device_count())"`.

---

## 2. Pretrained weights (external modules)

Place these in `ext_weights/` at the repo root. Full list from `docs/MODELS.md`:

| File | Purpose | Required for |
| --- | --- | --- |
| `v1-16.pth` | 16kHz VAE | 16kHz training (incl. `small_16k`, the GW default) |
| `v1-44.pth` | 44.1kHz VAE | 44.1kHz training |
| `best_netG.pt` | 16kHz BigVGAN vocoder | 16kHz decoding/eval |
| `synchformer_state_dict.pth` | Synchformer visual encoder | Video feature extraction + training |
| `empty_string.pth` | Pre-encoded empty text | Training (null-condition token) |

Download commands (minimal set for 16kHz GW experiments — matches the GW sweep default `MODEL=small_16k`):

```bash
mkdir -p ext_weights && cd ext_weights
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth
cd ..
```

Add these if you plan to train 44.1kHz too (the 44.1kHz vocoder downloads automatically):

```bash
wget -P ext_weights https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-44.pth
```

Flow-prediction network weights (only if you want to run `demo.py` or start training from a pretrained checkpoint — **not required** to start training from scratch):

```bash
mkdir -p weights
wget -P weights https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_small_16k.pth
# optional:
wget -P weights https://huggingface.co/hkchengrex/MMAudio/resolve/main/weights/mmaudio_large_44k_v2.pth
```

Expected layout after this step:

```
MMAudio/
├── ext_weights/
│   ├── v1-16.pth
│   ├── best_netG.pt
│   ├── synchformer_state_dict.pth
│   └── empty_string.pth
└── weights/                        # optional
    └── mmaudio_small_16k.pth
```

Paths are wired in `config/base_config.yaml` (`vae_16k_ckpt`, `bigvgan_vocoder_ckpt`, `synchformer_ckpt`, …) — no edits needed if you match the layout above.

---

## 3. Smoke test (before touching real data)

Verify the install with the bundled example data (a handful of clips in `training/example_videos/` and `training/example_audios/`):

- [ ] Extract example video features:
  ```bash
  torchrun --standalone training/extract_video_training_latents.py
  ```
- [ ] Extract example audio features (needs the partitioning step first):
  ```bash
  python training/partition_clips.py
  torchrun --standalone training/extract_audio_training_latents.py
  ```
- [ ] Run a 1-GPU debug training pass — this doesn't learn anything, it just confirms the pipeline is wired up:
  ```bash
  OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 train.py \
      exp_id=debug compile=False debug=True example_train=True batch_size=1
  ```
- [ ] Optional end-to-end sanity: run the demo on a sample video (downloads `mmaudio_small_16k.pth` automatically if missing):
  ```bash
  python demo.py --duration=8 --video=training/example_videos/<file>.mp4 --prompt "..."
  ```

If any of these fail, fix the environment/weights before moving on — real training will hit the same errors an hour in.

---

## 4. Training datasets

MMAudio trains on a joint mixture (see `docs/TRAINING.md`). **For this GW fork's experiments, VGGSound is the only strictly required dataset** — every experiment in `experiments/run_gw_experiments.sh` runs on `model=small_16k` with the ExtractedVGG dataset. The others are needed only if you want to reproduce upstream's multi-dataset training.

### 4a. VGGSound (required)

- Source: <https://www.robots.ox.ac.uk/~vgg/data/vggsound/> — gated, requires agreeing to the license. YouTube videos; most distributions ship CSVs + downloader scripts since the authors cannot redistribute.
- After download, expected layout (per `config/data/base.yaml`):
  ```
  ../data/video/<video_id>.mp4
  sets/vgg-train.tsv      (already in the repo)
  sets/vgg-test.tsv       (already in the repo)
  sets/vgg-val.tsv        (already in the repo)
  ```
  Note `config/data/base.yaml` currently references `sets/vgg3-{train,test,val}.tsv` but only `sets/vgg-{train,test,val}.tsv` are checked in — you will need to **either rename the tsvs or update `config/data/base.yaml`** before training. Pick one and stick with it.

### 4b. Optional audio-only datasets (for full mix)

Only needed if you want to replicate upstream's audio-text joint training:

- **AudioCaps** — <https://audiocaps.github.io/>
- **WavCaps** — <https://huggingface.co/datasets/cvssp/WavCaps>. HF release is 32kHz-downsampled — fine for 16kHz, but find the original-SR audio if you plan 44.1kHz training. The "SoundBible" subset is skipped upstream.
- **Clotho** — <https://zenodo.org/record/4783391>
- **AudioSet (strongly-labeled)** — <https://research.google.com/audioset/>
- **Freesound** — <https://freesound.org/> (via <https://huggingface.co/datasets/Meranti/CLAP_freesound> or similar)
- **BBC Sound Effects** — <https://sound-effects.bbcrewind.co.uk/>

Reference tsvs (with overlap/dedup fixes applied upstream) — use the **Mar 9, 2025 corrected** set:
<https://github.com/hkchengrex/MMAudio/releases/tag/v0.1>

### 4c. Evaluation assets

- [ ] **Precomputed eval cache** (saves you re-running `av-benchmark` extraction):
  <https://huggingface.co/datasets/hkchengrex/MMAudio-precomputed-results/tree/main>
  Place at `../data/eval-cache/vggsound-{test,val}/` to match `config/data/base.yaml`.
- [ ] Optional test-set CSVs / video directories are listed in `config/eval_data/base.yaml` (AudioCaps test set, VGGSound test videos, MovieGen). Most relevant for the GW sweep is `../data/test-videos/` + `../data/vggsound.csv` for VGGSound eval.

---

## 5. Precompute training features (the slow, one-time step)

Training **does not** run CLIP / Synchformer / VAE at step time — everything is precomputed into memmapped TensorDicts. Plan for 1–several days of extraction depending on GPU count.

### 5a. Video features (VGGSound)

Edit the constants at the top of `training/extract_video_training_latents.py` to match your paths (input video dir, `latent_dir`, `output_dir`, split). Target 16kHz for the GW sweep. Then:

```bash
torchrun --standalone --nproc_per_node=<N> training/extract_video_training_latents.py
```

Produces, in `output_dir`:
- `vgg-{split}/` — TensorDict with `mean.memmap`, `std.memmap`, `clip_features.memmap`, `sync_features.memmap`, `text_features.memmap`, `meta.json`
- `vgg-{split}.tsv` — `id`, `label` metadata

Run this for `train`, `val`, and `test` splits. The config expects outputs at `../data/v1-16-memmap/vgg-{train,test,val}/`.

### 5b. Audio features (AudioCaps / WavCaps / Clotho — only if doing full mix)

Two-step process:

```bash
python training/partition_clips.py \
    --data_dir <audio dir> --output_dir <clips.csv>
torchrun --standalone --nproc_per_node=<N> training/extract_audio_training_latents.py
```

Again, edit the constants at the top of each script. Repeat per dataset. Outputs land under `../data/v1-16-memmap/<dataset>/` and `<dataset>.tsv`.

### 5c. Point the config at your extracted features

Everything is defined in `config/data/base.yaml`. Defaults assume `../data/v1-16-memmap/...`. Update if your layout differs. Double-check:

- `ExtractedVGG.tsv` / `.memmap_dir`
- `ExtractedVGG_val.*` + `ExtractedVGG_val.gt_cache`
- `ExtractedVGG_test.*` + `ExtractedVGG_test.gt_cache`

---

## 6. Baseline training (no GW)

Before running GW experiments, confirm a vanilla run starts and logs cleanly:

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=2 train.py \
    exp_id=baseline_small_16k model=small_16k
```

Outputs land in `output/baseline_small_16k/`. Hydra auto-resumes if the same `exp_id` re-runs (looks for `<exp_id>_ckpt_last.pth`).

Useful overrides:
- `compile=False` — disable `torch.compile` if debugging.
- `batch_size=<n>` — shrink for smaller GPUs.
- `num_iterations=<n>` — matches the GW sweep's default of 300000.

---

## 7. GW regularization experiments

All experiments are driven by `experiments/run_gw_experiments.sh`. Defaults: 8 GPUs, 300k iters, `small_16k`. Override via env vars:

```bash
GPUS=4 ITERS=200000 MODEL=small_16k bash experiments/run_gw_experiments.sh <name>
```

### 7a. The 4 GW variants (from `mmaudio/model/gw_regularization.py`)

- `global` — pairwise dists on raw CLIP avg-pool vs normalized `a_mean` avg-pool
- `projected` — same but after `clip_input_proj` / `audio_input_proj`
- `c_g` — uses the video contribution to the global conditioning `c_g`
- `fused` — FGW with a cosine cross-domain cost

All four are computed in `runner.py::train_pass` **after** `train_fn` returns, over the `video_exist=True` subset only, in fp32 (Sinkhorn is unstable in bf16).

### 7b. The sweep

Run each group with `bash experiments/run_gw_experiments.sh <group>`:

- [ ] **`baseline`** — `gw_baseline` run with GW off. Reference point.
- [ ] **`variants`** — `gw_var_{global,projected,c_g,fused}` at λ=0.01. Identifies the best variant. Set `BEST_VARIANT` in env before running the remaining groups.
- [ ] **`lambda`** — λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1} on the best variant.
- [ ] **`detach`** — `detach_video=true|false` ablation. Default `true` shapes audio to video geometry, not vice versa.
- [ ] **`schedule`** — `constant | linear_rampup | cosine_anneal` with `warmup_steps=50000`.
- [ ] **`ood`** — held-out-class generalization (see 7c).
- [ ] **`gwxsync`** — 2×2 GW × sync-features-on/off.
- [ ] **`all`** — runs baseline + variants + lambda + detach + schedule + ood.

Example single launch:

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    exp_id=gw_global \
    gw_regularization.enabled=true \
    gw_regularization.variant=global
```

### 7c. OOD split (required before `exp_ood`)

Hold out 30 classes from train, keep them in test:

```bash
python experiments/make_ood_split.py \
    --train_tsv sets/vgg3-train.tsv \
    --test_tsv sets/vgg3-test.tsv \
    --n_holdout 30 \
    --out_dir sets/
```

This writes `sets/vgg3-train-ood.tsv`, `sets/vgg3-test-ood.tsv`, `sets/vgg3-ood-classes.txt`. You also need to filter the memmapped tsv to match — build `../data/v1-16-memmap/vgg-train-ood.tsv` (a row-subset of `vgg-train.tsv` keeping only ids whose label is not in the held-out set). The `ood` experiment group passes both tsvs as Hydra overrides.

---

## 8. Post-hoc analysis & evaluation

### 8a. Batch eval (FD / IS / IB / DeSync)

```bash
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=4 batch_eval.py \
    duration_s=8 dataset=vggsound model=small_16k num_workers=8 \
    weights=output/<exp_id>/<exp_id>_ema_final.pth
```

### 8b. GW-specific analysis scripts

- **Coupling structure** — inspect the learned transport plan T*:
  ```bash
  python experiments/analyze_couplings.py --weights output/<run>/<run>_ema_final.pth
  ```
  Requires the run to have been launched with `gw_regularization.save_couplings=true`.
- **Training-curve comparison** across the whole `gw_*` sweep:
  ```bash
  python experiments/analysis.py curves --runs output/gw_*
  ```

### 8c. Checkpoint shapes (from `CLAUDE.md`)

- `<exp_id>_last.pth` — weights only.
- `<exp_id>_ckpt_last.pth` — weights + optimizer + scheduler + EMA (used for resuming).
- `<exp_id>_ema_final.pth` — synthesized **post-training** from the EMA buffers. Use this for eval/demo.

---

## 9. Gotchas (worth re-reading before long runs)

From `CLAUDE.md` and the runner:

- `MMAudio.normalize` / `unnormalize` and `a_mean.sub_().div_()` are **in-place** — clone before GW if you need originals.
- `clip_f` is mutated in-place inside `train_fn` for CFG null-masking — GW already clones before calling, but if you edit `runner.py::train_pass`, preserve that clone.
- `torch.compile` is on by default; disable with `compile=False` when debugging.
- `GradScaler` is disabled (`enable_grad_scaler: False`) — this is deliberate, not a bug. Don't re-enable without reading the Feb 2025 stability note in the README.
- Validation uses `eval_rng_clone` for reproducibility — don't draw from `trainer.rng` inside val/eval without snapshotting.
- Hydra `run.dir` is `./output/${exp_id}` — re-using an `exp_id` auto-resumes.
- `_v2` networks (`mmaudio_large_44k_v2.pth`) **cannot be trained** with this script; inference only.

---

## 10. Quick-start TL;DR (minimal path to a GW run)

```bash
# 1. env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
pip install -e .
pip install git+https://github.com/hkchengrex/av-benchmark

# 2. external weights (16kHz only)
mkdir -p ext_weights && cd ext_weights
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/v1-16.pth
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/best_netG.pt
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/synchformer_state_dict.pth
wget https://github.com/hkchengrex/MMAudio/releases/download/v0.1/empty_string.pth
cd ..

# 3. data: VGGSound videos → ../data/video/ ; splits in sets/
#    fix mismatch: the config references sets/vgg3-*.tsv, repo has sets/vgg-*.tsv
#    → either rename or update config/data/base.yaml

# 4. precompute
torchrun --standalone --nproc_per_node=<N> training/extract_video_training_latents.py

# 5. smoke-test
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=1 train.py \
    exp_id=debug compile=False debug=True example_train=True batch_size=1

# 6. baseline + one GW run
bash experiments/run_gw_experiments.sh baseline
torchrun --standalone --nproc_per_node=8 train.py \
    exp_id=gw_global gw_regularization.enabled=true gw_regularization.variant=global
```

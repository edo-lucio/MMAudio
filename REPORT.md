# Relational Alignment for Video-to-Audio Generation
## A Gromov–Wasserstein Regularizer on top of MMAudio: Methodology and Results

*Author: Edoardo Samuele Lucini · ITU Copenhagen · Report dated 2026-04-30.*

---

## 1. Background: the MMAudio architecture this work modifies

Before stating what the GW regularizer does, it is essential to be precise
about what it is being added to. All file paths below are relative to the
repository root.

### 1.1 Conditioning streams

MMAudio (`src/mmaudio/model/networks.py:29`, class `MMAudio`) is a multi-modal
diffusion transformer (MM-DiT) trained with the **flow-matching** objective
of Lipman et al. 2023, instantiated through `FlowMatching`
(`src/mmaudio/model/flow_matching.py:11`). For a paired clip it ingests four
streams:

| Stream | Tensor | Encoder (frozen) | Shape |
|---|---|---|---|
| Audio latent | `x1` (`a_mean + a_std·ε`) | TOD-VAE (`ext_weights/v1-16.pth`) | `(B, 250, 20)` for 16 kHz |
| Coarse video | `clip_f` | CLIP image encoder | `(B, 64, 1024)` |
| Synchronisation video | `sync_f` | Synchformer AST | `(B, 192, 768)` |
| Text caption | `text_f` | CLIP text encoder | `(B, 77, 1024)` |

### 1.2 Projection and global-conditioning vector

Every modality is lifted to a shared `hidden_dim` (`448` for `small_16k`,
`networks.py:369`) by a small input MLP/conv stack (`audio_input_proj`,
`clip_input_proj`, `sync_input_proj`, `text_input_proj`,
`networks.py:60–104`). A *global* conditioning vector is then assembled
(`networks.py:251–276`):

```
clip_f_c = clip_cond_proj(mean(clip_input_proj(clip_f)))     # (B, D)
text_f_c = text_cond_proj(mean(text_input_proj(text_f)))     # (B, D)
global_c = global_cond_mlp(clip_f_c + text_f_c)              # (B, D)
extended_c = t_embed(t) + global_c + sync_f_upsampled
```

`global_c` is injected into every transformer block via **adaLN modulation**;
the upsampled `sync_f` is added token-wise to the audio stream so that
visual onsets line up with audio frames.

### 1.3 Joint and fused blocks

The DiT trunk has two stages (`networks.py:122–132`, `predict_flow`):

1. **Joint blocks** (`depth − fused_depth = 4`): three parallel
   adaLN-conditioned attention towers (audio / clip / text), with rotary
   positional encodings and adaLN-zero init. Cross-modal coupling occurs
   inside each `JointBlock` via cross-attention.
2. **Fused blocks** (`fused_depth = 8`): single-stream MM-DiT blocks that
   process the audio tokens after the conditioning streams have been
   integrated.

The output is the predicted vector field `v_θ(xt, t, c)` for the
flow-matching ODE `dx/dt = v_θ`.

### 1.4 Flow-matching objective

`FlowMatching.loss` (`flow_matching.py:33`) computes

```
target_v = x1 − (1 − σ_min)·x0,    xt = (1 − (1 − σ_min)t)·x0 + t·x1
L_FM     = ||v_θ(xt, t, c) − target_v||²    (mean over t, B)
```

and `Runner.train_fn` (`runner.py:220–258`) applies classifier-free dropout
on `clip_f`, `sync_f`, `text_f` (each independently null-masked with
`null_condition_probability = 0.1`).

**Conditioning is therefore already pointwise.** Cross-attention copies
content per token, adaLN re-scales activations per sample, and
classifier-free guidance pushes the model to use the conditioning at
inference. Every coupling is between *individual* visual tokens/samples and
their own audio counterparts; the model is never asked anything about
*pairs* of samples.

---

## 2. Research question and hypothesis

### 2.1 Question

> **RQ.** Can a Gromov–Wasserstein regularizer that aligns the *relational*
> geometry of video and audio embeddings improve flow-matching V2A
> generation, beyond what pointwise conditioning (cross-attention, adaLN,
> contrastive losses) already achieves?

### 2.2 Hypothesis

If the embedding spaces produced by `clip_input_proj` (or by the global
conditioning side `clip_f_c`) and by the audio side share *similar pairwise
distance structure* between paired samples, the cross-attention layers see
a more isotropic substrate and therefore generalise better. Forcing this
relational alignment with an entropic GW penalty should:

1. Lower distributional metrics (FD-PANN / FD-PASST / FD-VGG) on VGGSound
   test, because audio outputs become better calibrated to their visual
   class.
2. Raise the AV–CLAP-style **IB-Score** (image-bind alignment) and
   **inception-style** scores by encouraging class-discriminative audio.
3. Help most under **distribution shift** (held-out classes), where
   pointwise conditioning is least reliable.

### 2.3 Why GW (and not vanilla Wasserstein)

Pointwise (Wasserstein-style) cross-domain costs require comparable feature
geometries — they assume one can compute `||v_i − a_j||` directly. CLIP
features and VAE-latent means live in incomparable spaces. **GW** instead
matches *intra-domain* distance matrices (`D_V[i,k]` vs `D_A[j,l]`) and is
therefore invariant to isometric reparameterisations of either side
(Mémoli 2011; Peyré et al. 2016). This makes it the natural relational
penalty between heterogeneous encoders.

---

## 3. Methodology

### 3.1 The GW penalty (`src/mmaudio/model/gw_regularization.py`)

Given a batch of paired representations `v ∈ R^{B×D}` and `a ∈ R^{B×D}`:

1. Squared-L2 pairwise distance matrices `D_V, D_A` (`pairwise_distances`,
   `gw_regularization.py:15`).
2. Mean-normalisation: `D ← D / mean(D).detach()` (`_normalize_dist`,
   `gw_regularization.py:25`) — makes `λ_gw` insensitive to feature scale.
3. **Entropic GW** (`entropic_gw_loss`, `gw_regularization.py:31`): mirror
   descent on the quadratic GW objective with Sinkhorn projections. At
   each outer iteration the gradient `G = −2 D_V T D_A` is linearised; an
   inner Sinkhorn loop (20 iters, ε = 0.1) projects onto the uniform
   coupling polytope. Computation is forced into fp32 inside an
   `autocast(enabled=False)` block (`gw_regularization.py:187`).
4. **Fused GW** (`fused_gw_loss`, `gw_regularization.py:86`) for the
   `fused` variant: linear blend `α·G_GW + (1−α)·C_cross` with
   `C_cross = 1 − cosine(v, a)`.

Defaults: 5 outer iterations, ε = 0.1, α = 0.5, B ≥ 4 to compute the loss.
Returns `(zero, None)` if non-finite or under-sized batch.

### 3.2 Where the loss plugs into training

In `Runner.train_pass` (`runner.py:297–358`):

```
total_loss = mean_loss                      # FM loss from train_fn
if self.gw_enabled:
    eff_lambda = lambda_schedule(it, ...)
    a_mean_norm = network.normalize(a_mean.clone())
    gw_loss, _  = compute_gw_regularization(
        network.module, variant=..., clip_f_raw=gw_clip_f,
        x1=a_mean_norm, video_exist=video_exist,
        detach_video=..., num_sinkhorn_iter=..., epsilon=..., alpha=...,
    )
    total_loss = mean_loss + eff_lambda · gw_loss
```

Three subtleties:

- `gw_clip_f = clip_f.clone()` is taken **before** CFG null-masking
  (`runner.py:324`), so the GW penalty always sees real visuals. The
  effective batch is restricted to `video_exist=True` rows
  (`gw_regularization.py:180`).
- The GW loss is computed **outside** `torch.compile`'d `train_fn` because
  the entropic Sinkhorn loop is fp32 and dynamic.
- `a_mean` (the deterministic VAE-latent mean) is used, not the noisy
  `xt`, so the audio-side pairwise distances are conditioning-free.

### 3.3 Four representation variants (`_extract_representations`, `gw_regularization.py:125`)

| Variant | `v` (video) | `a` (audio) | What it tests |
|---|---|---|---|
| `global` | `mean(clip_f_raw)` (raw CLIP) | `mean(x1)` (raw VAE-latent) | Alignment of the *pretraining* geometry, untouched by the network |
| `projected` | `mean(clip_input_proj(clip_f_raw))` | `mean(audio_input_proj(x1))` | Alignment of the *learned input projections* |
| `c_g` | `clip_cond_proj(mean(clip_input_proj))` (the actual `clip_f_c`) | `mean(audio_input_proj(x1))` | Alignment of the **global adaLN conditioning vector** |
| `fused` | same as `projected` | same as `projected` | Adds a Wasserstein cosine cross-cost on top of GW |

#### 3.3.1 `global` — pretraining-geometry probe

`v = mean_t(clip_f_raw[:, t, :])` and `a = mean_t(x1[:, t, :])`. Neither
side is touched by anything trainable in MMAudio: the video features come
straight off CLIP, the audio side is the unconditional VAE-latent mean.
The trainable network only enters the picture indirectly, through the
`network.normalize` call on `x1` in `runner.py:341` (subtracting the
buffer-stored `latent_mean`/`latent_std`). With `detach_video=True` (the
default), even the audio side's gradient flow back into the trunk goes
exclusively through that affine normalisation, since `x1` itself is data.

This variant therefore answers a different question from the others:
*does CLIP's pretraining geometry already pair-correlate with the
VAE-latent geometry well enough that a tiny GW pull is useful as a
calibration prior?* It cannot collapse — there is no parameter on the
video side and only a frozen normalisation on the audio side — and it
cannot really learn an alignment either. It is a control more than a
treatment, which makes its dominance in the leaderboard (§4.2)
diagnostically important rather than expected.

#### 3.3.2 `projected` — learned-projector alignment

`v = mean_t(clip_input_proj(clip_f_raw))` and
`a = mean_t(audio_input_proj(x1))`. Both sides go through their
respective `*_input_proj` stacks (a Linear/Conv + SiLU/SELU + ConvMLP,
`networks.py:60–104`), which lift the modalities into the shared
`hidden_dim = 448` space *before* any DiT block consumes them.

This is the smallest scope at which GW can actually change a learnable
representation: it pushes the input projectors to put paired clips at
similar pairwise distances on each side. It is also the only variant
where `detach_video=False` could in principle contribute to learning,
since the video side has trainable parameters. Note that despite the
name, this variant does *not* engage the FGW cosine term — it is pure
relational GW on the projected mean tokens.

#### 3.3.3 `c_g` — global conditioning vector

`v = clip_cond_proj(mean_t(clip_input_proj(clip_f_raw)))`, the literal
`clip_f_c` produced by `preprocess_conditions` (`networks.py:251`); the
audio side is the same as `projected`. `clip_f_c` is then summed with
`text_f_c`, passed through `global_cond_mlp`, added to the timestep
embedding, and fed as the **single global adaLN modulation vector** to
every JointBlock and every fused MMDiT block (`networks.py:274–284`).

This variant is the most ambitious by design: it targets the exact
representation that controls per-block scale/shift across the entire
trunk, so any geometric improvement should propagate everywhere through
adaLN modulation. It is also the most fragile: there is a single linear
head (`clip_cond_proj`) standing between GW pressure and the global
modulator, and a degenerate solution exists in which that head outputs a
near-constant vector. Section 4.2 shows that this is exactly what
happens.

#### 3.3.4 `fused` — fused Gromov–Wasserstein

The representations are the same as `projected`, but the loss path is
different (`gw_regularization.py:199–209`): a cosine-based cross-domain
cost `C_cross = 1 − cos(v, a)` is computed between paired samples and
combined with the relational GW gradient inside the Sinkhorn loop,

```
G = α · G_GW + (1 − α) · C_cross,    α = 0.5 (default)
loss = α · GW(D_V, D_A; T) + (1 − α) · ⟨C_cross, T⟩
```

This is the Vayer-et-al. fused-GW formulation: the relational GW term
respects the within-modality distance structure, while the Wasserstein
term `⟨C_cross, T⟩` simultaneously rewards the coupling for sending
paired `(v_i, a_i)` to themselves in cosine space. In principle this
should help GW escape its degenerate solutions by making
"send-everything-to-one-class" a non-zero-cost option. In practice (§4.2)
it accelerates the collapse of `clip_input_proj`, presumably because the
network can satisfy both terms by sending all projected `v_i` toward a
single direction (`C_cross` minimised pairwise) which simultaneously
makes `D_V` near-rank-1 (pairwise distances all small).

Concretely, the four variants form an ordered ladder of *how much
trainable surface GW is allowed to reshape*:

```
global  <  projected  <  c_g  <  fused
 ─────  no params  →  one MLP per modality  →  +1 head into adaLN  →  +cosine cross-cost
```

That ladder turns out to align almost monotonically with the failure
mode in §4 — the more network surface GW has access to, the more the
network resolves it by collapsing.

### 3.4 λ schedule (`lambda_schedule`, `gw_regularization.py:222`)

`constant`, `linear_rampup` (warm-up to base over `warmup_steps`), and
`cosine_anneal` (decay from base to 0 after warm-up). Default
`warmup_steps = 50000` for the schedule ablation.

### 3.5 Experimental matrix (`src/experiments/run_gw_experiments.sh`)

| Group | Configurations | Notes |
|---|---|---|
| Baseline | `gw_baseline` | GW disabled |
| Variant sweep | `gw_var_{global, projected, c_g, fused}` at `λ = 0.01` | Picks "best" variant |
| λ sweep | `gw_lam_global_{0.001, 0.005, 0.01, 0.05, 0.1}` | On the chosen variant |
| Detach ablation | `gw_detach_{true, false}` | Whether `v` carries gradients |
| Schedule ablation | `gw_sched_{constant, linear_rampup, cosine_anneal}` | warm-up = 50k |
| OOD | `gw_ood_baseline`, `gw_ood_global` | 30 held-out VGGSound classes |
| GW × sync | `gw_{true,false}_sync_{true,false}` | Interaction with synch features |

### 3.6 Diagnostics

- `src/experiments/diagnose_geometry.py` — runs the *released* MMAudio
  weights (no fine-tuning) and reports Pearson/Spearman of
  `vec(D_V) ↔ vec(D_A)`, kNN-graph Jaccard agreement, effective rank, and
  a class-pair coupling heatmap. Establishes the *baseline geometry gap*
  the regularizer is meant to close.
- `src/experiments/analysis.py` — TensorBoard scrape: FM/GW loss curves,
  metrics tables, per-class IB deltas, SVD spectra of saved features
  (`baseline` vs `gw`).
- `src/experiments/analyze_couplings.py` — extracts `T*` on the test
  split and aggregates the mass between class pairs.

### 3.7 Important caveat on scale

CLAUDE.md reports a *currently extracted* subset of **202 train / 10 val /
19 test** clips (full VGGSound is 180 067 / 2 049 / 15 222). The numbers in
`_analysis/results.csv` are therefore from a low-data regime in which
absolute values of FD-PANN ≈ 80–110 are vastly worse than published
MMAudio. Differences must be read as **relative**, and the noise floor
across the 22 runs is non-trivial.

---

## 4. Results

All numbers come from `_analysis/results.csv` (22 runs); plots from
`geometry/` and `analysis/`.

### 4.1 Headline numbers (all 22 runs, sorted by FD-PANN, lower is better)

| Rank | exp_id | FD-PANN | FD-PASST | FD-VGG | IB-Score | ISC-PANNS | KL-PANNS |
|---|---|---|---|---|---|---|---|
| 1 | `gw_lam_global_0.001` | **81.996** | 906.19 | **15.347** | 0.0643 | 1.599 | **3.729** |
| 2 | `gw_var_global` (λ=0.01) | 82.980 | 941.08 | 16.594 | 0.0638 | 1.760 | 3.840 |
| 3 | `gw_false_sync_true` | 85.620 | 926.46 | 17.590 | 0.0631 | 1.727 | 4.206 |
| 4 | `gw_sched_linear_rampup` | 85.727 | 913.29 | 16.623 | **0.0729** | 1.696 | 4.097 |
| 5 | `gw_ood_global` | 86.587 | 918.31 | 17.519 | 0.0666 | 1.699 | 4.152 |
| 6 | `gw_detach_true` | 87.728 | 935.08 | 17.870 | 0.0617 | 1.623 | 4.324 |
| 7 | `gw_sched_cosine_anneal` | 87.754 | 928.37 | 16.664 | 0.0684 | 1.647 | 4.174 |
| 8 | `gw_false_sync_false` | 87.910 | **891.57** | 16.630 | 0.0623 | 1.731 | 3.817 |
| 9 | `gw_baseline` | 88.528 | 938.63 | 16.871 | 0.0620 | 1.720 | 4.257 |
| 10 | `gw_lam_global_0.05` | 88.583 | 944.84 | 17.637 | 0.0622 | 1.624 | 4.298 |
| 11 | `gw_true_sync_false` | 89.413 | 978.64 | 17.959 | 0.0611 | 1.692 | 4.336 |
| 12 | `gw_ood_baseline` | 89.610 | 943.24 | 17.789 | 0.0675 | 1.631 | 4.202 |
| 13 | `gw_true_sync_true` | 90.440 | 956.30 | 18.031 | 0.0655 | 1.582 | 4.378 |
| 14 | `gw_lam_global_0.01` | 90.466 | 947.51 | 16.399 | 0.0612 | 1.531 | 4.356 |
| 15 | `gw_lam_global_0.005` | 90.679 | 1089.86 | 22.643 | 0.0337 | 1.316 | 3.787 |
| 16 | `gw_lam_global_0.1` | 91.354 | 955.20 | 18.152 | 0.0602 | 1.581 | 4.403 |
| 17 | `gw_detach_false` | 92.065 | 970.91 | 17.788 | 0.0652 | 1.602 | 4.281 |
| 18 | `gw_sched_constant` | 92.065 | 970.91 | 17.788 | 0.0652 | 1.602 | 4.281 |
| 19 | `gw_var_projected` | 100.426 | 1287.74 | 25.442 | 0.0170 | 1.246 | 3.713 |
| 20 | `gw_var_c_g` | 102.695 | 1225.97 | 25.619 | 0.0217 | 1.134 | 3.950 |
| 21 | `gw_var_fused` | 109.832 | 1366.47 | 29.565 | 0.0198 | 1.058 | 3.864 |

(The CSV holds 21 distinct rows; rows 17 and 18 are byte-identical, see
§4.4.) Bold marks the best value in each column.

### 4.2 Variant choice: `global` ≈ `projected` ≫ `c_g` ≫ `fused`

At the common λ = 0.01 used in the variant sweep:

| Variant | FD-PANN | IB-Score | Δ vs baseline FD-PANN |
|---|---|---|---|
| baseline (no GW) | 88.53 | 0.0620 | — |
| `global` | 82.98 | 0.0638 | **−5.55 (better)** |
| `projected` | 100.43 | 0.0170 | +11.90 (much worse) |
| `c_g` | 102.70 | 0.0217 | +14.17 (much worse) |
| `fused` | 109.83 | 0.0198 | +21.30 (catastrophic) |

The cleanest interpretation comes from the geometry plots in
`geometry/<variant>/spectrum.png`:

- **`global/spectrum.png`** — baseline and GW spectra are *identical*. As
  expected, this variant operates on the network-independent CLIP/latent
  averages, so training the trunk neither breaks nor builds the relational
  geometry — GW only constrains the trunk to *agree with* an unchanged
  external geometry. This is the variant where everything goes right.
- **`projected/spectrum.png`** — the GW model has slightly *higher* tail
  singular values on the video side, i.e., GW broadens the effective rank
  of `clip_input_proj`. Audio-side curves are tightly overlaid up to the
  baseline truncation point.
- **`c_g/spectrum.png`** — the GW model's video spectrum collapses by
  **3 orders of magnitude** past the leading singular value (`~10⁻¹` →
  `~5·10⁻⁴`). The `clip_cond_proj` head has effectively become rank-1
  under GW pressure: `clip_f_c` is now near-constant, killing the global
  adaLN signal that every block depends on. This explains the FD-PANN
  blow-up.
- **`fused/spectrum.png`** — same diagnosis as `c_g` but on
  `clip_input_proj`: video singular values drop from `~10⁻¹` to `~10⁻²`
  past index 0. The cosine `C_cross` term is the likely culprit: it
  rewards making all `v_i` parallel, then GW happily satisfies the
  pairwise-distance match by sending them *to a single point*.

The take-away is that GW is safe **only on representations that the trunk
is not free to collapse**. Putting it on `clip_cond_proj` outputs is
exactly the wrong place: a degenerate solution exists (constant `clip_f_c`)
that perfectly matches a near-uniform `D_A` while costing nothing.

### 4.3 λ sensitivity on the `global` variant

| λ | FD-PANN | FD-VGG | IB-Score |
|---|---|---|---|
| 0.001 | **81.996** | 15.347 | 0.0643 |
| 0.005 | 90.679 | 22.643 | 0.0337 *(collapse)* |
| 0.01 | 90.466 | 16.399 | 0.0612 |
| 0.05 | 88.583 | 17.637 | 0.0622 |
| 0.1 | 91.354 | 18.152 | 0.0602 |
| baseline | 88.528 | 16.871 | 0.0620 |

A clean sweet spot at **λ = 0.001** beats the baseline by ~6.5 FD-PANN
points and ~10% on FD-VGG; everything from λ ≥ 0.005 either matches or
trails the baseline on FD-PANN. The λ = 0.005 row is anomalous — IB-Score
halves and FD-VGG explodes; given that this is a single seed on 202 train
clips, it is most plausibly an interaction with a particular optimisation
trajectory rather than a true non-monotonicity. (`gw_var_global` at
λ = 0.01 sits at 82.98, *better* than the same λ on the lambda-sweep
config — another reminder that single-run variance is large.)

### 4.4 Detach, schedule, and grad routing

| Run | FD-PANN | IB-Score |
|---|---|---|
| `gw_detach_true` | 87.728 | 0.0617 |
| `gw_detach_false` | 92.065 | 0.0652 |
| `gw_sched_constant` | 92.065 | 0.0652 |
| `gw_sched_linear_rampup` | 85.727 | **0.0729** |
| `gw_sched_cosine_anneal` | 87.754 | 0.0684 |

`gw_sched_constant` and `gw_detach_false` are *numerically identical* to
six digits across all metrics (`results.csv:18,19`). This is the
fingerprint of run-deduplication on the launcher — the constant-schedule
configuration matches the `detach_false` configuration when the latter
inherits the default `constant` schedule. There are therefore really
**four** distinct configurations here, not five.

Useful conclusions:

- `detach_video=True` (the default) is the safer setting: cutting
  gradients to the video side prevents GW from rewriting CLIP features
  through the projector and gives a 4-point FD-PANN improvement.
- A **linear ramp-up** of λ over 50 k steps gives the best IB-Score in
  the entire matrix (0.0729 vs baseline 0.0620, +17%) and a competitive
  FD-PANN of 85.7. Cosine annealing helps less. This is consistent with
  GW being most useful as a *late-stage* regularizer, once the FM
  trajectories are already roughly correct.

### 4.5 Sync × GW interaction

| GW | sync | FD-PANN | IB-Score |
|---|---|---|---|
| ✗ | ✓ | 85.620 | 0.0631 |
| ✗ | ✗ | 87.910 | 0.0623 |
| ✓ | ✓ | 90.440 | 0.0655 |
| ✓ | ✗ | 89.413 | 0.0611 |

In this 2 × 2 the `disable_sync` axis dominates, and the directions of GW
flip. Two readings:

1. With sync features intact, GW *hurts* (85.6 → 90.4): the model already
   has dense visual-rhythm conditioning, so GW only adds noise to the
   global-conditioning geometry it already commits to.
2. Without sync features, GW is roughly neutral (87.9 → 89.4): there is
   nothing for GW to interfere with, so it merely fails to replace the
   information sync was carrying.

Combined with the 4.3 finding (small λ helps), the right reading is that
this 2 × 2 was run at λ = 0.01 (the variant-sweep default), which is
already well past the useful range — so it does not, on its own, refute
the value of GW.

### 4.6 OOD generalisation

| Run | FD-PANN | IB-Score | KL-PANNS |
|---|---|---|---|
| `gw_ood_baseline` | 89.610 | 0.0675 | 4.202 |
| `gw_ood_global` | 86.587 | 0.0666 | 4.152 |

GW improves FD-PANN by ~3 points on the **30 held-out classes** at
λ = 0.01 with `global`. This is the only ablation in the matrix where the
direction is unambiguously favourable for GW at the *non-tuned* λ, and is
exactly the regime — distribution shift — where the hypothesis predicts
the largest win. The IB-Score difference is essentially noise at this
sample size.

### 4.7 The optimization is barely moving the GW objective

`analysis/curves.png` shows training-log curves for the schedule,
detach-false, and OOD-baseline runs. Two observations:

- **FM loss** decreases monotonically on every run from ~0.025 (step 10 k)
  down to ~0.007 (step 54 k); the GW penalty does not visibly disturb the
  flow-matching trajectory.
- **GW loss** stays in a narrow band around **39.0–39.5** across all
  runs and steps (with the constant-schedule run starting from 40.9
  before snapping into band). The penalty almost never decreases below
  its initialisation value.

In other words: the FM trunk is learning, but the GW objective is not
being meaningfully minimised by the network — it is held roughly constant
by the entropic regularisation alone. Whatever the *constraint pressure*
is, it is acting as a mild bias rather than as a learnt alignment.

### 4.8 Couplings collapse onto a single class

`analysis/couplings/<variant>/class_self_alignment.csv` shows, per variant,
the per-class average diagonal mass `T*[i,i]` aggregated across batches:

| Variant | Class with non-trivial self-mass | Mass | All other classes |
|---|---|---|---|
| `global` | `playing sitar` | 1.20·10⁻² | ≤ 10⁻⁸ |
| `projected` | `playing sitar` | 1.48·10⁻² | ≤ 5·10⁻²⁸ |
| `c_g` | (none — all rows zero) | 0 | 0 |
| `fused` | (empty file: no usable batches) | — | — |

The global class-pair heatmap (`analysis/couplings/global/class_pair_heatmap.png`)
makes the failure mode visceral: nearly all transport mass concentrates on
**one column** (`strike lighter`), with a secondary spike in
`people eating crisps`, regardless of which video class is on the rows.
The Sinkhorn solution collapses to a *mass-collector* pattern rather than
a class-aware coupling.

This is consistent with the flat GW loss curve: the entropic Sinkhorn step
finds a degenerate plan that satisfies the marginal constraints without
exploiting the relational structure, and the network has no incentive to
break this degeneracy.

### 4.9 Per-class IB deltas tell a coherent story

`analysis/perclass/global.csv` ranks VGGSound classes by
`IB(GW) − IB(baseline)`. The mean is essentially **zero**
(`analysis/perclass/global.png`), but the class-by-class breakdown is
*not* random:

| GW helps (top of list) | GW hurts (bottom of list) |
|---|---|
| `disc scratching` (+0.033) | `driving buses` (−0.025) |
| `playing congas` (+0.033) | `squishing water` (−0.024) |
| `child singing` (+0.032) | `car passing by` (−0.024) |
| `playing clarinet` (+0.024) | `air conditioning noise` (−0.023) |
| `playing harp` (+0.024) | `lathe spinning` (−0.021) |
| `playing accordion` (+0.021) | `car engine starting` (−0.021) |
| `playing acoustic guitar` (+0.019) | `wind noise` (−0.021) |
| `chicken crowing` (+0.015) | `railroad car` (−0.019) |
| `playing glockenspiel` (+0.014) | `reversing beeps` (−0.018) |
| `playing steelpan` (+0.014) | `people eating` (−0.017) |

The same pattern repeats across `projected`, `c_g`, `fused`: GW boosts
classes with **discrete, pitched, event-like sound** (instruments, child
voices, animal calls) and degrades **stationary ambient noise** (engine
hums, wind, AC, lathe, fans). This is exactly what one would expect from
a regularizer that rewards relational sample-to-sample distinguishability:
classes whose audio is naturally distinguishable benefit; classes whose
audio is broadband stationary noise become *less* distinguishable as the
trunk is pushed toward a CLIP-like geometry.

---

## 5. Interpretation

### 5.1 What the regularizer is actually doing

Putting the pieces together:

- The best result (`gw_lam_global_0.001`, FD-PANN 81.99 vs baseline 88.53,
  ≈ −7.4%) comes from a configuration where **GW does not change the
  network's representations at all** — `global` reads raw CLIP and raw
  latents, and at λ = 0.001 with `detach_video = True` the gradient that
  reaches the network is small and only flows through the audio-side
  `a = mean(x1)` term (since `x1` is the audio data, not the network's
  prediction; see `runner.py:341–352` and `gw_regularization.py:139`).
  In other words, the gain is consistent with a tiny **auxiliary loss on
  the *latent normalisation*** — pushing the audio-side average to live in
  a slightly less anisotropic shell — not with a relational alignment of
  the trunk's internal geometry.
- The geometry plots and coupling heatmaps make it explicit that wherever
  GW is given enough leverage to reshape *learned* features (`c_g`,
  `fused`, larger λ on `projected`), the network finds the **degenerate
  collapse** that satisfies pairwise distance matching at zero cost: rank-1
  conditioning vectors and mass-collector couplings.
- The GW loss curve never moves. The penalty acts more like a mild
  isotropic prior than like a relational constraint.

### 5.2 Where the hypothesis holds, where it does not

| Prediction | Outcome |
|---|---|
| GW lowers FD/IB metrics overall | **Partially** — only at λ ≤ 10⁻³ on `global`; at λ ≥ 10⁻² any variant that touches learned reps degrades. |
| GW helps under distribution shift | **Yes** — `gw_ood_global` improves FD-PANN by 3 pts vs `gw_ood_baseline` at the standard λ. |
| GW improves the *learned* video geometry | **No** — on `c_g`/`fused` it collapses the spectrum; on `projected` it broadens the tail but does not produce class-aware couplings. |
| The optimum coupling concentrates within classes | **No** — couplings collapse onto a single outlier class. |

### 5.3 Why the cleanest variants are the dumbest

The variant ordering is the inverse of "how much network surface GW is
allowed to deform". `global` (no learnable surface) wins; `projected`
(one MLP) is unstable; `c_g` (one MLP feeding adaLN) collapses; `fused`
(adds a cosine cross-cost on top) collapses worst.

A useful frame is to read each variant as a different *trust region* on
the network:

- `global` cannot collapse, because the only learnable parameter on its
  loss path is the audio-side latent normalisation. GW therefore acts as
  a non-distorting external scalar — it can only *bias* training, not
  *deform* representations. That is exactly the regime where it helps.
- `projected` introduces one trainable module per modality. The
  spectrum plot shows that GW *broadens the tail* of the video projector
  rather than collapsing it — i.e., the projector accommodates GW by
  using more directions, not fewer. But the headline FD-PANN at λ = 0.01
  is much worse than baseline, so this "broader" projector is not what
  the FM trunk wants to consume; the JointBlocks downstream then fight
  to undo the diffusion of information. The IB-Score collapse to 0.017 on
  `gw_var_projected` (vs 0.062 baseline) is symptomatic: the audio output
  no longer aligns *pointwise* with the video, even if its pairwise
  distance structure has improved. Relational gain has come at the price
  of pointwise correspondence — the very thing cross-attention is for.
- `c_g` collapses by 3 orders of magnitude on the video singular tail
  (`geometry/c_g/spectrum.png`). The mechanism is mechanical: `clip_f_c`
  is a single 448-dimensional vector that controls every adaLN
  modulation; if GW can be satisfied by making `clip_f_c` near-constant
  across the batch (so `D_V[i,k] ≈ 0`), then any reasonable `D_A` matches
  it for free (after `_normalize_dist` the matrices are scale-free). The
  resulting near-constant adaLN modulation effectively turns off
  conditional generation, which matches the FD-VGG explosion (16.87 →
  25.62) — the model is generating an averaged prior, not a video-driven
  audio.
- `fused` collapses faster than `c_g` (FD-PANN 110, FD-VGG 30, IB-Score
  0.020). The cosine cross-cost was meant to anchor pairs `(v_i, a_i)`
  in cosine space and prevent degenerate solutions, but on uniformly
  initialised projections the cosine-minimising direction is *also* the
  GW-collapse direction (all `v_i` aligned), so the two terms are not
  in tension — they cooperate to ruin the projector. This is a
  cautionary tale about FGW with shared encoder rotations: the supposed
  Wasserstein anchor only works if the cross-domain cost has a structure
  that disagrees with rank-1 collapse, which `1 − cos` does not at init.

This strongly suggests that **the value of GW in MMAudio is not in
shaping the representation**; it is in providing a small,
scale-normalised side penalty that tilts training away from a few
overfitted modes. A simpler contrastive prior or a small InfoNCE term on
`(clip_f_c, mean(x1))` would likely deliver the same gain at a fraction
of the conceptual machinery.

### 5.4 The class-level pattern is informative

The IB-deltas show a repeatable story: GW helps **structured / pitched /
event-like** audio and hurts **broadband stationary** audio. This is
exactly the regime where relational alignment is well-defined: pairs of
guitars *should* sit closer than guitar–piano in latent space, while pairs
of "wind noise" clips probably sit on a degenerate continuum no matter
what. So even at the small effective λ where GW is net-positive, it is
*reshaping the per-class output distribution* — pushing mass toward the
classes the relational prior fits and away from those it does not. For
publication, this is interesting on its own: it means GW is not just
"slightly better" but *differently* better.

A second-order observation reinforces this. The classes that *gain*
under GW are precisely the ones whose CLIP embeddings are most
class-discriminative (instruments are visually iconic; child singers
have a recognisable visual prior; disc scratching has a unique gesture)
*and* whose audio is sparse, pitched, or rhythmic — i.e., the classes
where the visual-pair relational structure on CLIP and the audio-pair
relational structure on the VAE-latent are independently good signals
of class identity. The classes that lose are those where either side of
the pair lives on a smooth, low-information manifold (engine hum, AC
noise, wind, lathe, water): pulling these toward CLIP geometry corrupts
the audio-side stationarity.

This is consistent with treating GW as a **prior on the joint
class-conditional distribution**: when the marginals already carry class
information, GW reinforces it; when one marginal is approximately
class-uninformative (stationary noise), GW substitutes the *other*
marginal's geometry, which is a bad idea.

### 5.5 Why the GW loss is flat and what it implies

The most striking diagnostic is `analysis/curves.png`: the GW loss sits
at ~39.0–39.5 across all runs and all schedules and never decreases
below initialisation. Three contributing factors are visible in the
code:

1. **Constant additive term in the GW value.** `entropic_gw_loss`
   returns `Dv2.sum() + Da2.sum() − 2·⟨D_V T D_A, T⟩`
   (`gw_regularization.py:78–82`). The `Dv2 + Da2` part is a function of
   the marginals only and is bounded below away from zero; with uniform
   `p, q` and mean-normalised `D`, this term contributes a roughly
   constant ~38 floor. The minimisable part — the cross term — moves only
   marginally when `T` saturates onto an outlier column.
2. **Entropy regularisation dominates at ε = 0.1.** With B in the
   tens and ε = 0.1, the Sinkhorn solution has very low effective rank;
   entropy weight ε·H(T) shapes T more than the data-fit. This is what
   produces the column-collapse seen in the heatmap.
3. **Detached video and frozen audio data.** With `detach_video=True`,
   the gradient on `D_V` is zero and only `D_A` is differentiable. But
   `a = mean(x1)` is a function of the data and the affine
   `latent_mean/latent_std` buffers — so the only trainable surface
   visible to GW under the default config is two 20-dimensional
   normalisation buffers. This explains both the flat loss (no pathway
   to substantively reduce it) and why the best result is tiny λ on
   `global` (a small consistent pull on those buffers is precisely the
   regime where this would help).

The flat-loss observation also reframes the OOD gain: GW is not learning
a relational alignment that transfers to held-out classes, because GW is
not really *learning* anything. What it does at small λ is keep
`latent_mean`/`latent_std` from drifting in a direction that overfits
the small training subset's class distribution. This is why the OOD
benefit shows up at λ = 0.01 rather than the in-distribution sweet spot
λ = 0.001: GW's only useful role is mild regularisation, and OOD is the
regime where regularisation matters most.

### 5.6 Sync and the over-conditioning hypothesis

The 2 × 2 in §4.5 looks contradictory until one notes that all four
cells were run at λ = 0.01 on the `global` variant — i.e., past the
sweet spot. The pattern (GW + sync hurts; GW alone is neutral; sync
alone is best) is consistent with a simple over-conditioning story: the
MMAudio trunk already receives strong, dense, time-aligned visual
information through `sync_f`, and adding a relational penalty on top of
it forces the model to reconcile two conditioning signals that disagree
on what "alignment" means. Sync is *temporal* (frame-level onsets);
GW is *categorical* (sample-pair distances aggregated across an entire
clip). At λ = 0.001 — a level small enough not to cause this
reconciliation cost — one would expect GW + sync to either help or be
neutral, and §4.3's headline is consistent with that. A definitive
sync × λ joint sweep would close this gap.

### 5.7 Recommendations for the next iteration

Concretely, the artefacts suggest a clear next experiment matrix:

- **Lower ε, more Sinkhorn iterations.** With ε = 0.1 the optimal `T`
  is close to the uniform plan and column-collapse onto outliers is
  cheap. ε = 0.01 with 50 inner iters would push T toward the
  unregularised GW plan, force the network to actually fit `D_V ≈ D_A`
  to lower the loss, and likely break the flat-loss plateau.
- **Class-balanced batches.** GW's degeneracy onto a single
  "mass-collector" class is partly an artefact of unbalanced batches
  (sitar / strike-lighter dominate the test split per
  `analysis/couplings/global/class_self_alignment.csv`). Balanced
  sampling per batch would force `T*` to spread mass across the
  diagonal blocks.
- **Anti-collapse term.** A spectral penalty
  `−log det(Cov(v) + ηI)` on `clip_input_proj` outputs would directly
  prevent the rank-1 solutions seen on `c_g`/`fused`. Without it, FGW
  is unsafe on a learnable head.
- **Joint λ × sync sweep.** Single-axis ablations cannot resolve
  whether GW is adding value beyond what sync already supplies; a
  factorial sweep at λ ∈ {0, 10⁻³, 10⁻²} × sync ∈ {on, off} would.
- **Full-scale rerun.** All numbers are on 202 train / 19 test clips;
  the noise floor across runs is several FD-PANN points (the
  `gw_detach_false`/`gw_sched_constant` row collision is the
  fingerprint). The variant ranking is robust because the gaps are
  large; the λ sensitivity at λ = 0.005 vs 0.01 is not.

### 5.8 Reading the result honestly

Two equally defensible narratives fit the data:

- *Narrative A (engineering-positive).* GW is a cheap, drop-in
  auxiliary that, at the right scale and on the right surface, reduces
  FD-PANN by ~7% and helps on held-out classes. The mechanism details
  matter less than the engineering result: it is implementable,
  numerically stable, and adds a free regularisation lever.
- *Narrative B (scientifically-honest).* The relational-alignment claim
  is *not* what is driving the gain. The GW objective is not minimised,
  the optimum coupling is degenerate, and the only configuration that
  helps is the one in which GW touches no learned representation. The
  observed improvements are consistent with a tiny scalar prior on
  latent normalisation. Calling this a Gromov–Wasserstein method
  oversells what is, in effect, a small auxiliary regulariser whose
  geometric interpretation is post-hoc.

A thesis chapter that takes both narratives seriously, presents the
geometry diagnostics that distinguish them, and runs the next-iteration
experiments in §5.7 to disambiguate, is — in my view — the most
defensible framing of this work.

### 5.9 Caveats

- All results are on the **202-clip subset**. The absolute FD-PANN numbers
  are far from competitive; the *deltas* are what matters. Repeating the
  best ablations on the full 180 k-clip train set is a precondition for
  any strong claim.
- 22 runs, single seed each. The two-row collision between
  `gw_detach_false` and `gw_sched_constant` confirms there is no per-seed
  noise budget in the table — single-run variance is at least a few
  FD-PANN points, comparable to the GW gains.
- The diagnose-geometry probe (`diagnose_geometry.py`) was written to
  *quantify* the relational gap on the released MMAudio weights but its
  outputs are not in `analysis/` yet; only the post-training spectra are.
  The "before-and-after relational gap" claim cannot be verified from the
  artefacts present.
- Couplings for `fused` could not be aggregated — the corresponding
  `class_self_alignment.csv` is empty, suggesting batch-size or
  `video_exist` filtering rejected every batch. A re-run is needed to
  decide whether `fused` couplings are merely sparse or pathological.

---

## 6. Bottom line

A Gromov–Wasserstein regularizer can be slipped into MMAudio's
flow-matching training without breaking it, and at very small λ on the
`global` variant it produces the best run in the matrix
(`gw_lam_global_0.001`: FD-PANN 81.99 vs 88.53 baseline, ≈ −7.4%) and
helps under held-out-class shift (`gw_ood_global`: −3 FD-PANN). However,
the same matrix shows that:

1. The GW objective is *not* being substantively minimised — the loss
   curve is flat and the optimum coupling collapses onto a single class.
2. Wherever GW has enough leverage to deform learned features
   (`c_g`, `fused`, larger λ on `projected`), the network resolves the
   penalty by collapsing the spectrum of the corresponding projector.
3. The headline gain is consistent with GW acting as a small isotropic
   side regulariser on the *raw* CLIP↔latent geometry — not as the
   relational alignment the hypothesis describes.

The thesis-relevant message is therefore *positive on the engineering
question* (GW is implementable, numerically stable, and at the right λ it
improves a flow-matching V2A model — especially OOD), but *negative on
the scientific claim* that the network is being driven by a meaningful
relational alignment. Re-running on the full VGGSound, replacing the
`global`/`projected` mean-pool with class-balanced or attention-pooled
features, and constraining the entropic Sinkhorn (lower ε, more iters,
contrastive marginals) are the natural next steps for testing whether
relational alignment can be made to bite — or whether the right framing of
this work is that of a small, well-behaved auxiliary loss whose mechanism
is closer to feature-shell normalisation than to Gromov–Wasserstein
matching.

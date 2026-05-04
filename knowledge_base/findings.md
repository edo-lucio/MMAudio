# Findings — Shared Agent State

Single source of truth for all sub-agents. Each section is append-only;
within a section, entries are timestamped and topic-tagged so multiple
research cycles coexist.

Convention for entry headings: `### YYYY-MM-DD HH:MM — <topic>`

---

## [CODE_FINDINGS]

<!--
The Code Interpreter (`/interpret_code <topic>`) writes here.
Each entry contains: relevant file:line refs, algorithmic shape (pseudocode),
hyperparameter table, metric definitions, inline-comment paper references,
and unresolved questions for the Literature Researcher.
-->

_(no entries yet — run `/interpret_code <topic>` or `/research <topic>`)_

---

## [LIT_FINDINGS]

<!--
The Literature Researcher (`/search_literature <topic>`) writes here.
Each entry contains: list of new BibTeX keys added, consensus / contention
points across papers, identified gaps the thesis can exploit, and answers to
open questions raised by the Code Interpreter.
-->

### 2026-05-03 14:00 — gromov wasserstein optimal transport for cross modal alignment

**Papers added** (16 new)

*Foundational GW theory & computation:*
- `peyre_2016_gromov` — entropic-Sinkhorn GW averaging of kernel/distance matrices; the algorithmic backbone cited inline by `mmaudio/model/gw_regularization.py`.
- `solomon_2016_entropic` — provably convergent entropic-GW solver for soft correspondences; companion to Peyré 2016, supports fp32 numerical stability claims.
- `vayer_2019_optimal` — introduces Fused Gromov-Wasserstein (W + GW); ICML reference for the `fused` representation variant.
- `vayer_2019_fused` — long arXiv companion proving FGW is a metric and giving sample-complexity bounds.
- `vayer_2019_sliced` — Sliced GW reduces cost to O(n log n); scalability alternative to entropic Sinkhorn-GW.
- `scetbon_2022_lineartime` — low-rank-coupling GW reaches O(n²) in general and O(n) when costs are low-rank.
- `zhang_2024_gromov` — Annals of Statistics: entropic GW achieves the parametric n^(-1/2) rate, independent of dimension.
- `rioux_2023_entropic` — first EGW solver with formal convergence guarantees + central-limit theory for empirical EGW.
- `sejourne_2021_unbalanced` — Unbalanced GW (UGW) lifts the equal-mass constraint; future-work pointer for silent / padded segments.
- `memoli_2022_distance` — modern mm-space framework + Gromov-Monge variant; substitute for the unavailable Mémoli 2011 in the *Preliminaries*.

*Cross-modal alignment with GW / OT:*
- `chen_2020_graph` — Graph OT (WD + GWD) as a drop-in regularizer across vision-language tasks; closest published precedent to the thesis recipe.
- `gong_2022_gromov` — GW barycenter for multimodal clustering; demonstrates GW alone aligns modalities without paired data.
- `lu_2025_crossmodal` — FGW (GM-OT) for cross-modal ASR knowledge transfer; direct analogue of the thesis idea applied to speech.
- `sasaki_2025_unsupervised` — GWOT toolbox for unsupervised brain/ANN representational alignment; principled motivation for GW when no map between embedding spaces is known.

*Audio-visual / video-to-audio with OT:*
- `rho_2025_lavcap` — OT-alignment loss + OT-attention block for audio-visual captioning (ICASSP 2025); most recent AV-specific OT precedent.
- `chandra_2025_realign` — Regularized Fused Partial GW for self-supervised procedural-video alignment; +18.9 % F1 over OT-only baselines.
- `you_2025_mover` — Multimodal OT + volume-based regularization across text/video/audio; comparator that uses W (not GW) for cross-modal structure.
- `wang_2024_inverse` — Inverse partial OT for music-guided trailer generation (ACMMM 2024); video↔audio matching in a generative pipeline.

(That is 18 keys; `vayer_2019_fused` and `vayer_2019_optimal` are the two complementary FGW papers, both retained because the methodology section needs the journal-grade theoretical statements separately from the ICML benchmark results.)

**Consensus across papers**
- *GW captures complementary signal beyond pointwise OT/contrastive losses.* Supported by `chen_2020_graph`, `lu_2025_crossmodal`, `chandra_2025_realign`, `gong_2022_gromov` — adding the GW edge-matching term consistently improves over W-only or contrastive-only baselines on retrieval / classification / alignment benchmarks. Mechanism: GW transports the *relational geometry* (intra-domain pairwise distances), which W ignores.
- *Entropic Sinkhorn-based GW is the de-facto solver.* `peyre_2016_gromov`, `solomon_2016_entropic` define it; `zhang_2024_gromov`, `rioux_2023_entropic` give matching theory (parametric rate, stability, CLT). Cubic per-iteration cost is acknowledged but tolerated at typical batch sizes.
- *FGW is the natural object when both features and structure are informative.* `vayer_2019_optimal`, `vayer_2019_fused`, `chen_2020_graph`, `lu_2025_crossmodal`, `chandra_2025_realign` all converge on the W + λ·GW formulation; `λ` analogous to the thesis `lambda_gw` schedule.

**Contention / open trade-offs**
- *Computational cost vs. fidelity.* Standard entropic GW is O(n³) per Sinkhorn-tensor step; `scetbon_2022_lineartime` (low-rank) and `vayer_2019_sliced` (slicing) provide accuracy↔speed trade-offs but lose either expressivity or isometry-invariance. The thesis must choose: stay with entropic-GW at small batch (current) or move to LR-GW / SGW for full-VGGSound scale.
- *Balanced vs. unbalanced.* Standard GW imposes equal mass on the two domains, which is unrealistic for V2A (silent frames, off-screen sounds, padded sequences). `sejourne_2021_unbalanced` provides the fix; not implemented in `mmaudio/model/gw_regularization.py`.
- *Pure GW vs. fused.* `gong_2022_gromov` and `sasaki_2025_unsupervised` argue pure GW suffices for unaligned regimes; `chen_2020_graph` and `lu_2025_crossmodal` argue the W component is needed when paired features exist. The thesis can resolve empirically through the `global` / `projected` / `c_g` / `fused` variants.

**Gaps the thesis can claim novelty on**
- **GW regularization for flow-matching generative V2A is unprecedented.** All cross-modal GW work to date targets discriminative tasks (retrieval, classification, captioning, ASR) or contrastive pretraining (`chen_2020_graph`, `lu_2025_crossmodal`, `you_2025_mover`); the closest generative use is trailer generation via inverse OT (`wang_2024_inverse`), which uses W not GW. No prior work places a GW penalty inside a flow-matching V2A objective.
- **No prior work studies a `lambda_gw` schedule.** Existing FGW work fixes α; the thesis schedules it during training, an unstudied design space.
- **No prior work compares GW representation variants in the V2A setting.** The four-way `global` / `projected` / `c_g` / `fused` ablation is novel — closest analogue is `gong_2022_gromov`'s pure-GW vs. `chen_2020_graph`'s fused, but those are reported in different papers on different tasks.

**Answers to code-side open questions**
- *No `[CODE_FINDINGS]` entries exist yet for this topic*, so this round resolves only inline-code references rather than explicit open questions. The Peyré 2016 reference inline-cited in `mmaudio/model/gw_regularization.py` is now `peyre_2016_gromov`; the entropic Sinkhorn formulation is grounded in `peyre_2016_gromov` + `solomon_2016_entropic`; the fused variant in `vayer_2019_optimal` + `vayer_2019_fused`; and the parametric sample-complexity rate that justifies per-batch EGW estimation comes from `zhang_2024_gromov`.

---

## [DRAFT]

<!--
The Writer (`/write_section <topic>`) writes here.
Each entry records: target .tex file, word count, citation keys used, code
refs used, and any TODOs / MISSING-CITE markers left in the LaTeX.
-->

_(no entries yet — run `/write_section <topic>` or `/research <topic>`)_

---

## [REVIEW_NOTES]

<!--
The Reviewer (`/review_section <topic>`) writes here.
Each entry records: counts of CRITICAL / MINOR / SUGGESTION issues, the
specific lines and fixes for CRITICAL items (already patched in the .tex),
and the open MINOR / SUGGESTION items for the user to address.
-->

_(no entries yet — run `/review_section <topic>` or `/research <topic>`)_

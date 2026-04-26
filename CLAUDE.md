# Thesis Orchestrator

This file is the brain of an agentic thesis-writing system. Sub-agents use it
to know who they are, what they share, and how their outputs combine into a
single LaTeX document.

---

## Project Overview

- **Title**: *Relational Alignment for Video-to-Audio Generation: A Gromov--Wasserstein Regularizer for Flow-Matching*
- **Research question**: *Can a Gromov--Wasserstein regularizer that aligns the relational geometry of video and audio embeddings improve flow-matching video-to-audio generation, beyond what pointwise conditioning (cross-attention, adaLN, contrastive losses) already achieves?*
- **Method**: Add a Gromov--Wasserstein (GW) penalty between intra-video and intra-audio pairwise distance matrices to MMAudio's flow-matching V2A objective, solved via entropic Sinkhorn relaxation in fp32, with four representation variants (`global` / `projected` / `c_g` / `fused`) and a `lambda_gw` schedule.
- **Dataset(s)**: VGGSound (Chen et al. 2020). Source manifests: 180,067 train / 2,049 val / 15,222 test clips. Currently extracted on the cluster: 202 train / 10 val / 19 test (subset cap; expansion via `down.py` + `jobs/extract_features.job`).
- **Author**: Edoardo Samuele Lucini
- **Advisor / Institution**: IT University of Copenhagen

---

## Codebase Map

This project keeps its existing repo layout. Sub-agents read **directly from
the repo root**, not from a `src/` directory. The Code Interpreter should
treat the following as its inputs (everything else under the repo root —
`output/`, `logs/`, `data/`, `ext_weights/`, `thesis/` — is generated, large,
or written by other agents and should not be mined for "method" content).

| Area | Paths | What lives here |
|---|---|---|
| Training entry point | `train.py` | Hydra entry, DDP setup, training loop |
| Core model + losses | `mmaudio/` | DiT, flow-matching, GW regularizer, runner |
| GW regularization | `mmaudio/model/gw_regularization.py` | `compute_gw_regularization`, `entropic_gw_loss`, `fused_gw_loss`, `lambda_schedule` |
| Configs | `config/` | Hydra YAMLs (`base_config.yaml`, `train_config.yaml`, `eval_config.yaml`, `data/base.yaml`) |
| Feature extraction | `training/extract_video_training_latents.py`, `training/extract_audio_training_latents.py`, `training/partition_clips.py` | Precompute CLIP / Synchformer / VAE-latent TensorDicts |
| Experiment scripts | `experiments/` | `analysis.py`, `analyze_couplings.py`, `diagnose_geometry.py`, `make_ood_split.py`, `run_gw_experiments.sh` |
| SLURM jobs | `jobs/` | `train_*.job`, `eval*.job`, `extract_features.job`, `download_vggsound.job`, `launch_all.sh` |
| Data download | `down.py` | yt-dlp + ffmpeg pipeline for VGGSound subset |
| Eval driver | `batch_eval.py`, `demo.py` | Post-training evaluation on VGGSound test set |

**Inline references the Literature Researcher should chase**: comments in
`mmaudio/model/gw_regularization.py` cite Peyré et al. 2016; `mmaudio/model/flow_matching.py`
cites Lipman et al. 2023; `mmaudio/runner.py` references the MMAudio paper
(Cheng et al. 2025).

---

## Agent Roles

Each role is implemented as a slash command in `.claude/commands/`. Agents are
spawned via the `Task` tool from the `/research` orchestrator. They never call
each other directly — coordination flows through `knowledge_base/findings.md`.

| Role | Slash command | Reads | Writes |
|---|---|---|---|
| **Code Interpreter** | `/interpret_code <topic>` | repo paths from the **Codebase Map** above | `[CODE_FINDINGS]` |
| **Literature Researcher** | `/search_literature <topic>` | MCP tools (consensus, scholar-gateway) | `[LIT_FINDINGS]`, `literature/refs.bib`, `literature/summaries/<key>.md` |
| **Writer** | `/write_section <topic>` | `[CODE_FINDINGS]`, `[LIT_FINDINGS]` | `paper/sections/<name>.tex`, `[DRAFT]` |
| **Reviewer** | `/review_section <topic>` | `[DRAFT]`, `[LIT_FINDINGS]`, latest `.tex` | `[REVIEW_NOTES]`, edits `.tex` in place |

Concretely:

- **Code Interpreter** scans the **Codebase Map** paths, extracts methods, algorithms, hyperparameters, metrics, and inline paper references. Cites every finding as `path/to/file.py:line` (real repo path — *not* `src/...`).
- **Literature Researcher** searches Consensus + Scholar Gateway MCPs (≥3 consensus queries + ≥1 scholar-gateway query per topic), adds BibTeX entries, and writes per-paper summaries.
- **Writer** translates the Code Interpreter's methodologies, experiments, and architectures into LaTeX, contextualising and comparing them against the Literature Researcher's findings. Every methodological claim ties back to a `path:line`; every external assertion is cited.
- **Reviewer** validates every claim in the draft against literature and code. Severity tags: `[CRITICAL]` (fixed directly in `.tex`), `[MINOR]` (noted, not fixed), `[SUGGESTION]` (noted).

---

## Shared State

Single source of truth: **`knowledge_base/findings.md`**. Use markdown headings
to delimit sections. Agents append (don't overwrite) within their section.

Tagged sections (always present, in this order):

```
## [CODE_FINDINGS]
## [LIT_FINDINGS]
## [DRAFT]
## [REVIEW_NOTES]
```

Conventions:

- Every entry begins with a `### YYYY-MM-DD HH:MM — <topic>` subheading.
- Code findings cite `path/to/file.py:line_number` using the **real repo path** (no `src/` prefix).
- Literature findings cite the BibTeX key the agent added to `refs.bib`.
- Drafts include the target `.tex` filename in the subheading.
- Review notes include severity tags: `[CRITICAL]`, `[MINOR]`, `[SUGGESTION]`.

---

## MCP Tools

The Literature Researcher uses these. If unavailable in your Claude Code
session, run `/mcp` to install them.

| MCP | Purpose | When to use |
|---|---|---|
| `consensus` | Aggregated peer-reviewed findings | "What does research say about X?" |
| `scholar-gateway` | Semantic search + citation retrieval | Free-text exploration, finding seed papers |

The agent must run **at least 3** consensus queries from different angles and
**at least one** scholar-gateway query per topic.

---

## LaTeX Conventions

- Section files live in `paper/sections/<name>.tex` and are `\input{}`-ed by `paper/main.tex`.
- Citations: `\citep{key}` (parenthetical) and `\citet{key}` (inline as a noun).
- Every key used in `\cite*{}` must exist in `literature/refs.bib`. The Reviewer agent enforces this.
- Math macros are defined in the `paper/main.tex` preamble. Currently available: `\LFM, \LGW, \LFGW, \DV, \DA, \Tstar, \R, \E`. Always use these; never redefine inline.
- Figures: `\includegraphics{figures/name}` (place files in `paper/figures/`, create as needed).
- Build artefacts go to `paper/out/` (configured via `.vscode/settings.json`).

---

## Current Status

- [x] CLAUDE.md filled in (title / question / method / dataset)
- [ ] At least one `/research <topic>` cycle has run end-to-end
- [ ] `paper/sections/introduction.tex` populated
- [ ] `paper/sections/related_work.tex` populated
- [ ] `paper/sections/methodology.tex` populated
- [ ] `paper/sections/results.tex` populated
- [ ] `paper/sections/discussion.tex` populated
- [ ] `literature/refs.bib` ≥ 20 entries
- [ ] All `[CRITICAL]` review notes resolved

---

## Workflow

User runs `/research "<topic prompt>"`. The orchestrator:

1. **Spawns in parallel**:
   - `interpret_code` subagent → writes `[CODE_FINDINGS]`
   - `search_literature` subagent → writes `[LIT_FINDINGS]` + adds entries to `refs.bib`
2. **Waits** for both to complete.
3. **Spawns** `write_section` → drafts `paper/sections/<name>.tex` and appends `[DRAFT]`.
4. **Spawns** `review_section` → validates citations and code grounding, appends `[REVIEW_NOTES]`, edits the `.tex` directly to fix `[CRITICAL]` issues.
5. **Reports** a one-paragraph summary of agent outputs to the user.

Individual agents can also be invoked directly (`/interpret_code`,
`/search_literature`, `/write_section`, `/review_section`) for surgical work.

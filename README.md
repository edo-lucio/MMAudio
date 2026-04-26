# Agentic Thesis-Writing System

A Claude-Code-driven pipeline that turns a topic prompt into a reviewed
LaTeX section, grounded in your own source code and peer-reviewed literature.

---

## TL;DR

```text
/research "<topic prompt>"
```

…spawns four sub-agents (code interpreter + literature researcher in parallel,
then writer, then reviewer) and produces a reviewed `.tex` section.

---

## First-time setup

1. **Fill in `CLAUDE.md`** — replace every `_TODO_` placeholder for title,
   research question, method, dataset, author, advisor.
2. **Confirm the Codebase Map** in `CLAUDE.md` matches your repo layout.
   This project keeps its native layout (no `src/` indirection); the agents
   read directly from the paths listed in that section (e.g. `train.py`,
   `mmaudio/`, `training/`, `experiments/`, `config/`, `jobs/`). If you
   move or rename anything, update the Codebase Map.
3. **Connect MCP servers.** From the Claude Code session:

   ```
   /mcp
   ```

   You need `consensus` (peer-reviewed aggregation) and `scholar-gateway`
   (semantic search). The Literature Researcher will fail with a clear
   message if either is missing.
4. **Optional**: install the LaTeX Workshop extension in VS Code. The
   provided `.vscode/settings.json` builds `paper/main.tex` to `paper/out/`
   on save.

---

## Usage

### Full orchestrator loop

```text
/research "Gromov-Wasserstein regularization for video-to-audio generation"
```

This runs the entire pipeline:

```
   /research
       ├── (parallel) /interpret_code   →  knowledge_base/findings.md  [CODE_FINDINGS]
       └── (parallel) /search_literature →  literature/refs.bib + summaries/ + [LIT_FINDINGS]
                       ↓
                /write_section           →  paper/sections/<name>.tex + [DRAFT]
                       ↓
                /review_section          →  edits .tex in place + [REVIEW_NOTES]
```

### Individual agents

Run any agent on its own when you need surgical work:

| Slash command | When to use |
|---|---|
| `/interpret_code "<focus>"` | You changed code and want findings refreshed |
| `/search_literature "<topic>"` | You want more citations on a specific angle |
| `/write_section "<topic>"` | You already have findings; just want a draft |
| `/review_section "<topic>"` | You hand-edited a draft and want it validated |

---

## The shared-state convention

Everything coordinates through `knowledge_base/findings.md`. It has four
append-only sections, each written by exactly one agent role:

| Tag | Written by | Read by |
|---|---|---|
| `[CODE_FINDINGS]` | Code Interpreter | Writer, Literature Researcher |
| `[LIT_FINDINGS]` | Literature Researcher | Writer, Reviewer |
| `[DRAFT]` | Writer | Reviewer |
| `[REVIEW_NOTES]` | Reviewer | You |

Each entry is timestamped (`### YYYY-MM-DD HH:MM — <topic>`) so multiple
research cycles on different topics coexist cleanly.

---

## File layout

```
.
├── CLAUDE.md                    ← orchestrator brain (FILL IN PLACEHOLDERS)
├── README.md                    ← this file
├── .claude/
│   └── commands/                ← slash-command definitions for sub-agents
│       ├── research.md
│       ├── interpret_code.md
│       ├── search_literature.md
│       ├── write_section.md
│       └── review_section.md
├── .vscode/
│   └── settings.json            ← LaTeX Workshop config
├── knowledge_base/
│   └── findings.md              ← shared agent state
├── literature/
│   ├── refs.bib                 ← BibTeX (auto-populated)
│   └── summaries/               ← per-paper summaries (auto-populated)
├── paper/
│   ├── main.tex                 ← LaTeX root
│   ├── out/                     ← build artefacts
│   └── sections/
│       ├── introduction.tex
│       ├── related_work.tex
│       ├── methodology.tex
│       ├── results.tex
│       └── discussion.tex
│
└── (your codebase, in its native layout)
    ├── train.py
    ├── mmaudio/
    ├── training/
    ├── experiments/
    ├── config/
    └── jobs/                    ← see CLAUDE.md "Codebase Map"
```

---

## Updating `CLAUDE.md`

After your first cycle, keep `CLAUDE.md` in sync:

- Update the `Current Status` checklist as sections fill in.
- If your research question shifts, update the `Project Overview` section —
  every agent re-reads it on each invocation, so changes propagate
  immediately.

---

## FAQ

**Q: Can I run two `/research` cycles in parallel?**
Not safely. Both would append to `findings.md`; interleaved entries are hard
to untangle. Run them sequentially.

**Q: An agent invented a citation that doesn't exist.**
The agents are instructed never to fabricate keys, but if it happens, delete
the bogus entry from `refs.bib` and re-run `/search_literature` with a more
specific topic.

**Q: How do I build the PDF?**
With LaTeX Workshop in VS Code: just save `main.tex`. Without it:

```bash
cd paper
pdflatex -output-directory=out main.tex
bibtex out/main
pdflatex -output-directory=out main.tex
pdflatex -output-directory=out main.tex
```

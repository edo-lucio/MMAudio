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

_(no entries yet — run `/search_literature <topic>` or `/research <topic>`)_

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

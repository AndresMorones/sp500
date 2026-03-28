# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core rule

Read all files in `docs/` before starting any work. Challenge any assumption — nothing in these docs is final. If you think a decision is wrong, say so with reasoning. Update the relevant doc after every major decision or experiment.

## Documentation system

- `docs/objective.md` — The problem we're solving, success criteria, data we have, and constraints. Read this first.
- `docs/journal.md` — Chronological research log. Every decision, dead end, and insight is timestamped. Add new entries at the bottom. Never delete old entries — they explain why things changed.
- `docs/open-questions.md` — Active questions that need resolution, plus answered questions with evidence. Move items between sections as they get resolved. This is where methodology evolves.

When starting a session: read all three docs, check open-questions.md for what needs attention.
When ending a session: update journal.md with what was tried, update open-questions.md if anything was resolved or new questions emerged.

## Project structure

```
sp500/
├── data/
│   ├── raw/          # Source CSVs (price.csv, S&P 500, individual stock histories)
│   └── output/       # Generated outputs (scores_output.csv)
├── docs/             # objective.md, journal.md, open-questions.md
├── src/              # Production pipeline code
│   └── score_pipeline.py
├── sandbox/          # Exploration tools (HTML matrix, Python comparisons)
│   ├── combined_news_impact_matrix.html
│   ├── combined_news_impact_matrix.py
│   └── compare_scoring_options.py
└── CLAUDE.md
```

## Running the pipeline

```bash
python src/score_pipeline.py
```
Outputs `data/output/scores_output.csv`. Uses rolling 120-day beta estimation per ticker.

## Running the visualization sandbox

Open `sandbox/combined_news_impact_matrix.html` directly in a browser — no build step, server, or dependencies required. The only external dependency is Chart.js 4.4.1 loaded from CDN. This is a demo tool for exploring how parameter changes affect scoring — not the production pipeline.

## Architecture of the sandbox

Single-file HTML application (~330 lines) — all CSS, HTML, and JavaScript live in `sandbox/combined_news_impact_matrix.html`.

### Model overview

The tool scores how much a stock's daily move was a "news event" by comparing it against two baselines:

1. **vs Index**: How far the stock deviated from what its beta model predicted given the index move.
2. **vs Own History**: How unusual the move was relative to the stock's own typical daily behavior.
3. **Volume confidence**: How much trading volume confirms or dampens the price signal.

The asymmetric beta model uses a soft split (`splitM`) to separate up-market beta (`bu`) from down-market beta (`bd`), avoiding a hard threshold at zero. `K=50` controls the sharpness of this split.

### Key functions

- `mu(rm)` — expected stock return given index return `rm`, using `alpha + bu*pos + bd*neg`
- `calc(rs, rm)` — returns all metrics for a (stock move, index move) scenario
- `volMult(vratio)` — volume confidence multiplier: `max(0.5, 1 + 0.5 * ln(V/V_avg))`
- `scoreVal(c, m)` — selects which score to display based on the active radio button
- `render()` — rebuilds the entire matrix table, expected-move row, row buttons, and chart on every parameter change

### Current scoring formulas (subject to change — see docs/open-questions.md)

```
vs Index score    = sign(z_i) * z_i²
vs Own score      = sign(z_o) * z_o²
vol_multiplier    = max(0.5, 1 + 0.5 * ln(V / V_avg))
Combined score    = sign(z_i) * |z_i| * |z_o| * vol_multiplier
```

### Parameter system

`P` is an object keyed by parameter name. Each entry holds the slider/text IDs and `toD`/`frD` functions for converting between raw float values and formatted display strings (e.g., `0.009 ↔ "0.90%"`). The `wire(key)` function sets up bidirectional sync between each slider and its text input.

### State

- `selRow` — index into `MV` for the currently selected stock row in the profile chart
- `hovC` — `[rowIdx, colIdx]` of the currently hovered cell, or `null`
- `profChart` — the Chart.js instance; destroyed and recreated on each `render()` call
- `MV = [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]` — the fixed set of move percentages for both axes


## Documentation maintenance rules

### When to update docs/journal.md
- After trying something new (worked or failed), add a timestamped entry at the bottom
- Format: `## Entry N — [short title]` followed by what was tried, what happened, and why
- NEVER delete or modify old entries — only append
- If an old entry is now known to be wrong, add a new entry that references it: "Supersedes Entry N"

### When to update docs/open-questions.md
- When a new question emerges during work, add it under "Active questions" with a QN number
- When an active question gets resolved, move it to "Answered questions" with the evidence and date
- When you disagree with an answered question, reopen it: move it back to active with a note explaining why
- Keep the numbering sequential — never reuse a Q or A number

### When to update docs/objective.md
- Only when the user explicitly changes the project scope or success criteria
- Never modify silently — always confirm with the user first

### Auto-update trigger
At the end of every session where code was written or a methodology decision was made, 
ask: "Should I update the docs?" Then:
1. Propose the specific changes (show diff)
2. Wait for user confirmation
3. Write the changes

### Format for journal entries
```
## Entry [N] — [Title] — [YYYY-MM-DD]

**Tried**: [what was attempted]
**Result**: [what happened]  
**Decision**: [what we decided and why]
**References**: [papers, data, or prior entries that support this]
```
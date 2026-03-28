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
│   ├── raw/          # Source CSVs (price.csv, news.csv, S&P 500, individual stock histories)
│   └── output/       # Generated outputs (scores_output.csv, baseline_predictions.csv)
├── docs/             # objective.md, journal.md, open-questions.md
├── src/              # Production pipeline code
│   ├── common.py            # Shared constants, math helpers, OLS, scoring functions, data loaders
│   ├── score_pipeline.py    # Main scoring pipeline — produces scores_output.csv
│   ├── score_analysis.py    # Metric evaluation (overlap, Cohen's d, Precision@K, forward returns)
│   ├── rank_divergence.py   # Per-ticker rank comparison between metrics
│   ├── beta_window_test.py  # Out-of-sample beta window and model comparison
│   └── model_baseline.py    # Stage 2 baseline models (Ridge, LASSO, LightGBM)
├── tests/
│   └── test_smoke.py        # Pipeline invariant checks (run with pytest or standalone)
├── sandbox/          # Exploration tools — not production
│   ├── combined_news_impact_matrix.html
│   ├── combined_news_impact_matrix.py
│   ├── compare_scoring_options.py
│   └── Notes
├── requirements.txt  # All dependencies — update whenever a new import is added
├── .gitignore
└── CLAUDE.md
```

## Dependency management

All non-stdlib imports must be listed in `requirements.txt`, grouped by the script that needs them. When adding a new import to any script, update `requirements.txt` in the same commit.

## Running the pipeline

```bash
python src/score_pipeline.py
```
Outputs `data/output/scores_output.csv`. Uses rolling 120-day single beta estimation per ticker. Two return types per day: gap (prev close -> open) and close-to-close (prev close -> close).

## Current scoring model

Five scoring functions are computed for each return type (gap, cc):

| Metric | Formula | Role |
|--------|---------|------|
| **A** (primary) | `sign(zi) * zi^2` | Pure standardized abnormal return, squared. Best at top-K precision and most consistent across tickers (Entry 11, 15). |
| D | `sign(zi) * zi^2 * max(1, \|zo\|)` | A + own-history amplifier. Nearly redundant with A (Entry 11). |
| E | `sign(zo) * zo^2` | Own-history only. Weaker predictor than market model (Entry 12). |
| Ev | `sign(zo) * sqrt(zo^2 + zv^2)` | Mahalanobis own+volume. Best Cohen's d but high per-ticker variance (Entry 15). |
| Dv | `sign(zi) * sqrt((zi^2 * max(1,\|zo\|))^2 + zv^2)` | Mahalanobis A+own+volume. Best signal persistence (Entry 16). |

Where:
- `zi` = (stock_return - expected_return) / residual_std — abnormal return z-score vs index
- `zo` = (stock_return - rolling_median) / own_std — z-score vs own history
- `zv` = (ln(volume) - rolling_mean_ln_vol) / std_ln_vol — volume z-score

All metrics are stored as separate columns. A is the primary metric for news detection (confirmed by Entries 11, 14, 15, 16).

## After modifying the pipeline

Run this checklist after any change to scoring logic or data processing:

```bash
# 1. Run pipeline — expect 0 NaN, ~1500+ rows, all 7 tickers
python src/score_pipeline.py

# 2. Run metric evaluation — check overlap matrices, precision@K
python src/score_analysis.py

# 3. Run rank divergence — check for anomalies per ticker
python src/rank_divergence.py
```

**Expected invariants:**
- 0 NaN values in output
- All 7 tickers present: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- AAPL beta_gap roughly 1.0-2.0, beta_cc roughly 0.9-1.2
- Row count > 1500 (currently ~1567)
- Gap precision@10 for A metric > 90%

If any invariant breaks, investigate before proceeding.

## Experiment standard

Every experiment must follow this protocol:

### Before running
1. State the hypothesis as an open question in `docs/open-questions.md` (if not already there)
2. Define the success criterion BEFORE running — what result would make you adopt vs reject the change?
   - Example: "If precision@10 > 80%, adopt" or "If MAE improves by >5%, switch"
3. Write the experiment script in `src/` (not sandbox — experiments produce evidence, not toys)

### After running
4. Record raw results in `docs/journal.md` using the entry format below
5. Evaluate against the pre-committed criterion — do not move the goalposts after seeing results
6. Tag the commit: `git tag entry-N-short-description` so the exact code + data state is reproducible
7. Update `docs/open-questions.md` — resolve, update, or add new questions based on findings

### Never do
- Change the success criterion after seeing results (p-hacking)
- Delete or modify old journal entries (append "Supersedes Entry N" instead)
- Run experiments without recording them — even failed/boring ones contain information

## Documentation maintenance rules

### When to update docs/journal.md
- After trying something new (worked or failed), add a timestamped entry at the bottom
- Format: `## Entry N — [short title]` followed by what was tried, what happened, and why
- NEVER delete or modify old entries — only append
- If an old entry is now known to be wrong, add a new entry that references it: "Supersedes Entry N"

### When to update docs/open-questions.md
- When a new question emerges during work, add it under "Active questions" with a QN number
- When an active question gets resolved, move it to "Answered questions" with evidence and date
- Delete the active entry entirely once answered — don't leave redirects (the answer lives in Answered)
- When you disagree with an answered question, reopen it: move it back to active with a note explaining why
- Keep the numbering sequential — never reuse a Q or A number

### When to update docs/objective.md
- Only when the user explicitly changes the project scope or success criteria
- Never modify silently — always confirm with the user first

### Auto-update trigger
Always update docs automatically after code is written or a methodology decision is made. Do NOT ask — just do it.
1. Update journal.md with what was tried and the result
2. Update open-questions.md if anything was resolved or new questions emerged
3. Update objective.md only when the user explicitly changes scope

### Format for journal entries
```
## Entry [N] — [Title] — [YYYY-MM-DD]

**Tried**: [what was attempted]
**Result**: [what happened]
**Decision**: [what we decided and why]
**References**: [papers, data, or prior entries that support this]
```

## Metric evaluation angles

When evaluating whether a metric or formula change is an improvement, use multiple lenses (not just one):

| Evaluation | What it tests | Script |
|------------|---------------|--------|
| News presence (hit rate) | Does this metric's top-K contain more news days? | `score_analysis.py` |
| Cohen's d | Does this metric separate news vs no-news days by magnitude? | `score_analysis.py` |
| Precision@K curve | How does hit rate degrade as K increases? | `score_analysis.py` |
| Per-ticker consistency | Does it work equally well across all 7 tickers? (std of rates) | `score_analysis.py` |
| Forward returns | Do extreme events continue or revert? (information vs noise) | `score_analysis.py` |
| Rank stability (bootstrap) | If you drop 10% of data randomly, does the top-50 change? | not yet built |
| Cross-temporal validation | Train on first half, score second half — does ranking hold? | not yet built |
| Sentiment correlation | For news days, does score magnitude correlate with headline sentiment? | requires NLP (stage 2) |

Build new evaluations as separate functions in `score_analysis.py`. Keep them modular — each should print its own section header and be independently runnable.

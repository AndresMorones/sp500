# Open questions

Questions that need resolution. Challenge these — propose alternatives, run experiments, cite papers.

---

## Active questions

### Q1: Is the volume multiplier formula the right shape?

Current: `vol_mult = max(0.5, 1 + 0.5 * ln(V/V_avg))`

This was chosen for reasonable properties (log scaling, floor at 0.5, neutral at 1x volume). But it was not derived from data. Alternatives to test once we have real volume data:
- Fit the multiplier empirically: for known news days, what's the relationship between volume ratio and news impact?
- Use quantile-based thresholds instead of a continuous function
- Should the 0.5 coefficient be different? Should the floor be different?
- Does the multiplier shape differ between close-to-close and gap periods?

### Q2: Should we use Fama-French factors instead of single-index?

Current model uses only S&P 500 as the market factor. Fama-French 3-factor (market, size, value) or 4-factor (+momentum) could explain more variation, making the residual AR a cleaner news signal.

Counter-argument: Kothari & Warner note that even with 50% beta misestimation, the error is small relative to 1%+ abnormal returns we're detecting. Multi-factor adds complexity and data requirements for marginal improvement in short-horizon studies.

Test: compare AR distributions with single-index vs 3-factor. If the ranking of top-20 outsized days barely changes, single-index is sufficient.

### Q3: Should we separate positive and negative news impact scores?

Current combined score preserves sign (positive = outperformance, negative = underperformance). But the downstream model might need to treat these differently — a +500 bps AR from an earnings beat and a -500 bps AR from an FDA rejection might require different architectures for news prediction.

Consider: keep the signed score for ranking, but add |combined| as a direction-agnostic "something happened" feature.

### Q4: Is p=2 really better than p=1.5 for our goal?

We chose p=2 because we're ranking days, not computing moments. But we haven't tested whether p=2 actually produces a better ordering for news detection than p=1.5.

Empirical test needed: once we have labeled news days, compare recall@20 (of the top 20 scored days, how many had identifiable news?) at p=1.0, p=1.5, p=2.0. If p=1.5 and p=2.0 give identical recall, use p=1.5 for robustness.

Note (2026-03-28): Entry 22's corrected cc Precision@K shows metrics do NOT converge by K≥50 — cc holds 75-81% even at K=500, well above the ~61% baseline rate. The earlier convergence finding (Entry 15) was an artifact of incomplete cc news matching. Cohen's d remains small for all metrics (max 0.180 for Ev). The formula still matters less than expected for gap, but cc now shows meaningful separation at all K values. Still worth testing p directly.

### Q13: MOVED TO ANSWERED → A13

### Q14: MOVED TO ANSWERED → A14

### Q15: How many of the 14 LLM dimensions actually help prediction?

No paper in the literature uses more than 3 news features per stock. We extract 14 dimensions. With ~600 articles per stock, risk of overfitting is real — the model may latch onto noise in rarely-occurring dimension combinations.

Ablation needed:
- Round 1: 3 dimensions only (severity, direction, surprise)
- Round 2: add temporal_horizon, materiality, information_density
- Round 3: all 14
- Compare: does marginal improvement justify additional dimensions?

Paper 2 (ICIAAI 2025 Tesla) found some news categories HURT prediction — more features is not always better.

Note (2026-03-31): Entry 39 confirms the "more features = worse" finding for sentiment features specifically. Config F (10 features) was worst; Config C (1 sentiment scalar) was best. However, these were all sentiment-derived features through an LSTM — the ablation for LLM structured dimensions through Ridge (Entry 38's architecture) is still needed.

### Q8: Should the directional contradiction term be added?

The deep research recommended a contradiction term: c_t = max(0, -a_t * e_t), where a_t is the actual standardized move and e_t is the expected standardized move. This fires when the stock moved opposite to what beta predicted.

Current position: deferred because z_AR already captures this implicitly. But there's a case for it: z_AR treats "moved less than expected in the same direction" and "moved opposite to expected" as the same magnitude. A contradiction term would specifically amplify the opposite-direction case.

Test: once we have data, check whether opposite-direction days have systematically different news characteristics than same-direction misses.

### Q9: Why do ~20-31% of top/bottom-100 extreme days have no news?

Entry 22 (corrected): A_gap top/bottom 100 has ~31.5% no-news days, A_cc top/bottom 100 has ~20.5% no-news days. CC is better because it captures both gap and intraday news windows. Two hypotheses:
1. **Incomplete news coverage**: the news.csv dataset may not capture all relevant articles (especially pre-market/after-hours wire reports, analyst notes, SEC filings).
2. **Non-news extreme moves**: some extremes are driven by options expiration, index rebalancing, or liquidity events that don't generate headlines.

Test: manually inspect 20 random no-news extreme days — search for news via external sources. If most have identifiable events missing from our dataset, it's a coverage gap. If many are genuinely news-free, we need a microstructure filter.

### Q10: Why do bottom-100 extreme events universally revert?

Entry 16 found that all metrics' most negative scores show strong positive excess returns over 1-5 days (+55 to +93 bps cumulative by day+5). This is reversal, not continuation. Possible explanations:
1. **Overreaction**: markets overshoot on bad news and correct over subsequent days
2. **Liquidity-driven extremes**: some large drops are caused by forced selling (margin calls, fund redemptions) that reverse once selling pressure subsides
3. **Asymmetric information incorporation**: bad news may be priced in faster (panic) while the correction is gradual

This matters because if we use forward returns as training signal, negative-score days would confusingly show positive forward returns. The downstream model may need asymmetric treatment of positive vs negative events.

### Q11: Should the prediction target be cumulative 3-5 day returns instead of day+1?

Entry 16 showed day+1 continuation rates near 50% (coin-flip) for all metrics, but cumulative returns at day+3 to day+5 show clear directional drift for positive events. This suggests the information signal takes 2-3 days to fully incorporate. Using day+1 returns as labels for the downstream model may add noise; cumulative 3-5 day excess returns may be better training targets.

Note (2026-03-28): Entry 25 tested this. **Confirmed**: cum3d_excess Ridge achieves CW p=0.002 (highly significant) vs p=0.132 for day+1 Ridge. The 3-day horizon captures information incorporation that day+1 misses. However, Huber loss hurts cum3d (CLT makes 3-day returns less fat-tailed). Use MSE-based Ridge for cum3d, Huber for day+1. Both targets should be used — they capture different signals.

### Q12: Can the baseline be improved with longer test period or different features?

### Q12: Can the baseline be improved with longer test period or different features?

Entry 21 baseline: Ridge R²_OOS=1.4%. Entry 24 tested 6 literature-validated improvements. LightGBM best R²_OOS (5.6% on cc_excess). Entry 25 added multi-target: cum3d Ridge R²_OOS=2.9% (CW p=0.002), gap Ridge R²_OOS=2.4% (CW p=0.054).

Note (Entry 26): Evaluation reframed to return/price prediction, not direction. Key finding: models shrink predictions to ~20% of actual return variance. Tail days (top 20% extreme) have 4.5-4.9x higher MAE than core days. This is the gap news features must close.

Remaining improvements to test:
- More historical data if available
- More tickers beyond mega-cap tech
- Bear/volatile market regime testing
- Tail-focused loss functions (asymmetric loss penalizing extreme-day misses more)

---

## Answered questions

### A1: Fixed vs conditional sigma for normalization

**Answer**: Fixed (unconditional) sigma.

Conditional σ(rm) = γ₀ + γ₁·|rm| broke the matrix ordering — denominator grew faster than numerator at extreme market moves, making a -5% stock with +5% index score LOWER than with 0% index. This is backwards for news detection.

Evidence: visual inspection in the interactive matrix, comparing fixed-σ and conditional-σ views. See journal entry 3.

### A2: How to handle volume — separate signal or multiplier?

**Answer**: Multiplier.

Using max(|z_own|, |z_vol|) as an alternative signal was wrong because it treated high-volume/low-price and high-price/low-volume identically, and both scored high. Volume as a multiplier means: price signal determines the score, volume determines confidence in that score. Low volume dampens (0.5x floor), high volume amplifies (log-scaled).

Evidence: scenario analysis. See journal entry 6.

### A3: Linear AR vs non-linear scoring

**Answer**: Non-linear (p=2) for the ranking score, keep linear AR in bps as a separate column.

Linear AR is the correct economic magnitude ("this day had +500 bps of news impact"). But for ranking which days to investigate, sign(z)·z² better separates genuine events from noise. Both are stored as columns — AR for interpretation, score for ranking.

Evidence: the inverse cubic law (Gopikrishnan 1998) means extreme events are far more common than Gaussian predicts but still rare enough that non-linear amplification helps identify them. The p=2 connection to chi-squared testing in event studies (Patell 1976, Boehmer et al. 1991) provides theoretical grounding.

### A4: Should z-score (AR/σ) be a separate view?

**Answer**: No. Removed from the visualization.

z = AR / σ₀ is just AR divided by a constant (since σ₀ is the same for every cell). The resulting matrix has identical shape to AR in bps, just rescaled. It adds zero information and clutters the interface.

### A5: What softplus centering to use?

**Answer**: Subtract ln(2)/K from rm_pos, add ln(2)/K to rm_neg.

Without centering, at rm=0 the split produces +ln(2)/K and -ln(2)/K instead of zero. When beta_up ≠ beta_down, this creates a bias: expected(0) ≠ alpha. The centering fix ensures the expected stock move at zero index move equals alpha, which is the model's intended behavior.

### A6: Intraday vs gap — should they get different models?

**Answer**: Replaced intraday (open-to-close) with close-to-close (prev close to close). Kept gap as a separate signal.

The intraday return missed gap-triggered events: news causing a -5% gap followed by +3% intraday recovery would score the intraday as +3% (missing the news). Close-to-close captures the full day's verdict (-2% net). Gap is kept because it isolates the initial market reaction to overnight news — a complementary signal.

Evidence: MacKinlay 1997 uses close-to-close as the standard event study return. Entry 10 validated close-to-close for beta estimation. See journal Entry 13. Resolved 2026-03-27.

### A7: What NLP model / approach for news feature extraction in Stage 3?

**Answer**: LLM structured extraction, not embeddings or sentiment scores.

Literature review (Entry 27) found three approaches ranked by effectiveness:
1. **Structured extraction** (LLM extracts event type, severity, duration, etc.) — best (arXiv:2512.19484)
2. **Embeddings** (FinBERT/BERT/LLM hidden states) — moderate
3. **Sentiment scores** (VADER/FinBERT pos/neg/neu) — weakest

Our implementation (news_scorer.py) uses Claude to extract 14 structured dimensions + 8 per-ticker business categories per article. This is the richest approach in the literature — no paper extracts more than 3 news features.

FinBERT (2019) is no longer SOTA for sentiment — Llama-3-70B (79.3%) and DeBERTa (75%) outperform it (59.7%) — but this is moot since we bypass sentiment entirely with structured extraction.

Evidence: 6 papers reviewed in docs/stage3_methodology_review.md. Resolved 2026-03-28.

### A8: What are the inherent limits of price-only return prediction?

**Answer**: Price-only models can predict normal days reasonably but fundamentally cannot predict extreme return days. This is not a modeling limitation — it is an inherent property of what price data contains.

Evidence (Entry 26):
- Prediction range: [-159, +150] bps vs actual range [-939, +2170] bps. Models use only ~20% of the actual return variance.
- Tail/core MAE ratio: 4.5-4.9x. The 20% most extreme days have nearly 5x higher prediction error than the 80% normal days.
- Best R²_OOS: 5.6% (LightGBM on cc_excess), which is actually strong by academic standards (GKX 2020 reports 0.4% monthly for full cross-section).

Why this is inherent: Extreme days are driven by news events (earnings, product launches, regulatory actions). Past prices contain no information about tomorrow's news. Regularization correctly learns to shrink toward the mean since predicting extremes from price alone would just add noise.

**Implication for Model 2**: The value of news features is specifically in widening predictions on news days — reducing the tail/core MAE ratio from ~4.8x toward ~2.5x. Overall R²_OOS may improve modestly, but the tail improvement is what matters. Resolved 2026-03-28.

### A13: Should the prediction target be raw closing price or excess return?

**Answer**: Excess return (via Metric A), not raw price or raw return.

Entry 39 tested raw price and raw return prediction with LSTM + sentiment. Both targets fail to beat naive. Raw price LSTM learns "predict ≈ yesterday" with noise. Raw return LSTM predicts worse than "predict 0% change." Neither captures useful signal.

Entry 38 showed that Metric A (abnormal return z-score) fed to Ridge achieves 0.631% gap MAPE and 0.967% cc MAPE — beating naive (0.632% and 0.998%). The excess return target removes the market component, letting the model focus on stock-specific news signal. Raw price/return targets force the model to also predict market-level moves, which is pure noise at this scale.

Evidence: Entries 38, 39. Resolved 2026-03-31.

### A14: What downstream model should consume the LLM-structured news features?

**Answer**: Ridge. LSTM is a dead end for sentiment-based prediction at this scale.

Entry 39 ran a controlled experiment: 6 feature configs × 5 sentiment models × 2 targets × 5 seeds × 7 tickers. No LSTM configuration beat the naive baseline, regardless of features or sentiment model. Direction accuracy was ~50% (coin flip). The sentiment model choice (FinBERT vs DeBERTa vs Gemma vs Qwen vs Llama-FinSent) made <0.1% MAE difference.

Entry 38 confirmed Ridge + Metric A features beats naive on GOOGL. The bottleneck is not the downstream architecture — it's the feature quality. A single sentiment scalar provides no useful signal. The 30 LLM-extracted structured dimensions (from news_scorer.py) provide genuine predictive value that even Ridge can exploit.

Evidence: Entries 38, 39, `src/lstm_feature_experiment.py`, `src/lstm_sentiment_compare.py`. Resolved 2026-03-31.

### A7: Which scoring metric to use for news impact identification?

**Answer**: A (pure SAR: sign(zi) × zi²) as primary metric.

Evaluated A, D, E, Ev, Dv across six tests: news presence, precision@K, Cohen's d, per-ticker consistency, cumulative forward drift, and continuation rates at K=20/50/100. A wins on extreme-event precision (95% at gap K=10), per-ticker consistency (std 7.8%), and cumulative drift at K=100 (89.8 bps). D is redundant (rho=0.999). Ev has best Cohen's d but weakest drift. Keep zi, zo, zv as separate feature columns for the downstream model.

Evidence: Entries 14-18. Resolved 2026-03-27.

### A16: Should Stage 3 include a naive "predict yesterday's close" baseline?

**Answer**: Yes, absolutely. The naive baseline is essential.

Entry 38 Phase 3 results proved this: GOOGL naive MAPE is 0.632% (gap) and 0.998% (cc). Price LSTM achieved 2.638% and 2.541% — nearly 4x worse than naive. Without the naive baseline, LSTM's 2.5% MAPE looks reasonable in isolation but is actually terrible. Only Metric A Ridge (0.631% gap, 0.967% cc) meaningfully beat naive. Every paper reporting MAPE without a naive comparison is suspect.

Evidence: Entry 38, `phase3/results/price_naive_results.csv`. Resolved 2026-03-30.

### A17: Should Stage 3 use LSTM or tree models (LightGBM) as primary downstream architecture?

**Answer**: Ridge. Neither LSTM nor LightGBM outperform Ridge at this data scale.

Entry 38 tested all three on GOOGL with 47 test days. Ridge beat both LSTM and LightGBM on both tracks. Key reasons: (1) With ~170 training samples, complex models overfit. (2) Price LSTM/LightGBM severely overfit raw dollar prices (2.5-3.3% MAPE vs 0.63% naive). (3) Metric A LSTM performed well (0.71%) but Ridge was still better (0.63%). (4) LightGBM's tree structure has no "tomorrow ≈ today" inductive bias, making raw price prediction fundamentally unsuited.

At larger data scales (>1000 samples), LSTM may outperform per Paper 3 (arXiv:2508.04975). But for our ~240 trading days per ticker, Ridge is the right choice.

Evidence: Entry 38, `phase3/compare.py`. Resolved 2026-03-30.

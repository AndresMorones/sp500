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
- Does the multiplier shape differ between intraday and gap periods?

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

### Q5: Rolling window size — is 60 days optimal?

We inherited 60 from the initial pipeline (standard in event study literature for the estimation window). But:
- Shorter (30-40 days): adapts faster to regime changes, more NaN warmup
- Longer (90-120 days): more stable estimates, misses structural breaks

Test: compare beta stability and AR distributions at window=40, 60, 90. If the stock's beta genuinely shifts over time (e.g., biotech entering a new trial phase), shorter windows catch it.

### Q6: Intraday vs gap — should they get different models?

The pipeline computes separate intraday and gap returns. Currently they use the same formula structure with different fitted parameters. But gap returns might have fundamentally different properties:
- Overnight news is the primary driver of gaps
- Gap volume doesn't exist in the same way (pre-market is thin)
- The volume multiplier may not apply to gaps at all

Consider: for gap returns, drop the volume multiplier and rely only on z_AR and z_own.

### Q7: What NLP model for news embeddings in stage 2?

Not immediate but worth tracking. Literature suggests:
- FinBERT (domain-adapted BERT) — standard baseline for financial sentiment
- GPT-4 with financial prompts — potentially better but expensive
- Hybrid: FinBERT embeddings as features into a lighter model

Stanford CS224N research warns: "only a small part of the variance in financial data can be explained by news" — the signal-to-noise ratio is inherently low. The quality of our stage 1 scoring directly determines how much signal the NLP model has to work with.

### Q8: Should the directional contradiction term be added?

The deep research recommended a contradiction term: c_t = max(0, -a_t * e_t), where a_t is the actual standardized move and e_t is the expected standardized move. This fires when the stock moved opposite to what beta predicted.

Current position: deferred because z_AR already captures this implicitly. But there's a case for it: z_AR treats "moved less than expected in the same direction" and "moved opposite to expected" as the same magnitude. A contradiction term would specifically amplify the opposite-direction case.

Test: once we have data, check whether opposite-direction days have systematically different news characteristics than same-direction misses.

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

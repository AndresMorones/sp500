# Research journal

Each entry is a decision, dead end, or insight. Never delete entries — they explain why things changed.

---

## Entry 1 — Semantic target matrix

The scoring system should produce values that match this intuitive behavior:

```
Stock \ Index    Big down    Small down   Flat    Small up    Big up
Big down stock   Normal-ish  Bad          Bad     Very bad    Worst case
Small down       Resilient   Slightly bad Mild bad Bad        Very bad
Flat stock       Good        Slight good  Neutral Slight bad  Bad
Small up stock   Very good   Good         Good    Slight good Weak if mkt surged
Big up stock     Best case   Very good    Very good Good      Normal if mkt surged
```

Key property: for a fixed stock move, the score should get *more negative* as the index moves *more positive* (because the expected stock move also rises, making the same realized move worse by comparison). The diagonal should be near-neutral.

## Entry 2 — Asymmetric beta model chosen

Standard CAPM (single beta) can't capture stocks that move 1.6x in up markets but 3.8x in down markets. We adopted the asymmetric split:

```
expected = alpha + beta_up * rm_pos + beta_down * rm_neg
```

Using softplus-based smooth splitting to avoid discontinuity at rm=0. The centering fix (subtracting ln(2)/K) ensures expected(0) = alpha, not alpha + (beta_up - beta_down) * ln(2)/K.

Literature support: Market model is used in 79.1% of event studies (Holler 2014 meta-review). Our asymmetric version is strictly more expressive.

## Entry 3 — DEAD END: conditional volatility sigma(rm)

Tried σ(rm) = γ₀ + γ₁·|rm| as the denominator for the z-score. The idea was sound — bigger market days are noisier — but it **broke the matrix ordering**. At index +5%, σ was 3x larger than at index 0%, which *shrank* the z-score at extreme market moves. A -5% stock with +5% index had a LOWER |z| than -5% stock with 0% index, which is backwards.

Root cause: the denominator grows faster than the numerator in the tails.

Fix: use fixed unconditional σ (just γ₀) as denominator. The conditional volatility concept is valid for risk management but wrong for event detection where you want extreme days to score high, not get normalized away.

## Entry 4 — Scoring function: why sign(z)·z² not linear

Linear AR in bps treats a 5σ event as 5x a 1σ event. For news detection, a 5σ event is astronomically rarer and almost certainly news-driven. We need non-linear amplification.

Options considered:
- p=1 (linear): too flat, noise and signal overlap
- p=1.5 (Huber boundary): stable moments under cubic-law tails, 11:1 ratio at 5σ
- p=2 (squared): variance-weighted, chi-squared connection, 25:1 ratio at 5σ
- p=3 (cubed): unstable under cubic-law tails (infinite mean), never use

Chose p=2 because the goal is ranking individual days for news detection, not computing population statistics. Moment instability doesn't matter when sorting a list. The 25:1 amplification at 5σ vs 1σ clearly separates genuine events from noise.

Literature: Gopikrishnan et al. 1998 — stock return tails follow power law with α≈3. Gabaix 2009 review confirmed ζ_r = 3.1 ± 0.1 across 1000 largest US stocks. Student-t fits to residuals show ν≈3-5 (degrees of freedom), meaning residuals after beta removal have somewhat thinner tails than raw returns but still fat.

## Entry 5 — Added own-history dimension

Index-relative AR alone misses cases where: (a) market is flat, stock moves +3%, and that's actually normal for this volatile stock, or (b) a utility stock that normally moves ±40 bps suddenly moves +300 bps but beta explains it all.

Added z_own = (R_stock - rolling_median) / σ_own as a second dimension. This catches "the stock did something unusual for itself."

Combined formula: sign(z_index) · |z_index| · |z_own| — requires BOTH dimensions to fire for a high score. A high AR with normal own-history stays moderate. A normal AR with unusual own-move also stays moderate. Only days that are extreme on both reach the top.

## Entry 6 — Added volume as confidence multiplier

Volume tells you something price cannot. Kim and Verrecchia 1991: "price change reflects the average change in traders' beliefs, whereas volume reflects the sum of differences in traders' reactions."

Initially used max(|z_own|, |z_vol|) — picking whichever was larger. This was wrong because:
- High price move + low volume (block trade, not news) scored high via z_own
- Low price move + high volume (ambiguous news, disagreement) scored high via z_vol
- Both are less newsworthy than high price + high volume

Switched to volume as a confidence multiplier:
```
vol_mult = max(0.5, 1 + 0.5 * ln(V / V_avg))
combined = sign(z_i) * |z_i| * |z_o| * vol_mult
```

Properties: normal volume = 1.0x (no effect), half volume = 0.65x (dampened), 5x volume = 1.8x (amplified), 10x volume = 2.15x. Log scaling prevents options expiration days from overwhelming. Floor at 0.5 means low volume dampens but never zeroes the score.

Literature: Karpoff 1987 survey — volume positively related to magnitude of price change. Karpoff 1986 theory — volume increases even when investors interpret information identically.

## Entry 7 — K=50 smoothness parameter

K controls the sharpness of the softplus up/down market split. Not from a paper — it's a tuning parameter.
- K=30: smoother around zero
- K=50: balanced default (chosen)
- K=80-100: closer to hard max(rm,0)/min(rm,0)

With returns in decimals, 1/K ≈ 0.02, so the rounding concentrates around ±2%, meaning -0.01%, 0%, +0.01% behave almost identically. This was validated visually with the interactive matrix.

## Entry 8 — MAD chosen over standard deviation

MAD (median absolute deviation) with 1.4826 scaling factor is strictly better than standard deviation for fat-tailed stock returns. Standard deviation has breakdown point of 0 — a single extreme observation can make it arbitrarily large. MAD is robust.

Using rolling MAD on past residuals (for z_AR) and on raw returns (for z_own). Volume uses log(volume) before MAD because volume distributions are extremely right-skewed.

The rolling pipeline already had a double-warmup bug: RollingOLS needs 60 rows, then MAD needs 60 non-NaN values, producing NaN until row ~120. Fixed with min_periods = max(15, window//4) on the MAD, so sigma starts at row ~75 instead.

## Entry 9 — Recommended production pipeline layers

From deep research review, the strongest formulation has four layers:

1. **Expected move model**: rolling per-ticker asymmetric beta (already built)
2. **Residual surprise**: z_AR = AR / rolling_MAD_sigma(AR_history) — how unusual was the miss vs what this stock normally does after controlling for market
3. **Own stock magnitude**: z_own = (R_stock - rolling_median) / σ_own — how unusual was the move for this stock regardless of direction
4. **Volume confirmation**: vol_mult = max(0.5, 1 + 0.5 * ln(V/V_avg)) — trading participation confirms the signal

Keep all signals as separate columns. Only combine at ranking layer. The downstream news-matching model needs access to individual signals as features.

A fifth layer (directional contradiction: did the stock move opposite to what beta predicted?) was proposed but deferred — z_AR already captures contradiction implicitly when the sign flips, and adding it would be double-counting.

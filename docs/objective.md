# Objective and purpose

## Problem

Predicting stock prices based on historical data and corresponding market news. The task requires developing a predictive model from scratch that demonstrates robust performance on unseen data, synthesizing different data sources in stock price prediction.

## What we're building

A multi-stage pipeline:

**Stage 1 (current)**: News impact detection — for each trading day, quantify how much the stock's movement was driven by stock-specific news rather than market conditions or normal noise. This produces feature columns that score each day's "newsworthiness."

**Stage 2 (next)**: News matching — given the scored days, cross-reference with actual news articles to build a training set of (news → impact) pairs.

**Stage 3 (final)**: Predictive model — given new news, predict expected stock price impact. The model should combine news embeddings (likely FinBERT or similar) with the historical impact scores as training signal.

## Success criteria

- The scoring system should rank days by news impact in a way that matches human intuition (the semantic matrix — see journal.md entry 1)
- When we cross-reference top-scored days with actual news, the vast majority should correspond to identifiable events (earnings, FDA decisions, acquisitions, etc.)
- The final predictive model should outperform baselines that use only price history or only news sentiment

## Data available

Will be provided by the user. Expected format:
- Stock daily OHLCV (open, high, low, close, volume) for one or more tickers
- S&P 500 daily OHLCV as the index reference
- Previous close prices for gap return computation
- News articles with dates and headlines (for stage 2)

## Key constraints

- All model parameters (alpha, beta, sigma) must be estimated from historical data via rolling windows, not hand-picked
- The methodology should be grounded in established finance literature (event study methodology, abnormal return framework)
- Two return types per day: gap (prev close to open) and close-to-close (prev close to close) — see journal Entry 13
- Rolling estimation window of 120 days for the beta model (tested 30/60/120 — 120d produced lowest MAE for 6/7 tickers, see journal Entry 10)
- MAD-based volatility estimation (robust to fat tails, per the inverse cubic law — Gopikrishnan et al. 1998)

## What this is NOT

- Not a real-time trading system
- Not a portfolio optimizer
- Not trying to predict direction from price alone — the goal is to measure news impact as a feature for a downstream model

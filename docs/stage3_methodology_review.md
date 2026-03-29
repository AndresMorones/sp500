# Stage 3: Methodology Review — Stock Price Prediction Using News + Historical Data

## Purpose

Critical comparison of our methodology against validated academic research. Not confirmation bias — let the papers tell us what's right, wrong, or missing in our approach.

## Our Setup

- **7 mega-cap tech stocks**, each modeled independently (per-ticker)
- **~600 news articles per stock**, with headline + summary, ~1 year (Nov 2023 - Oct 2024)
- **Daily OHLCV** + S&P 500
- **Precomputed features**: abnormal returns (A metric), z-scores (zi, zo, zv), betas, volume
- **News processing**: LLM (Claude) extracts 14 structured dimensions + 8 business categories per article, bucketed into gap/cc periods
- **Goal**: predict stock price, using excess return as intermediate target
- **Model and evaluation methodology**: open — research should inform this

---

## PART 1: Closest-Match Papers (Same Problem, Similar Scale)

### Paper 1: Advanced LSTM with News Sentiment (arXiv 2505.05325, May 2025)

**THE closest match to our setup.**

**Setup**: 4 mega-cap tech stocks (Apple, Google, Microsoft, Amazon). Daily OHLCV from Yahoo Finance. 1 year of data (April 2024 - April 2025). Financial news articles processed with VADER sentiment.

**Prediction target**: Next-day closing price (continuous value, not direction).

**Architecture**:
```
60-day sliding window of [normalized close prices + VADER sentiment scores]
    |
    v
LSTM Layer 1 (64 units, 20% dropout)
    |
    v
LSTM Layer 2 (32 units)
    |
    v
Dense Output Layer (1 neuron)
    |
    v
Next-day closing price prediction
```

**How they process news**:
1. Collect financial news articles daily
2. VADER sentiment analysis produces compound polarity score (-1 to +1) per article
3. Daily aggregation: average sentiment across all articles for that stock-day
4. Single scalar sentiment feature concatenated with normalized price

**How they handle no-news days**: Not explicitly addressed. 60-day sliding window implicitly smooths over gaps.

**Evaluation**: 80/20 chronological train/test split. Metrics: MAPE, MAE, MSE, RMSE.

**Results per stock**:
| Stock | MAPE |
|-------|------|
| Apple | 2.72% |
| Google | 2.65% |
| Microsoft | 2.91% |
| Amazon | 3.05% |

Sentiment contribution: ~5% relative improvement over price-only LSTM.

**Limitations acknowledged**: Struggles with abrupt market shifts (geopolitical, regulatory). No macro indicators. Day-ahead only.

**Comparison to our approach**:

| Aspect | Their Approach | Our Approach | Assessment |
|--------|---------------|-------------|------------|
| News processing | VADER → 1 sentiment score | LLM → 14 dimensions + 8 categories | Ours is far richer. VADER collapses all information into one number. |
| Model | LSTM (2 layers) | TBD | They validated LSTM for this exact scale. |
| Data scale | 4 stocks, 1 year | 7 stocks, 1 year | Nearly identical. |
| Target | Raw closing price | Excess return (proposed) | Different — they predict price directly. |
| Per-ticker? | Yes, separate model per stock | Yes | Same approach. Validated. |
| Features | Price + 1 sentiment | Price + 14 dimensions + categories + A scores | Ours has much more signal (if dimensions are informative). |

**Key takeaway**: This paper validates that per-ticker LSTM with news on 1 year of data for mega-cap tech stocks WORKS. Our richer news features should outperform their single VADER score — but we need to confirm this, not assume it.

---

### Paper 2: Tesla Stock Prediction with News Sentiment (ICIAAI 2025)

**Single-stock deep dive with news category analysis.**

**Setup**: TSLA only. 5 different LSTM model variants tested with different news inputs.

**Architecture**: LSTM with different combinations of news sentiment inputs.

**Key finding**: **Financial news sentiment helped prediction. Tech news sentiment sometimes HURT prediction.** Not all news categories are equally useful — some add noise.

**Result**: Best model MSE = 187.72

**Comparison to our approach**:

This directly validates our Phase 1 design — discovering per-ticker categories and filtering which dimensions matter. The Tesla paper found that generic "tech news" hurt the model. Our approach of having the LLM identify 8 business-specific categories per ticker (e.g., for TSLA: EV competition, autonomous driving, regulatory, Elon Musk leadership, etc.) should avoid mixing helpful and harmful news signals.

**Key takeaway**: News category matters per stock. Not all news helps. Our per-ticker category discovery is methodologically sound.

---

### Paper 3: LLM-Generated Formulaic Alpha (arXiv 2508.04975, August 2025)

**Most methodologically similar to our LLM-based approach.**

**Setup**: 5 stocks (Apple, HSBC, Pepsi, Toyota, Tencent). May 2016 - May 2024 (8 years). Per-stock models.

**Prediction target**: Next-day closing price.

**Architecture**:
```
Daily OHLCV + 12 technical indicators + per-stock VADER sentiment + related-company sentiment
    |
    v
LLM (GPT-4) generates 5 formulaic alpha factors per stock
    Example: alpha_1 = (Momentum_3 + Momentum_10)/2 + (AAPL_sent + MSFT_sent)/3
    |
    v
Alphas computed as daily features
    |
    v
Downstream model: Transformer / LSTM / TCN / SVR / Ridge / Random Forest / XGBoost
    |
    v
Next-day closing price
```

**How they use the LLM**: NOT for embeddings. NOT for direct prediction. The LLM generates mathematical formulas that combine existing features. These formulas are computed as new features and fed to traditional models.

**Results**: 10-26% MSE improvement when LLM alphas are added. Transformer and LSTM benefit most. Ridge benefits least.

**Evaluation**: 70/30 chronological split. Diebold-Mariano test for statistical significance. Holm-Bonferroni correction for multiple comparisons.

**Comparison to our approach**:

| Aspect | Their Approach | Our Approach | Assessment |
|--------|---------------|-------------|------------|
| LLM role | Generate formulaic alpha factors | Extract structured dimensions from articles | Different use of LLM. They use it for feature engineering on existing features. We use it for news interpretation. |
| News processing | VADER sentiment → scalar | LLM → 14 dimensions + categories | They still use simple sentiment. We go much deeper. |
| Downstream model | Tested 7 models | TBD | They found Transformer/LSTM best, Ridge worst for LLM features. |
| Evaluation | Diebold-Mariano + Bonferroni | TBD | Their evaluation is rigorous. We should adopt similar. |

**Key takeaway**: LLM-generated features improve downstream models by 10-26%. Non-linear models (Transformer, LSTM) benefit more than linear ones (Ridge). This suggests our downstream model should probably NOT be Ridge alone.

---

### Paper 4: FinBERT-LSTM (arXiv 2407.16150, 2024)

**Hierarchical news categorization — matches our category approach.**

**Setup**: NASDAQ-100 stocks. Benzinga news (843,062 articles). Per-stock models.

**Prediction target**: Next-day closing price.

**Architecture**:
```
News articles → FinBERT → sentiment polarity (pos/neg/neu probabilities)
    |
    v
Three hierarchy levels:
    1. Market-level news sentiment
    2. Industry-specific news sentiment
    3. Stock-specific news sentiment
    |
    v
Weighted combination (learned weights for each level)
    |
    v
Concatenated with 8 days of normalized closing prices
    |
    v
LSTM (3 layers, 50 neurons each) → Dense → Price prediction
```

**Key innovation**: They weight news at three levels — market, sector, stock-specific. Not all news matters equally. Learned weights determine contribution of each level.

**Results**: MAE 173.67, MAPE 4.5%. Outperforms standalone LSTM and DNN.

**Comparison to our approach**:

| Aspect | Their Approach | Our Approach | Assessment |
|--------|---------------|-------------|------------|
| News hierarchy | 3 fixed levels (market/sector/stock) | 8 discovered categories per ticker | Ours is more granular and stock-specific. |
| Weighting | Learned weights per level | TBD — could use model to learn importance | We should let the downstream model weight dimensions. |
| Sentiment model | FinBERT (3 probs) | Claude (14 dimensions) | Ours is much richer. |
| Time window | 8-day lookback | TBD | Suggests short lookback is sufficient. |

**Key takeaway**: Hierarchical news categorization with learned weights improves prediction. Our 8 per-ticker categories are a richer version of their 3-level hierarchy. The downstream model should learn which categories matter.

---

### Paper 5: Impact of LLMs on Stock Price Movement Prediction (arXiv 2602.00086, February 2026)

**Most comprehensive model comparison on mega-cap tech stocks.**

**Setup**: 5 stocks (Microsoft, Amazon, Apple, Netflix, Tesla). March 2022 - April 2025 (3 years). 96,000+ news articles via AlphaVantage.

**Prediction target**: Stock price movement (direction + regression targets).

**They tested 3 sentiment models**:
- DeBERTa: 75% accuracy (BEST individual)
- FinBERT: 70.9% accuracy
- RoBERTa: 58.9% accuracy
- Ensemble (SVM combining all 3): ~80% accuracy

**They tested 4 time-series architectures**:
1. LSTM
2. PatchTST (patch-based transformer)
3. TimesNet (temporal variation modeling)
4. tPatchGNN (temporal graph neural networks)

**Key finding**: DeBERTa outperforms domain-specific FinBERT. TimesNet and PatchTST benefited most from sentiment features.

**Evaluation**: 70/10/20 train/validation/test. 30-day rolling windows.

**Comparison to our approach**:

They use sentiment scores (positive/negative/neutral). We extract 14 structured dimensions. Their finding that different architectures benefit differently from sentiment suggests we should test multiple downstream models, not commit to one upfront.

**Key takeaway**: Test multiple downstream models. Sentiment improvement is "modest but consistent." Architecture choice matters as much as feature quality.

---

## PART 2: Papers on Structured News Feature Extraction

### Paper 6: Structured Event Representation for Stock Return Predictability (arXiv 2512.19484, 2025)

**Most methodologically aligned with our structured extraction approach.**

**Method**: Uses GPT APIs to extract structured event triplets from news: (subject, action, object, context). Transforms unstructured news into interpretable structured representations.

**Results**: Daily predictions: 10.93% annualized return (Sharpe 0.78). **Outperforms both sentiment-based and embedding-based benchmarks.**

**Key takeaway**: Structured event representation > embeddings > sentiment. This validates our approach of extracting structured information rather than using embeddings or simple sentiment.

---

### Paper 7: FININ — News Interaction Networks (EMNLP 2024 Findings, arXiv 2410.10614)

**Addresses a gap in most approaches: news-news interactions.**

**Method**: Models not just news → price, but also news → news interactions. A second article about the same event doesn't double the impact — it confirms it. An article contradicting the first changes the signal entirely.

**Architecture**: Two components:
1. Data Fusion Encoder (encodes multi-modal information)
2. Market-Aware Influence Quantifier (models news-news and news-price relationships)

**Results**: S&P 500: +0.429 daily Sharpe improvement. NASDAQ 100: +0.341.

**Data**: 2.7 million articles, 15 years.

**Comparison to our approach**: Our Phase 2 (day-level scoring) partially addresses this — the prompt instructs Claude to "identify duplicates and treat them as one" and report "distinct_events." But we don't model how articles interact (confirm, contradict, amplify). For 7 stocks with ~600 articles each, this may be overkill, but the insight is valuable: **deduplication and interaction effects matter**.

---

### Paper 8: Weighted and Categorized News for LSTM (PMC 9990937)

**Validates learned weights across news categories.**

**Method**: Three news categories (market, sector, stock-specific) with learned weights (alpha, beta, gamma) optimized via grid search. LSTM with concatenation.

**Key finding**: Learned category weights significantly outperform equal-weight averaging. Different stocks have different optimal weight distributions.

**Data**: 41 stocks, 6 sectors, 12 years. 2,616 market headlines + 5,523 sector headlines.

**Comparison to our approach**: Our 8 per-ticker categories with per-dimension scores are richer. But we haven't defined how to WEIGHT them. This paper suggests the model should learn weights, not us hardcoding them. With categories as features in a tree model (LightGBM), the model naturally learns which categories matter — each split is effectively a learned weight.

---

## PART 3: Reality Checks

### Paper 9: FINSABER — Do LLM Strategies Actually Work Long-Term? (KDD 2026, arXiv 2505.07078)

**20-year backtest of LLM trading strategies.**

**Finding**: "LLM advantages deteriorate under broader evaluation."
- Conservative in bull markets (underperform buy-and-hold)
- Aggressive in bear markets (take losses)
- Survivorship bias and data snooping inflate results in most papers

**Implication**: Our 1-year test period is short. Results must be interpreted cautiously. Pre-committed success criteria and Bonferroni correction are essential.

---

### Paper 10: Lopez-Lira & Tang — Alpha Decay (arXiv 2502.10008, Feb 2025)

**LLM-based trading Sharpe ratios declining over time**:
- 2021 Q4: Sharpe 6.54
- 2022: Sharpe 3.68
- 2023: Sharpe 2.33
- 2024: Sharpe 1.22

**Implication**: LLM-based signals lose edge as more participants adopt them. Our 2023-2024 data window may already reflect diminished LLM alpha. This isn't a reason not to try — it's a reason to set realistic expectations.

---

## PART 4: Synthesis — What the Research Says About Our Approach

### What we're doing RIGHT (validated by literature):

1. **Per-ticker models** — Standard for small stock universes. All closest-match papers (Papers 1-4) use per-stock models. Validated.

2. **Per-ticker news categories** — Paper 2 (Tesla) found some news categories hurt prediction. Paper 4 (FinBERT-LSTM) found hierarchical categorization with learned weights helps. Paper 8 confirms learned category weights matter. Our Phase 1 category discovery is well-grounded.

3. **LLM for structured extraction, not direct prediction** — Paper 6 shows structured event representations outperform embeddings and sentiment. Paper 3 shows LLM-generated features improve downstream models by 10-26%. Our approach of having Claude extract 14 dimensions is the richest version of this.

4. **Multi-dimensional scoring** — No paper extracts as many dimensions as we do (14). Most use 1 (sentiment score) or 3 (pos/neg/neu probabilities). If our dimensions are informative, this is an advantage. If they're noisy, it's a liability. The downstream model must handle this.

5. **Gap/CC bucketing** — No paper we found separates pre-market from intraday news timing. This is unique to our approach and methodologically sound (different news windows affect different return periods).

### What the research CHALLENGES about our approach:

1. **Downstream model choice is critical.** Paper 3 found Transformer/LSTM benefit 10-26% from LLM features, but Ridge benefits modestly. Paper 5 found TimesNet/PatchTST benefit most. If we only use Ridge/LightGBM, we may underexploit our rich features. **The literature suggests testing LSTM as a downstream model, not just Ridge/LightGBM.**

2. **Excess return vs raw price as target.** Every closest-match paper (Papers 1-5) predicts **raw closing price or raw return**, not excess return. Our A-metric framework (excess return = stock - beta*market) is grounded in event study literature, but it's NOT standard in the news+ML prediction literature. This is a methodological choice we should explicitly justify or test both.

3. **14 dimensions may be too many for ~600 articles.** No paper uses more than 3 news features per stock. With ~600 articles and 14 dimensions each, we risk overfitting — the model may latch onto noise in rarely-occurring dimension combinations. **Consider starting with fewer dimensions (severity, direction, surprise) and adding incrementally.**

4. **Evaluation methodology.** Paper 3 uses Diebold-Mariano test + Holm-Bonferroni correction. Paper 1 uses simple train/test split. Our Clark-West test is from our baseline work but may not be the standard in this specific literature. We should justify our evaluation choice.

5. **60-day lookback window.** Paper 1 uses 60 days. Paper 4 uses 8 days. If using LSTM, the lookback window is a hyperparameter. With our structured features, the "memory" is encoded in the dimensions (temporal_horizon captures persistence), so a shorter lookback may suffice.

6. **No paper validates our specific approach** — LLM extracting 14 dimensions per article for per-ticker LSTM/ML prediction on 1 year of mega-cap tech stocks. We are in novel territory. This is both the risk and the opportunity.

### Open questions the research raises:

1. **Should we predict raw price or excess return?** Literature does raw price. Our framework does excess return. Test both?

2. **LSTM or Ridge/LightGBM as downstream model?** Literature suggests LSTM for temporal features with news. Our features are already temporal (they're per-day scores). Does LSTM add value on top of that, or is the temporal structure already captured in the features?

3. **How many of the 14 dimensions actually help?** Need ablation: start with 3 (severity, direction, surprise), add more, measure marginal improvement.

4. **What's the right lookback window?** With structured features that encode temporal information (temporal_horizon, expected_duration), maybe the model needs less lookback than raw-price LSTM models.

5. **Should we weight news categories?** Paper 8 shows learned weights help. LightGBM does this naturally via splits. LSTM would need explicit category features.

---

## PART 5: Pseudocode from Most Relevant Papers

### Paper 1: Advanced LSTM (closest match to our scale)

```
FUNCTION predict_stock_price(ticker, price_data, news_data):

    # --- Step 1: News processing ---
    FOR EACH trading_day in price_data:
        articles = news_data[ticker][trading_day]
        IF articles:
            sentiment = mean([VADER(article.text) for article in articles])
        ELSE:
            sentiment = 0  # or carry forward previous day

    # --- Step 2: Feature construction ---
    features = []
    FOR EACH day in price_data:
        features.append([
            normalize(day.close),  # Min-Max scaled to [0,1]
            sentiment[day]
        ])

    # --- Step 3: Sliding window ---
    WINDOW_SIZE = 60  # days
    X, y = [], []
    FOR i in range(WINDOW_SIZE, len(features)):
        X.append(features[i-WINDOW_SIZE : i])   # shape: (60, 2)
        y.append(price_data[i].close)            # next-day close

    # --- Step 4: Train/test split ---
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- Step 5: Model ---
    model = Sequential([
        LSTM(64, return_sequences=True, dropout=0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_split=0.1, early_stopping=patience(10))

    # --- Step 6: Evaluate ---
    predictions = model.predict(X_test)
    predictions = inverse_normalize(predictions)
    mape = mean(abs(predictions - y_test) / y_test) * 100
    RETURN predictions, mape
```

### Paper 3: LLM-Generated Features + Downstream Models

```
FUNCTION predict_with_llm_features(ticker, price_data, news_data):

    # --- Step 1: Compute base features ---
    features = compute_technical_indicators(price_data)
    # features: Close, Open, High, Low, Volume, SMA_20, EMA_12, RSI_14,
    #           MACD, Bollinger_upper/lower, OBV, Momentum_3, Momentum_10

    # --- Step 2: Compute sentiment ---
    daily_sentiment = {}
    FOR EACH day in trading_days:
        articles = news_data[ticker][day]
        IF articles:
            daily_sentiment[day] = mean([VADER(a.text).compound for a in articles])
        ELSE:
            daily_sentiment[day] = 0

    # Also compute related-company sentiment
    related_tickers = find_related_companies(ticker)  # via NER co-occurrence
    FOR EACH related in related_tickers[:2]:
        features[f"{related}_sentiment"] = daily_sentiment_for(related)

    features["sentiment"] = daily_sentiment

    # --- Step 3: LLM generates alpha factors ---
    prompt = f"""
    Given these features for {ticker}:
    {list(features.columns)}

    Generate 5 formulaic alpha factors combining these features.
    Each alpha should be a mathematical formula.
    Output as JSON with name, formula, reasoning.
    """
    alphas = LLM(prompt)  # Returns 5 formulas

    # --- Step 4: Compute alpha values ---
    FOR EACH alpha in alphas:
        features[alpha.name] = evaluate_formula(alpha.formula, features)

    # --- Step 5: Train per-stock model ---
    X = features[all_columns]
    y = price_data.close.shift(-1)  # next-day close

    # 70/30 chronological split
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Test multiple models
    models = {
        "Transformer": VanillaTransformer(),
        "LSTM": LSTM_model(),
        "LightGBM": LGBMRegressor(),
        "Ridge": Ridge(),
        "XGBoost": XGBRegressor(),
    }

    results = {}
    FOR name, model in models:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mse": mse(y_test, pred),
            "dm_pvalue": diebold_mariano_test(baseline_errors, model_errors)
        }

    RETURN results
```

### Paper 4: Hierarchical News with Learned Weights

```
FUNCTION predict_with_categorized_news(ticker, price_data, news_data):

    # --- Step 1: Categorize news ---
    FOR EACH article in news_data[ticker]:
        IF is_market_level(article):
            market_sentiment.append(FinBERT(article.text))
        ELIF is_sector_level(article):
            sector_sentiment.append(FinBERT(article.text))
        ELSE:
            stock_sentiment.append(FinBERT(article.text))

    # --- Step 2: Aggregate per day, per level ---
    FOR EACH day:
        market_score[day] = mean(market_sentiment[day]) or 0
        sector_score[day] = mean(sector_sentiment[day]) or 0
        stock_score[day] = mean(stock_sentiment[day]) or 0

    # --- Step 3: Learn category weights ---
    # Grid search over alpha, beta, gamma where alpha+beta+gamma <= 1
    best_weights = grid_search(
        alpha_range=[0, 0.1, 0.2, ..., 1.0],
        beta_range=[0, 0.1, 0.2, ..., 1.0],
        gamma_range=[0, 0.1, 0.2, ..., 1.0],
        constraint=alpha+beta+gamma <= 1,
        objective=minimize_validation_mse
    )

    # --- Step 4: Combined weighted sentiment ---
    FOR EACH day:
        combined[day] = (best_weights.alpha * market_score[day]
                       + best_weights.beta * sector_score[day]
                       + best_weights.gamma * stock_score[day])

    # --- Step 5: LSTM with weighted news ---
    LOOKBACK = 8  # days (from paper)
    X = sliding_windows(
        [normalized_close, combined_sentiment],
        window=LOOKBACK
    )

    model = Sequential([
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])

    model.fit(X_train, y_train)
```

### Adapted Pseudocode: Our Approach (informed by all papers)

```
FUNCTION our_pipeline(ticker):
    """
    Combines: Paper 1's per-ticker LSTM scale,
    Paper 3's LLM feature approach,
    Paper 4's category weighting,
    Paper 6's structured extraction superiority.
    """

    # --- Step 1: Already done — LLM structured extraction ---
    # news_scorer.py Phase 1: 8 categories per ticker
    # news_scorer.py Phase 2: 14 dimensions per (ticker, date, period)
    # Output: news_day_features.csv with columns:
    #   date, ticker, period, direction, distinct_events,
    #   cat_<category_id> (0-10 relevance per category),
    #   materiality, surprise, temporal_horizon, sentiment_strength,
    #   information_density, directional_clarity, scope,
    #   competitive_impact, regulatory_risk, management_signal,
    #   narrative_shift, repeatedness, actionability, capital_commitment_scale

    structured_features = load("news_day_features.csv")

    # --- Step 2: Combine with price features ---
    price_features = load("scores_output.csv")
    # Has: A_gap, A_cc, zi, zo, zv, beta, volume, returns

    merged = merge(price_features, structured_features,
                   on=["ticker", "date"])

    # Add has_news indicator
    merged["has_news"] = (merged["materiality"].notna()).astype(int)

    # Fill no-news days: 0 for category relevance, neutral for dimensions
    merged[category_cols].fillna(0)
    merged[dimension_cols].fillna(5)  # middle of 1-10 scale
    merged["direction"].fillna("neutral")

    # --- Step 3: Feature matrix per ticker ---
    ticker_data = merged[merged.ticker == ticker].sort_values("date")

    feature_cols = (
        price_feature_cols     # from scores_output.csv
        + dimension_cols       # 14 LLM dimensions
        + category_cols        # 8 category relevance scores
        + ["has_news", "distinct_events", "direction_encoded"]
    )

    # --- Step 4: Target ---
    # OPTION A: Raw closing price (standard in news+ML literature)
    # OPTION B: Excess return (standard in event study literature)
    # Test both — Paper 1 uses raw price, our framework uses excess return

    # --- Step 5: Downstream model ---
    # Literature says: test multiple, don't commit to one
    # Paper 3: Transformer/LSTM best for LLM features (10-26% MSE improvement)
    # Paper 5: TimesNet/PatchTST benefit most from sentiment
    # Paper 3: Ridge benefits least from LLM features

    # Start simple, escalate:
    # Round 1: LightGBM (tree-based, handles mixed features naturally)
    # Round 2: LSTM (if temporal patterns in news features matter)
    # Round 3: Compare, pick winner

    # --- Step 6: Evaluation ---
    # Paper 1: 80/20 chronological split
    # Paper 3: 70/30 + Diebold-Mariano + Holm-Bonferroni
    # Paper 5: 70/10/20 + 30-day rolling windows

    # Our evaluation: walk-forward expanding window (most rigorous)
    # Metrics: MAPE, MAE, RMSE, R²_OOS
    # Statistical test: Diebold-Mariano (standard in this literature)

    # --- Step 7: Ablation ---
    # Paper 2 found some news categories hurt prediction
    # Test: all 14 dims vs top 3 (severity, direction, surprise)
    # Test: with vs without category relevance scores
    # Test: with vs without price features (isolate news contribution)
```

---

## PART 6: Key References

### Closest matches (MUST READ):
1. **Advanced LSTM** — arXiv:2505.05325 (May 2025) — 4 tech stocks, 1 year, VADER+LSTM
2. **Tesla LSTM** — ICIAAI 2025 — single stock, news categories help/hurt
3. **LLM-Generated Alpha** — arXiv:2508.04975 (Aug 2025) — 5 stocks, LLM features + 7 models
4. **FinBERT-LSTM** — arXiv:2407.16150 (2024) — hierarchical news + learned weights
5. **LLM Sentiment Impact** — arXiv:2602.00086 (Feb 2026) — 5 tech stocks, 96K articles, 4 architectures

### Structured extraction:
6. **Structured Event Representation** — arXiv:2512.19484 (2025) — structured > embeddings > sentiment
7. **FININ** — EMNLP 2024 — news-news interaction effects

### Reality checks:
8. **FINSABER** — arXiv:2505.07078 (KDD 2026) — LLM strategies deteriorate over 20 years
9. **Lopez-Lira Alpha Decay** — arXiv:2502.10008 (Feb 2025) — Sharpe declining 6.5→1.2

### Per-ticker validation:
10. **PSO-LSTM** — Springer 2026 — per-ticker hyperparameter optimization outperforms generic
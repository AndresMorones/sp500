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

## PART 9: Novelty Analysis — Does Any Paper Do What We Do? (2026-03-30)

### The question

Our Phase 1 methodology asks the LLM to (1) discover company-specific news categories from the articles themselves, then (2) rate each article on 12-15 structured numeric dimensions. Is there any precedent for this in the literature?

### Answer: No paper combines both. Our approach is genuinely novel.

After exhaustive search across arXiv, Semantic Scholar, and conference proceedings (2023-2026), no paper combines LLM-driven category discovery with multi-dimensional numeric rating for downstream stock prediction. Two papers come closest to individual components:

### 9.1 LLMFactor (ACL Findings 2024, arXiv:2406.10811) — Closest to Category Discovery

**What it does**: Prompts the LLM to "extract the top k factors that may affect the stock price of [stock]." Factors are **NOT pre-defined** — the LLM discovers them from news content. Examples discovered: "Selection of Nvidia Drive Thor by EV makers," "$250M investment in supplier Corning" (for Apple).

**Key difference from our approach**: Discovers free-text factors **per individual article**, not a reusable taxonomy across all articles. Factors are not rated on numeric scales — they're text descriptions. Factors feed back into the LLM itself for prediction, not into a separate quantitative model.

| Aspect | LLMFactor | Our Phase 1 |
|--------|-----------|-------------|
| Categories | Discovered per-article | Discovered per-company (reusable across all articles) |
| Format | Free-text factor descriptions | Structured 9-category taxonomy with snake_case IDs |
| Numeric rating | None | 12-15 dimensions, each 0-10 |
| Downstream model | Same LLM (GPT-4) | Separate quant model (LSTM/LightGBM) |
| Validation | Outperforms SOTA on 4 benchmarks | TBD |

**What we take from this**: LLMFactor validates that letting the LLM discover what matters per stock (rather than pre-defining categories) produces better signal. Our approach goes further by building a reusable taxonomy and adding numeric dimensions.

### 9.2 Event-Aware Sentiment Factors (ICML 2025, arXiv:2508.07408) — Closest to Multi-Dimensional Rating

**What it does**: Assigns multi-label event tags from a curated dictionary of **70+ financially relevant event types** (Rumor/Speculation, Retail Investor Buzz, Geopolitical Tension, Brand Boycott, etc.) plus a continuous net tone score. Constructs cross-sectional factors for long-short portfolios. Found "Rumor/Speculation" is a powerful contrarian signal (Sharpe -0.38).

**Key difference from our approach**: Categories are **pre-defined** (70+ event types in a curated dictionary), not discovered from the articles. Only 2 output dimensions (multi-label tags + tone), not 12-15 numeric scores.

| Aspect | Event-Aware | Our Phase 1+2 |
|--------|-------------|---------------|
| Categories | 70+ pre-defined event types | 9 discovered per company |
| Discovery | None (curated dictionary) | LLM reads all articles, induces taxonomy |
| Dimensions per article | 2 (event labels + tone) | 12-15 (numeric 0-10 scales) |
| Scale | Cross-sectional (many stocks) | Per-ticker |
| Data source | Tweets | News articles |

**What we take from this**: Rich event categorization (beyond pos/neg/neu) produces meaningful financial signals. Their 70+ categories show appetite for granularity in the field. But their categories are static — ours adapt to each company.

### 9.3 Exhaustive Near-Miss Analysis — Papers That Almost Match (updated 2026-03-30)

A second, targeted search (25+ queries across arXiv, SSRN, NeurIPS, EMNLP, ICAIF, GitHub, Chinese-language sources) specifically for LLM + stock-specific categories + **price regression** (not direction) found these additional near-misses. Each fails on at least one of the three required criteria: (1) LLM-discovered categories or multi-dimensional scoring, (2) downstream regression model, (3) stock price as target.

#### Near-misses for Approach A (LLM discovers categories + price regression)

| Paper | What LLM does | Downstream | Target | Why it doesn't qualify |
|-------|--------------|------------|--------|----------------------|
| **"Structuring News, Shaping Alpha"** (NeurIPS 2025 GenAI Workshop) | Creates event classes via RL (PPO), refined by price impact | XGBoost | Quantile probability | Target is **quantile classification**, not regression |
| **"StockMem"** (arXiv:2512.02720, Dec 2025) | Discovers 57 event types in 13 groups via iterative induction (DeepSeek-V3) | Memory-augmented predictor | Ternary (up/down/flat) | Target is **classification**, not regression |
| **LLMFactor** (ACL 2024, arXiv:2406.10811) | Discovers free-text factors per article | Same LLM (GPT-4) | Binary direction | Target is **classification**; no separate quant model |

#### Near-misses for Approach B (Multi-dimensional scoring + price regression)

| Paper | Dimensions extracted | Downstream | Target | Why it doesn't qualify |
|-------|---------------------|------------|--------|----------------------|
| **"Not All News Is Equal"** (arXiv:2603.09085, Mar 2026) | 3-class sentiment + 12 topic categories + event type | LSTM, TFT (R², RMSE, MAE) | **Aluminum price** | Not stocks; categories **pre-defined**; only sentiment polarity, no severity/surprise/materiality |
| **Event-Aware Sentiment** (arXiv:2508.07408, ICML 2025) | 70+ pre-defined event types + tone | Portfolio sorting | Continuous log returns | Categories **pre-defined**; portfolio sorting, not regression model |
| **Structured Event Repr.** (arXiv:2512.19484, 2025) | Event triplets (subject, action, object, context) | TransE → MLP, **trained with MSE** | Stock returns (genuine regression) | Extracts **triplets**, not categories or multi-dimensional scores |
| **Emotion Analysis of Headlines** (PETRA 2024) | 6 Ekman emotions + neutral (7 dimensions) | Classifier | Trend direction | Target is **classification** |

#### Other related work (further from our approach)

| Paper | What it does | Gap vs our approach |
|-------|-------------|-------------------|
| **Extracting Structured Insights** (ICAIF 2024, arXiv:2407.15788) | Polygon.io: structured JSON with ticker discovery + sentiment + keywords | Only pos/neg/neu sentiment, no rich dimensions |
| **FinDKG** (ICAIF 2024, arXiv:2407.10909) | Knowledge graph triplets: 12 entity types + 15 relation types from 400K articles | All pre-defined, no per-company discovery |
| **SEP** (WWW 2024, arXiv:2402.03659) | Self-reflective LLM discovers factors implicitly through error correction | No structured taxonomy, factors are embedded in text |
| **Narratives from GPT Networks** (IJDSA 2024, arXiv:2311.14419) | Entity extraction → co-occurrence graphs → topic clusters emerge | Categories emerge from graph analysis, not LLM-directed discovery |
| **StockEmotions** (AAAI 2023, arXiv:2301.09279) | 12 fine-grained emotion classes for finance (anxiety, panic, excitement) | Pre-defined, pre-LLM era (DistilBERT), emotions not event categories |
| **Lopez-Lira & Tang** (arXiv:2304.07619, JF 2025) | Single sentiment score (-1/0/1) from LLM | Return regression | Only 1 dimension (sentiment) |
| **Fine-Tuning LLMs for Stock Returns** (arXiv:2407.18103, EMNLP 2024) | Opaque LLM embeddings | Return regression | Embeddings, not structured features |

### 9.4 Confirmation of Novelty (verified 2026-03-30)

**After two rounds of exhaustive search (35+ queries, arXiv, SSRN, NeurIPS, EMNLP, ICAIF, KDD, GitHub, Chinese-language sources), no published paper simultaneously:**

1. Uses an LLM to **discover company-specific news categories** or **score articles on multiple rich dimensions** (severity, surprise, materiality, etc.)
2. Feeds those structured features into a **separate downstream regression model**
3. Predicts **continuous stock price or return** (MAPE, MAE, R²)

The field is converging from three directions — but nobody has connected all three:
- **Event category discovery** (StockMem, Structuring News) → classification only
- **Structured event extraction** (arXiv:2512.19484) → regression, but triplets not multi-dimensional scores
- **Multi-dimensional commodity analysis** (Not All News Is Equal) → regression, but pre-defined categories and non-stock target

### 9.5 What Is Novel in Our Approach

Three elements are individually present in the literature but have **never been combined**:

1. **Dynamic company-specific taxonomy discovery** — LLMFactor discovers per-article factors; we discover per-company taxonomies. No paper builds a reusable, stock-specific category system from the corpus.

2. **Rich multi-dimensional numeric rating** (12-15 dimensions, each 0-10) — The richest in the literature. Event-Aware uses 2 dimensions. LLMFactor uses 0 (text only). Most papers use 1-3 (sentiment).

3. **Two-phase architecture** (discover taxonomy → rate within it) — No precedent. All existing approaches either use pre-defined categories or extract per-article factors in a single pass.

4. **LLM-structured features fed to a separate quantitative model** — LLMFactor feeds factors back to the LLM. We extract structured numeric features for LSTM/LightGBM. Paper 3 (arXiv:2508.04975) does something similar with formulaic alphas, but those are mathematical formulas on existing features, not news interpretations.

### 9.6 Risk Assessment

Novelty cuts both ways:

| Advantage | Risk |
|-----------|------|
| Richer signal than any existing approach | No external validation of the full pipeline |
| Company-specific categories should reduce noise (Paper 2 showed generic categories hurt) | 12-15 dimensions on ~600 articles = overfitting risk (no paper uses >3) |
| Reusable taxonomy enables consistent scoring | Taxonomy quality depends on LLM prompt engineering — not reproducible across models |
| Separate quant model allows rigorous evaluation | Two-phase architecture means errors compound (bad categories → bad ratings) |

**Mitigation**: The ablation plan in Q15 (start with 3 dimensions, add incrementally) directly addresses the overfitting risk. Per-ticker feature selection addresses the noise risk.

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

### LLM-based category/factor discovery:
11. **LLMFactor** — ACL Findings 2024, arXiv:2406.10811 — LLM discovers stock-specific factors per article (not pre-defined). Closest to our category discovery approach. See Part 9.
12. **Event-Aware Sentiment Factors** — ICML 2025, arXiv:2508.07408 — 70+ pre-defined event types + continuous tone. Closest to multi-dimensional rating. See Part 9.
13. **Extracting Structured Insights** — ICAIF 2024, arXiv:2407.15788 — Polygon.io; structured JSON extraction (tickers, sentiment, keywords)
14. **FinDKG** — ICAIF 2024, arXiv:2407.10909 — LLM extracts knowledge graph triplets with 12 entity types + 15 relation types (all pre-defined)
15. **SEP** — WWW 2024, arXiv:2402.03659 — Self-reflective LLM implicitly discovers important factors, no structured taxonomy
16. **MarketSenseAI 2.0** — arXiv:2502.00415 (2025) — 5 LLM agents, news summarized into progressive narratives (not categorized). 125.9% vs 73.5% index on S&P 100.

---

## PART 7: Empirical Baseline — FinBERT-LSTM on Our Data (2026-03-29)

### What we ran

Adaptation of arXiv:2211.07392 (not Paper 4 — a different, simpler FinBERT-LSTM paper) on our exact dataset. Three models compared: MLP (price only), LSTM (price only), FinBERT-LSTM (price + FinBERT mean sentiment per ticker-day). All code in `llm_baseline_model_modified/FinBERT-LSTM/`. Results in `data/output/finbert_lstm_results/`.

**Setup**:
- 7 tickers, 241 trading days (Nov 2023 – Oct 2024)
- 4209 news articles after date-alignment (down from 4440 — 231 fell on non-trading days)
- 1242 (ticker, date) sentiment pairs generated by FinBERT
- 85/15 chronological train/test split
- 10-day price lookback + 1 sentiment value = 11-step input
- Scaler fitted on train only (original paper had data leakage — fixed)
- Target: raw Close price (not excess return)

### Results — MAPE by ticker

| Ticker | MLP | LSTM | FinBERT-LSTM | Best model |
|--------|-----|------|--------------|-----------|
| AAPL | 2.01% | 1.71% | **1.42%** | FinBERT-LSTM |
| AMZN | 1.74% | **1.05%** | 1.18% | LSTM |
| GOOGL | 1.12% | 0.94% | **0.74%** | FinBERT-LSTM |
| META | 10.94% | **3.68%** | 4.80% | LSTM |
| MSFT | **0.60%** | 0.61% | 0.92% | MLP |
| NVDA | 6.22% | 3.34% | **2.84%** | FinBERT-LSTM |
| TSLA | **3.00%** | 4.89% | 3.07% | MLP |
| **Aggregate** | 3.66% | 2.32% | **2.14%** | FinBERT-LSTM |

### Key empirical findings

**1. FinBERT-LSTM wins on aggregate but loses on 3/7 tickers.**
AMZN, MSFT, and TSLA: adding sentiment hurt vs LSTM alone. This directly confirms Paper 2's finding (Tesla, ICIAAI 2025): not all news is signal. For MSFT, which has smooth, trend-following price behavior, the FinBERT score adds noise. For AMZN, news from our dataset (941 articles — highest volume) may be too mixed to produce a clean scalar signal from simple mean aggregation.

**2. META outlier reveals model failure mode.**
MLP MAPE of 10.94% (vs LSTM 3.68%) shows that a flat feature vector model completely fails during META's high-volatility earnings periods. The LSTM's temporal memory absorbs the shock. This supports Paper 5's recommendation to use temporal architectures (LSTM, TimesNet) rather than feedforward models when stocks have episodic volatility.

**3. Aggregate MAPE of 2.14% matches Paper 1's results in context.**
Paper 1 (arXiv:2505.05325) reported 2.65–3.05% MAPE for the same 4 tickers (AAPL, AMZN, GOOGL, MSFT) using VADER+LSTM. Our FinBERT-LSTM produces 0.74–1.42% on those same tickers — substantially better. Caveat: different data periods and different model implementation; not a controlled comparison.

**4. Original paper's ~1.4% MAPE (on NDX index) replicates at stock level.**
We get 0.74–1.42% for GOOGL and AAPL, 2.84% for NVDA, 3.07% for TSLA. Individual stocks are more volatile than the NDX index, so higher MAPE is expected. The results are consistent with the paper's claims.

### Implications for Stage 3 model design

| Finding | Implication |
|---------|-------------|
| Sentiment helps 4/7 tickers, hurts 3/7 | Our 14-dimension LLM features need per-ticker validation — don't assume all dimensions help all tickers |
| META needs temporal architecture | LSTM or temporal model is required; Ridge/LightGBM without sequence structure will likely fail on volatility spikes |
| AMZN: high article volume + simple mean = worst sentiment result | Aggregation method matters. Simple mean dilutes signal. Our LLM approach (materiality-weighted, per-period bucketing) should do better |
| MSFT: smooth ticker, sentiment adds noise | Consider feature selection or regularization — lasso out low-signal dimensions |
| FinBERT-LSTM aggregate wins vs LSTM (2.14% vs 2.32%) | News signal is present at aggregate level. Our richer features (14 dims vs 1 scalar) should amplify this — but must avoid overfitting on ~600 articles |

### This baseline sets the bar for Stage 3

**Our Stage 3 model must beat FinBERT-LSTM (2.14% avg MAPE) on at least 4/7 tickers to claim meaningful improvement over the literature baseline.**

Beating it on AAPL, GOOGL, NVDA is most plausible (news-driven, our structured features should add signal). Beating it on MSFT and AMZN is harder (MSFT: smooth, news adds noise; AMZN: high volume, simple aggregation already noisy).

---

## PART 8: Empirical Baseline Comparison — FinBERT-LSTM vs DeBERTa-TimesNet Framework (2026-03-29)

### Purpose

Head-to-head scientific comparison of Float_Price (raw closing price) prediction across both baseline frameworks in `llm_baseline_model/`. This section documents methodology differences, raw results, confounding variables, and implications for our Stage 3 model design.

### 8.1 Experimental Setup — Side-by-Side

| Aspect | FinBERT-LSTM (our modified run) | DeBERTa-TimesNet Framework |
|--------|--------------------------------|---------------------------|
| **Source paper** | arXiv:2211.07392 | arXiv:2602.00086 (Paper 5 in Part 1) |
| **Tickers** | 7: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA | 5: AAPL, AMZN, MSFT, NFLX, TSLA |
| **Overlapping tickers** | AAPL, AMZN, MSFT, TSLA | AAPL, AMZN, MSFT, TSLA |
| **Date range** | Nov 2023 – Oct 2024 (~1 year, 241 trading days) | Mar 2022 – Apr 2025 (~3 years, ~750 days/ticker) |
| **News source** | Our curated dataset (4,209 articles after alignment) | AlphaVantage API (96,000+ articles) |
| **Total samples/ticker** | ~241 | ~750 |
| **Train/test split** | 85/15 chronological (~205 train, ~36 test) | 80/20 chronological (~600 train, ~150 test) |
| **Validation set** | None | 10% from train (~60 samples) |
| **Sequence length** | 10 days | 30 days |
| **Normalization** | MinMaxScaler, fit on train only | MinMaxScaler, fit on combined features+targets |
| **Sentiment model(s)** | FinBERT only → single scalar [-1, 1] | 6 models: DeBERTa, FinBERT, RoBERTa, SVM, LR, RF |
| **Sentiment features** | 1: mean daily FinBERT score | 7-9: majority vote, counts, probability sums |
| **Prediction target** | Next-day raw closing price ($) | Next-day raw closing price ($) |
| **Metrics computed on** | Inverse-transformed dollar values | Inverse-transformed dollar values |
| **Number of runs** | 1 (single seed) | Multiple seeds |
| **Architectures tested** | MLP, LSTM, FinBERT-LSTM | LSTM, TimesNet, PatchTST, tPatchGNN |
| **Framework** | TensorFlow/Keras | PyTorch |

### 8.2 Architecture Differences — LSTM vs LSTM

Both frameworks use LSTM as their primary architecture for Float_Price, but with substantially different configurations:

| Aspect | FinBERT-LSTM | DeBERTa-TimesNet LSTM |
|--------|-------------|----------------------|
| **LSTM layers** | 3 (70→30→10 units, tanh) | 2 (64→64 units, default activation) |
| **Total LSTM parameters** | Decreasing pyramid: 70→30→10 | Uniform: 64→64 |
| **Dropout** | None | 0.5 after LSTM stack |
| **Learning rate** | 0.02 | 0.001 |
| **LR scheduling** | None | ReduceLROnPlateau (factor=0.5, patience=5) |
| **Epochs** | 100 (no early stopping) | 20 (early stopping, patience=10 on val loss) |
| **Gradient clipping** | None | max_norm=1.0 |
| **Weight initialization** | TensorFlow defaults | Xavier (input), orthogonal (hidden), forget bias=1.0 |
| **Input shape** | (11, 1) — 10 prices + 1 sentiment scalar | (30, 9) — 30 days × 9 features |
| **Sentiment integration** | Appended as 11th timestep | Parallel features at every timestep |
| **Loss function** | MSE | MSE |
| **Optimizer** | Adam | Adam |
| **Batch size** | Default (Keras=32) | 32 |

**Critical observation**: The FinBERT-LSTM uses **zero regularization** — no dropout, no early stopping, no gradient clipping, and a learning rate 20× higher than the DeBERTa-TimesNet LSTM. Combined with only ~36 test samples per ticker, this configuration has significant overfitting risk. The DeBERTa-TimesNet framework applies 4 regularization techniques simultaneously.

### 8.3 Float_Price Results — Full Detail

#### 8.3.1 FinBERT-LSTM Results (from Entry 29)

All metrics computed on inverse-transformed dollar values. Single run (n=1).

**MAPE by ticker and model variant:**

| Ticker | # Articles | MLP MAPE | LSTM MAPE | FinBERT-LSTM MAPE | Best |
|--------|-----------|----------|-----------|-------------------|------|
| AAPL | 667 | 2.01% | 1.71% | **1.42%** | FinBERT-LSTM |
| AMZN | 941 | 1.74% | **1.05%** | 1.18% | LSTM |
| GOOGL | 476 | 1.12% | 0.94% | **0.74%** | FinBERT-LSTM |
| META | 426 | 10.94% | **3.68%** | 4.80% | LSTM |
| MSFT | 468 | **0.60%** | 0.61% | 0.92% | MLP |
| NVDA | 733 | 6.22% | 3.34% | **2.84%** | FinBERT-LSTM |
| TSLA | 498 | **3.00%** | 4.89% | 3.07% | MLP |
| **Aggregate** | 4,209 | 3.66% | 2.32% | **2.14%** | FinBERT-LSTM |

**Key patterns**:
- FinBERT-LSTM wins on 4/7 tickers (AAPL, GOOGL, NVDA — news-sensitive stocks)
- Plain LSTM wins on 2/7 (AMZN, META — high-volume / high-volatility)
- MLP wins on 2/7 (MSFT, TSLA — smooth-trending / erratic)
- Sentiment HURTS on 3/7 tickers (AMZN, META, MSFT)

#### 8.3.2 DeBERTa-TimesNet LSTM Results — Float_Price (All Sentiment Models)

All metrics computed on inverse-transformed dollar values. Multiple seeds averaged.

**AAPL:**

| Sentiment Model | MAE ($) | RMSE ($) | CORR | MAPE | RSE | MSLE |
|----------------|---------|----------|------|------|-----|------|
| DeBERTa | 7.07 | 8.43 | 0.707 | 3.02% | 0.804 | 0.00131 |
| FinBERT | 8.91 | 10.33 | 0.805 | 3.78% | 0.984 | 0.00197 |
| LR | 9.22 | 10.61 | 0.785 | 3.93% | 1.011 | 0.00210 |
| RF | 9.47 | 10.95 | 0.765 | 4.02% | 1.044 | 0.00222 |
| **RoBERTa** | **5.83** | **7.28** | 0.746 | **2.50%** | **0.694** | **0.00097** |
| SVM | 9.16 | 10.56 | 0.774 | 3.90% | 1.006 | 0.00208 |

**AMZN:**

| Sentiment Model | MAE ($) | RMSE ($) | CORR | MAPE | RSE | MSLE |
|----------------|---------|----------|------|------|-----|------|
| **DeBERTa** | **6.30** | **8.04** | 0.919 | **2.99%** | **0.435** | **0.00145** |
| FinBERT | 11.53 | 13.87 | 0.913 | 5.41% | 0.751 | 0.00438 |
| LR | 10.47 | 12.70 | 0.925 | 4.91% | 0.688 | 0.00368 |
| RF | 11.41 | 13.42 | 0.941 | 5.35% | 0.727 | 0.00409 |
| RoBERTa | 10.29 | 12.47 | 0.925 | 4.83% | 0.676 | 0.00355 |
| SVM | 9.58 | 11.97 | 0.926 | 4.46% | 0.649 | 0.00319 |

**MSFT:**

| Sentiment Model | MAE ($) | RMSE ($) | CORR | MAPE | RSE | MSLE |
|----------------|---------|----------|------|------|-----|------|
| **DeBERTa** | **20.00** | **21.84** | 0.943 | **5.31%** | **0.872** | **0.00352** |
| FinBERT | 25.26 | 26.99 | 0.962 | 6.68% | 1.077 | 0.00537 |
| LR | 22.31 | 24.37 | 0.945 | 5.90% | 0.973 | 0.00434 |
| RF | 21.91 | 24.13 | 0.948 | 5.77% | 0.963 | 0.00422 |
| RoBERTa | 22.59 | 24.58 | 0.957 | 5.96% | 0.981 | 0.00439 |
| SVM | 20.90 | 23.15 | 0.945 | 5.51% | 0.924 | 0.00388 |

**NFLX (not in FinBERT-LSTM dataset):**

| Sentiment Model | MAE ($) | RMSE ($) | CORR | MAPE | RSE | MSLE |
|----------------|---------|----------|------|------|-----|------|
| DeBERTa | 26.91 | 30.32 | 0.888 | 5.73% | 0.504 | 0.00445 |
| **FinBERT** | **23.55** | **28.93** | **0.929** | **4.81%** | **0.481** | **0.00349** |
| LR | 23.87 | 29.14 | 0.922 | 4.93% | 0.485 | 0.00371 |
| RF | 26.02 | 32.76 | 0.905 | 5.30% | 0.545 | 0.00453 |
| RoBERTa | 27.94 | 34.93 | 0.917 | 5.69% | 0.581 | 0.00531 |
| SVM | 26.02 | 32.17 | 0.908 | 5.34% | 0.535 | 0.00450 |

**TSLA:**

| Sentiment Model | MAE ($) | RMSE ($) | CORR | MAPE | RSE | MSLE |
|----------------|---------|----------|------|------|-----|------|
| DeBERTa | 13.83 | 15.87 | 0.783 | 6.19% | 0.680 | 0.00517 |
| FinBERT | 27.00 | 29.79 | 0.793 | 11.89% | 1.277 | 0.01959 |
| LR | 11.37 | 13.81 | 0.831 | 5.03% | 0.592 | 0.00377 |
| **RF** | **10.36** | **12.86** | **0.845** | **4.62%** | **0.552** | **0.00328** |
| RoBERTa | 10.22 | 13.06 | 0.845 | 4.73% | 0.560 | 0.00350 |
| SVM | 10.79 | 13.18 | 0.843 | 4.81% | 0.565 | 0.00348 |

**Cross-ticker summary (best sentiment model per ticker):**

| Ticker | Best Sentiment | MAE ($) | RMSE ($) | CORR | MAPE |
|--------|---------------|---------|----------|------|------|
| AAPL | RoBERTa | 5.83 | 7.28 | 0.746 | 2.50% |
| AMZN | DeBERTa | 6.30 | 8.04 | 0.919 | 2.99% |
| MSFT | DeBERTa | 20.00 | 21.84 | 0.943 | 5.31% |
| NFLX | FinBERT | 23.55 | 28.93 | 0.929 | 4.81% |
| TSLA | RF | 10.36 | 12.86 | 0.845 | 4.62% |
| **Average** | — | **13.21** | **15.79** | **0.876** | **4.05%** |

#### 8.3.3 Non-LSTM Architectures (DeBERTa-TimesNet Framework)

The framework also tested Float_Price with transformer and graph architectures. These report metrics on **normalized** value ranges (not dollar-scale), making direct MAPE comparison with LSTM invalid. Correlation (CORR) is scale-independent and comparable.

| Architecture | Best Sentiment | Avg CORR | Notes |
|-------------|---------------|----------|-------|
| **LSTM** | varies | **0.876** | Dominant on Float_Price |
| **tPatchGNN** | FinBERT | 0.827 | Competitive; temporal graph structure helps |
| **TimesNet** | SVM | 0.581 | Poor on price regression |
| **PatchTST** | LR | 0.446 | Poor on price regression |

**Finding**: LSTM is clearly the best architecture for Float_Price prediction. Transformer-based models (TimesNet, PatchTST) are designed for pattern recognition and classification — they do not outperform LSTM on direct price regression. tPatchGNN's graph structure captures inter-stock relationships that help, but still falls short of LSTM.

### 8.4 Head-to-Head Comparison on Overlapping Tickers

Four tickers appear in both frameworks: AAPL, AMZN, MSFT, TSLA. MAPE is the only scale-independent metric available from both.

| Ticker | FinBERT-LSTM MAPE | DeBERTa-TimesNet MAPE (best) | Difference | Winner |
|--------|-------------------|------------------------------|------------|--------|
| AAPL | 1.42% | 2.50% (RoBERTa) | -1.08 pp | FinBERT-LSTM |
| AMZN | 1.18% | 2.99% (DeBERTa) | -1.81 pp | FinBERT-LSTM |
| MSFT | 0.92% | 5.31% (DeBERTa) | -4.39 pp | FinBERT-LSTM |
| TSLA | 3.07% | 4.62% (RF) | -1.55 pp | FinBERT-LSTM |
| **Average** | **1.65%** | **3.86%** | **-2.21 pp** | **FinBERT-LSTM** |

FinBERT-LSTM wins on all 4 overlapping tickers, with the largest gap on MSFT (-4.39 percentage points).

### 8.5 Confounding Variables — Why This Comparison Is Not Controlled

**The 2.21 percentage-point difference cannot be attributed to model architecture alone.** The following confounds make a direct causal comparison impossible:

#### Confound 1: Different test periods (MOST IMPACTFUL)

- **FinBERT-LSTM test period**: roughly Aug–Oct 2024. This was a low-volatility bull market with smooth trending prices. Low volatility → low MAPE because prices change less day-to-day, making "predict ≈ yesterday" easier.
- **DeBERTa-TimesNet test period**: roughly Oct 2024 – Apr 2025 (20% of 3 years). This likely includes the late-2024 / early-2025 period with higher uncertainty. Additionally, 2022 bear market data is in the training set, meaning the model learned from a different regime than the test.

*Expected impact*: Bull market test periods systematically produce 1-3% lower MAPE. This alone could explain most of the difference.

#### Confound 2: Training data proportion

- FinBERT-LSTM: 85% train = ~205 days for ~36 test days (5.7:1 ratio)
- DeBERTa-TimesNet: 72% effective train (after validation) = ~540 days for ~150 test days (3.6:1 ratio)

More training data per test sample generally helps, especially with LSTMs that need sufficient sequence examples.

#### Confound 3: No validation discipline in FinBERT-LSTM

Without early stopping or a validation set, the FinBERT-LSTM trains for a fixed 100 epochs. If this happens to be near-optimal for the test period, the model looks good — but this is not guaranteed to generalize. The DeBERTa-TimesNet framework's early stopping (patience=10) sacrifices peak in-sample performance for better generalization.

#### Confound 4: Sequence length and feature dimensionality

- FinBERT-LSTM: 10-day lookback, 1 feature → model learns short-term momentum (easy)
- DeBERTa-TimesNet: 30-day lookback, 9 features → model must learn complex multi-feature patterns (harder)

Shorter lookback with fewer features produces a simpler learning problem. The 10-day window essentially learns "extrapolate the recent 2-week trend" — which works well in trending markets.

#### Confound 5: Single run vs averaged seeds

FinBERT-LSTM reports one run. This could be an above-average or below-average run — we have no way to know. The DeBERTa-TimesNet framework averages across seeds, which provides a more realistic central estimate but includes runs that may have converged suboptimally.

### 8.6 Sentiment Model Analysis (DeBERTa-TimesNet Framework)

The DeBERTa-TimesNet framework's 6 sentiment models provide a unique view into the value of sentiment sophistication:

**Average Float_Price metrics across all 5 tickers:**

| Sentiment Model | Avg MAE ($) | Avg MAPE | Avg CORR | Type |
|----------------|-------------|----------|----------|------|
| DeBERTa | **14.82** | **4.65%** | 0.848 | Transformer (finance-tuned) |
| SVM | 15.47 | 4.80% | 0.879 | ML ensemble of 3 transformers |
| RoBERTa | 15.37 | 4.74% | 0.878 | Transformer (social media) |
| LR | 15.45 | 4.94% | 0.881 | ML ensemble of 3 transformers |
| RF | 15.83 | 5.01% | 0.881 | ML ensemble of 3 transformers |
| FinBERT | 19.05 | **6.51%** | **0.881** | Transformer (finance-specific) |

**Key observations:**

1. **Bias-variance tradeoff**: DeBERTa has the lowest MAE/MAPE (best bias) but lowest CORR (worst variance capture). FinBERT has the highest CORR (best variance capture) but worst MAE/MAPE (worst bias). DeBERTa tracks the price level better; FinBERT tracks the pattern of ups and downs better but is systematically offset.

2. **Traditional ML matches transformers**: SVM, LR, and RF (which ensemble the 3 transformer models) perform comparably to individual transformers. This suggests the **sentiment signal itself — not the model sophistication — is the binding constraint** for price prediction. More sophisticated NLP does not proportionally improve downstream price predictions.

3. **FinBERT is worst on MAE despite being the "finance-specific" model**: Its finance-domain training produces sentiment scores that capture financial relevance (high CORR) but introduce systematic level bias (high MAE). This may be because FinBERT's training data distribution differs from the test period's news characteristics.

4. **No sentiment model dominates across all tickers**: AAPL prefers RoBERTa, AMZN prefers DeBERTa, MSFT prefers DeBERTa, NFLX prefers FinBERT, TSLA prefers RF. This per-ticker variation mirrors our finding from Entry 29 that sentiment impact is stock-specific.

### 8.7 Scientific Assessment

#### What we can conclude with confidence:

1. **LSTM is the best architecture for Float_Price prediction** among all tested (LSTM, TimesNet, PatchTST, tPatchGNN). Transformer architectures are not competitive on price-level regression for individual stocks at this data scale.

2. **Sentiment adds marginal value to price prediction.** FinBERT-LSTM's own results show sentiment hurts on 3/7 tickers (Entry 29). The DeBERTa-TimesNet framework shows near-identical CORR (~0.85-0.88) across all 6 sentiment models — the downstream LSTM captures most of the predictable variance from price history alone.

3. **Sentiment impact is stock-specific.** Both frameworks confirm that different stocks respond differently to news features. This validates our per-ticker model approach and suggests per-ticker feature selection will be important.

4. **Low MAPE does not imply useful predictions.** Both frameworks achieve 1-5% MAPE, which sounds impressive but is achievable because tomorrow's price ≈ today's price. Neither framework reports comparison to a naive "predict yesterday's close" baseline — a critical omission. The DeBERTa-TimesNet framework's Binary_Price accuracy (~60%) is a more honest measure of actual directional predictive power.

5. **Traditional ML sentiment models match transformer sentiment models** for downstream price prediction. The sentiment signal is the bottleneck, not the sentiment model.

#### What we cannot conclude:

1. **Which architecture is truly better** — the FinBERT-LSTM vs DeBERTa-TimesNet comparison is confounded by 5+ variables (time period, train size, validation, sequence length, single vs multi-run). A controlled experiment with identical data, splits, and evaluation would be needed.

2. **Whether the MAPE differences are statistically significant** — neither framework reports confidence intervals or statistical tests (e.g., Diebold-Mariano). The observed 2.21 pp difference on overlapping tickers could be within noise bounds.

3. **Whether any model outperforms naive baseline** — without a "predict yesterday's close" comparison, we cannot separate genuine predictive ability from autocorrelation exploitation.

### 8.8 Implications for Stage 3 Model Design

| Finding | Implication for Stage 3 |
|---------|------------------------|
| LSTM dominates Float_Price | Use LSTM (or LSTM-based) as primary downstream architecture, not just Ridge/LightGBM |
| Sentiment adds marginally on average | Our 14-dimension features must provide MORE signal than scalar sentiment, not just more dimensions. Feature selection is critical — ablation required |
| Sentiment hurts 3/7 tickers | Per-ticker feature gating or regularization needed. Not all 14 dimensions should be active for all tickers |
| No sentiment model dominates across tickers | Our LLM-extracted structured features should be evaluated per-ticker, not on aggregate alone |
| Transformers (TimesNet, PatchTST) underperform on price regression | Do not use transformer architectures for Float_Price. May revisit for Binary_Price |
| Bias-variance tradeoff in sentiment | Consider separate loss terms or multi-task learning to balance level accuracy (bias) and movement tracking (variance) |
| MAPE alone is insufficient | Report: MAPE, MAE, CORR, directional accuracy, naive baseline delta, and confidence intervals |
| Both frameworks omit naive baseline | Our Stage 3 MUST include naive baseline ("predict yesterday's close") to prove actual predictive value |

### 8.9 Updated Stage 3 Targets

Based on this dual-baseline comparison:

| Metric | Target | Source |
|--------|--------|--------|
| **Float_Price MAPE** | Beat 2.14% avg on ≥4/7 tickers | FinBERT-LSTM (Entry 29) |
| **Directional accuracy** | Beat 60% on ≥4/5 tickers | DeBERTa-TimesNet Binary_Price |
| **Naive baseline delta** | MAPE < naive "yesterday's close" MAPE | Both frameworks omit this — we must include it |
| **Statistical significance** | p < 0.05 on Diebold-Mariano test | Paper 3 standard |
| **Confidence intervals** | Report 95% CI via bootstrap or multi-seed | Neither framework reports — we must |
| **Per-ticker consistency** | Improvement on ≥4/7 tickers, not just aggregate | Prevent aggregate masking per-ticker failures |

---

## PART 10: Cross-Paper Results Comparison (2026-03-30)

### Purpose

Consolidate reported results from all reviewed papers into comparable tables. Different papers use different tasks (price regression vs direction classification vs trading), different metrics, and different stocks — direct comparison is often impossible. This section documents what each paper actually reports so we know what bar exists.

### 10.1 Price Prediction (MAPE) — The Closest Comparisons to Our Task

These papers predict next-day closing price and report MAPE. Lower = better.

#### Paper 1: VADER + LSTM (arXiv:2505.05325) — 4 stocks, 1 year, 60-day window

| Ticker | MAE ($) | MSE | RMSE ($) | MAPE |
|--------|---------|-----|----------|------|
| AAPL | 6.12 | 58.03 | 7.62 | 2.72% |
| GOOGL | 5.89 | 52.14 | 7.22 | 2.65% |
| MSFT | 6.45 | 60.27 | 7.76 | 2.91% |
| AMZN | 6.78 | 65.43 | 8.09 | 3.05% |
| **Avg** | **6.31** | **58.97** | **7.67** | **2.83%** |

Sentiment contribution: AAPL MAPE drops from 3.15% → 2.72% with VADER (~15.8% relative improvement).

#### Paper 6: FinBERT-LSTM Hierarchical (arXiv:2407.16150) — NASDAQ-100 aggregate

| Model | MAPE | MAE | Notes |
|-------|------|-----|-------|
| DNN (no sentiment) | 22.0% | 489.32 | Feedforward baseline |
| LSTM (no sentiment) | 7.2% | 183.36 | Price-only temporal |
| **FinBERT-LSTM** | **4.5%** | **173.67** | 3-level hierarchy + learned weights |

No per-stock breakdown. Aggregate across NASDAQ-100 stocks. Sentiment reduces MAPE from 7.2% → 4.5% (37.5% relative improvement). MAE is in raw price units (index-level).

#### Paper 7: DeBERTa-TimesNet (arXiv:2602.00086) — 5 stocks, 3 years

This paper reports DeltaPrice MAE/RMSE (price change, not price level) and trend classification accuracy — not Float_Price MAPE in the paper tables. Their key finding: **"Sentiment news has very low impact on the obtained results."** The DeltaPrice results averaged across 5 stocks:

| Architecture | No Sentiment MAE | With Sentiment (SVM) MAE | Change |
|--------------|-----------------|--------------------------|--------|
| LSTM | 6.063 | 6.097 | +0.6% (worse) |
| PatchTST | 0.347 | 0.306 | -11.8% (better) |
| TimesNet | 0.440 | 0.266 | -39.5% (better) |
| tPatchGNN | 0.137 | 0.143 | +4.4% (worse) |

Sentiment helps transformer-based models, hurts LSTM and graph models on DeltaPrice.

#### Our Baselines (our data, our runs)

| Ticker | MLP | LSTM | FinBERT-LSTM | FinBERT v1 LSTM* | DeBERTa v3 LSTM* |
|--------|-----|------|-------------|------------------|-------------------|
| AAPL | 2.01% | 1.71% | 1.42% | 0.99% | 0.99% |
| AMZN | 1.74% | 1.05% | 1.18% | 1.60% | 1.87% |
| GOOGL | 1.12% | 0.94% | 0.74% | 0.74% | 0.78% |
| META | 10.94% | 3.68% | 4.80% | 5.20% | 3.47% |
| MSFT | 0.60% | 0.61% | 0.92% | 0.83% | 0.85% |
| NVDA | 6.22% | 3.34% | 2.84% | 3.84% | 4.44% |
| TSLA | 3.00% | 4.89% | 3.07% | 1.78% | 2.84% |
| **Avg** | **3.66%** | **2.32%** | **2.14%** | **2.14%** | **2.18%** |

*FinBERT v1 and DeBERTa v3 columns are from the DeBERTa-TimesNet framework run on our data with different sentiment models.

#### MAPE Comparison Across All Sources (overlapping tickers only)

| Ticker | Paper 1 (VADER) | Paper 6 (FinBERT-H) | Our FinBERT-LSTM | Our DeBERTa v3 |
|--------|-----------------|---------------------|------------------|----------------|
| AAPL | 2.72% | — | 1.42% | 0.99% |
| AMZN | 3.05% | — | 1.18% | 1.87% |
| GOOGL | 2.65% | — | 0.74% | 0.78% |
| MSFT | 2.91% | — | 0.92% | 0.85% |
| **Avg** | **2.83%** | **4.5% (NDX agg)** | **1.07%** | **1.12%** |

Our baselines beat Paper 1 on all 4 overlapping tickers. Caveat: different time periods, different data, not a controlled comparison (see Part 8.5 for confounds).

### 10.2 Direction/Classification (Accuracy, MCC) — Binary Movement Prediction

These papers predict up/down movement. Higher accuracy and MCC = better. Random baseline = 50% accuracy, 0.0 MCC.

| Paper | Method | News Processing | Accuracy | MCC | Dataset |
|-------|--------|----------------|----------|-----|---------|
| **LLMFactor** (GPT-4) | LLM discovers factors | Per-article factor extraction | **66.3%** | **0.238** | StockNet (87 US stocks) |
| **LLMFactor** (GPT-4) | Same | Same | **65.3%** | **0.284** | CMIN-US (110 US stocks) |
| **LLMFactor** (GPT-3.5) | Same | Same | 57.6% | 0.145 | StockNet |
| **SEP** (GPT-4) | Self-reflective LLM | Summarize → predict → reflect | 54.4% | 0.099 | Informative texts |
| **DeBERTa-TimesNet** | LSTM + sentiment | DeBERTa/FinBERT scores | 57.5% | — | 5 tech stocks |
| FinBERT baseline | Sentiment classifier | FinBERT polarity | 55.4% | 0.111 | StockNet |
| FinGPT | Fine-tuned LLM | Direct prediction | 47.6% | 0.016 | StockNet |
| GPT-4-turbo (sentiment) | Prompt-based | Sentiment prompt | 53.6% | 0.060 | StockNet |
| Random | — | — | 50.0% | 0.000 | — |

**Key observations**:
- LLMFactor's factor discovery is the clear winner at 66% — discovering what matters per stock adds ~11pp over FinBERT sentiment (55%)
- SEP's self-reflection only reaches 54% — marginal above random
- All methods are in the 50-66% range — binary stock movement prediction remains hard
- MCC matters more than accuracy here (accounts for class imbalance)

#### LLMFactor Ablation — What Drives the Improvement

| Component | Accuracy | MCC | Delta vs Base |
|-----------|----------|-----|---------------|
| Price only | ~52% | 0.041 | — |
| + Factor layer (LLM-discovered) | ~58% | 0.166 | +6pp / +0.125 |
| + Factor + Relation layer | ~63% | 0.203 | +11pp / +0.162 |
| Full SKGP (GPT-4) | 66.3% | 0.238 | +14pp / +0.197 |

The LLM-discovered factors alone add 6pp. Adding inter-stock relations adds another 5pp. This is the strongest evidence that **LLM-based factor discovery beats fixed sentiment features**.

### 10.3 Trading/Factor Performance (Sharpe, Returns)

These papers report trading strategy results, not prediction accuracy.

| Paper | Method | Ann. Return | Sharpe | Notes |
|-------|--------|-------------|--------|-------|
| **Structured Event Repr.** (arXiv:2512.19484) | GPT triplets → TransE embeddings | 10.93% (daily) | 0.78 | Cross-sectional US stocks, 2003-2022 |
| **Event-Aware** (arXiv:2508.07408) | 70+ event categories | 8% | 5.0 | Lexicon-based; max drawdown -15.2% |
| **Event-Aware** (Geopolitical) | Single category, contrarian | — | -0.70 | Strongest single-category short signal |
| **Event-Aware** (Brand Boycott) | Single category | — | IC=0.612 | Highest information coefficient at 1-day |
| **MarketSenseAI 2.0** | 5 LLM agents | 125.9% vs 73.5% index | — | S&P 100, but no Sharpe reported |

Not directly comparable to our price prediction task, but informative about the value of structured news processing over sentiment.

### 10.4 Cross-Paper Synthesis

#### What's comparable and what isn't

| Comparison | Possible? | Why / Why Not |
|------------|-----------|---------------|
| Our MAPE vs Paper 1 MAPE | Partially | Same stocks (4 overlap), same task, but different time periods and data |
| Our MAPE vs Paper 6 MAPE | No | They report NASDAQ-100 aggregate, not per-stock |
| Our accuracy vs LLMFactor | No | Different task (we predict price, they predict direction), different stocks |
| Our features vs any paper | No | No paper uses 12-15 structured dimensions — no benchmark exists |

#### The gap in the literature

No paper reports MAPE (or any price prediction metric) from LLM-structured multi-dimensional news features fed to a downstream model. The field splits into:
1. **Price prediction papers** (Papers 1, 6, 7) — use simple sentiment (1-3 features), report MAPE
2. **Classification papers** (LLMFactor, SEP) — use richer LLM features, report accuracy/MCC
3. **Trading papers** (Structured Event, Event-Aware) — use structured extraction, report Sharpe

Our pipeline sits at the intersection: rich LLM-structured features (like group 2-3) for price prediction (like group 1). This means our Stage 3 results will be the first to fill this gap — and there is no external benchmark to compare against directly.

#### What the numbers tell us about our approach

| Evidence | Source | Implication |
|----------|--------|-------------|
| LLM factor discovery adds +14pp accuracy over price-only | LLMFactor ablation | Our category discovery should add signal |
| Simple sentiment adds ~5-16% relative MAPE improvement | Papers 1, 6 | Our richer features should do at least this well |
| Sentiment hurts some tickers | Paper 2, our Entry 29 | Per-ticker feature selection is critical |
| More sophisticated sentiment models ≠ better price prediction | Paper 7, our Part 8.6 | The signal quality matters, not the NLP model sophistication |
| Structured extraction > embeddings > sentiment for trading | Paper 6 (arXiv:2512.19484) | Our structured approach is on the right track |
| 50-66% is the realistic range for direction prediction | All classification papers | Don't expect >70% directional accuracy |

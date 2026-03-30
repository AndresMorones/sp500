# FinBERT-LSTM — Adapted Baseline

Adaptation of [FinBERT-LSTM (arXiv:2211.07392)](https://arxiv.org/pdf/2211.07392.pdf) for our multi-ticker stock prediction dataset.

## What changed from the original

| Aspect | Original | Adapted |
|--------|----------|---------|
| Tickers | NDX (Nasdaq-100 index) | AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA |
| News source | NYT API (generic market news) | Our `data/raw/news.csv` (ticker-specific) |
| Price source | Yahoo Finance (yfinance) | Our `data/raw/price.csv` |
| News format | Wide (10 columns per day) | Long (one row per article) |
| Sentiment | Per-day average of 10 articles | Per (ticker, date) mean of all articles |
| Scaler | `fit_transform` on both train AND test (data leakage) | `fit` on train only, `transform` on test |
| Models | Single ticker | Per-ticker loop (7 independent models) |
| Prediction target | Raw Close price | Raw Close price (same — for fair comparison) |

**Note on prediction target:** Our project's own baselines (Ridge, LightGBM in `src/model_baseline.py`) use excess returns. This baseline predicts raw Close price to match the original paper. Individual stocks are more volatile than the NDX index, so MAPE will be higher than the original's ~1.4%.

## How to run

Run scripts sequentially (each depends on the previous):

```bash
cd llm_baseline_model_modified/FinBERT-LSTM

# Data preparation
python 1_news_collection.py      # Load news → news_data.csv
python 2_stock_data_collection.py # Load prices → stock_price.csv
python 3_news_data_cleaning.py    # Align dates → overwrites news_data.csv

# Sentiment
python 4_news_sentiment_analysis.py  # FinBERT → sentiment.csv (~5-30 min)

# Models
python 5_MLP_model.py            # MLP baseline → mlp_results.csv
python 6_LSTM_model.py           # LSTM baseline → lstm_results.csv
python 7_lstm_model_bert.py      # FinBERT-LSTM → bert_lstm_results.csv

# Analysis
python analysis.py               # Comparison plots + summary table
```

## Dependencies

```
pandas
numpy
scikit-learn
tensorflow
transformers
matplotlib
```

## Output files

| File | Description |
|------|-------------|
| `news_data.csv` | Cleaned news articles (date, ticker, text) |
| `stock_price.csv` | Price data in original schema |
| `sentiment.csv` | FinBERT scores per (ticker, date) |
| `mlp_results.csv` | MLP predictions (date, ticker, actual, predicted) |
| `lstm_results.csv` | LSTM predictions |
| `bert_lstm_results.csv` | FinBERT-LSTM predictions |
| `plots/` | Per-ticker comparison charts + summary CSV |

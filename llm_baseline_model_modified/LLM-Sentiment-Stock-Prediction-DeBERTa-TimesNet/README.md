# Stock Prediction Framework (Adapted for sp500 Project)

A comprehensive framework for predicting stock prices by analyzing the effect of financial news on stock movements. Uses state-of-the-art NLP models (FinBERT, RoBERTa, DeBERTa) for sentiment analysis and deep learning architectures (LSTM, TimesNet, PatchTST, tPatchGNN) for time-series forecasting.

## Adaptation Notes

This is an adapted version of the original DeBERTa-TimesNet framework, modified to work with the sp500 project's consolidated data format:

- **7 tickers**: AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA
- **Data source**: Single consolidated CSVs (`data/raw/price.csv` and `data/raw/news.csv`)
- **Sentiment models**: 3 transformer models only (FinBERT, RoBERTa, DeBERTa) — classical models (SVM, LR, RF) removed since no human-labeled benchmark dataset is available
- **Price data**: Nov 2023 onwards (daily OHLCV)
- **News data**: Oct 2024 onwards (headline + summary)

## Project Structure

```
├── layers/                  # Neural network layer definitions
│   ├── Formers/            # Transformer components (attention, embedding, enc/dec)
│   └── TimesNet/           # TimesNet convolution blocks
├── sentiment_analysis/      # NLP sentiment pipeline
│   ├── models/             # Base class + BERT wrapper
│   ├── comparison/         # Financial and stock sentiment comparisons
│   ├── trainers/           # Classical model trainer (unused in this adaptation)
│   └── main.py             # Entry point: runs 3 transformers on per-ticker news
├── stock_prediction/        # Time-series prediction pipeline
│   ├── config.py           # Paths, tickers, model config
│   ├── dataset.py          # Data loading & sentiment aggregation
│   ├── dataset_pipeline.py # Sequence creation & normalization
│   ├── features.py         # Target variable generation
│   ├── plots.py            # EDA visualizations
│   ├── trading_simulation.py # Backtesting engine
│   └── modeling/           # LSTM, TimesNet, PatchTST, tPatchGNN (train + predict)
├── utils/                   # Config dataclass, metrics, tools
├── scripts/                 # Shell scripts for batch training & analysis
├── Makefile                 # Convenience commands
└── requirements.txt         # Dependencies
```

## Getting Started

1. **Install dependencies**
   ```
   make requirements
   ```

2. **Run sentiment analysis** (processes all 7 tickers through FinBERT/RoBERTa/DeBERTa)
   ```
   make sentiment_analysis
   ```

3. **Process datasets** (merge price + sentiment, generate target features)
   ```
   make data
   ```

4. **Generate EDA plots**
   ```
   make plots
   ```

5. **Train all model architectures** (LSTM, TimesNet, PatchTST, tPatchGNN)
   ```
   make run_all
   ```

6. **Run trading simulation**
   ```
   make simulate
   ```

## Data Format

**Input** (from `../../data/raw/`):
- `price.csv`: `date, ticker, open, high, low, close, volume`
- `news.csv`: `datetime, ticker, headline, summary`

**Processed** (in `data/processed/`):
- `{TICKER}_preprocessed_dataset_with_features.csv`: Price + aggregated sentiment + target columns

**Targets**:
- `Float_Price`: Next day's closing price
- `Binary_Price`: 1 if next close > today's close
- `Factor_Price`: next_close / today_close
- `Delta_Price`: next_close - today_close

## Reference

Based on: "Evaluating Large Language Models and Advanced Time-Series Architectures for Sentiment-Driven Stock Movement Prediction" — Walid Siala, Ahmed Khanfir, Mike Papadakis (University of Luxembourg, 2025)

--------

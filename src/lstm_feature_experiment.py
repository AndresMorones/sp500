"""
Controlled LSTM Feature Comparison Experiment.

Tests 6 feature configurations × 2 prediction targets (gap/cc) × 5 seeds × 7 tickers.
All hyperparameters are fixed — only features vary.

Predicts RETURNS (% change), not raw price levels:
  Gap model: overnight/pre-market news → predicts (open - prev_close) / prev_close
  CC model:  all news (includes gap news) → predicts (close - prev_close) / prev_close

Usage:
    python src/lstm_feature_experiment.py                    # run all
    python src/lstm_feature_experiment.py --config A B C     # run specific configs
    python src/lstm_feature_experiment.py --target gap       # gap only
    python src/lstm_feature_experiment.py --ticker AAPL MSFT # specific tickers
    python src/lstm_feature_experiment.py --skip-sentiment   # use cached sentiment
"""
import argparse
import csv
import json
import math
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy import stats

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output", "lstm_feature_experiment")

PRICE_CSV = os.path.join(RAW_DIR, "price.csv")
NEWS_CSV = os.path.join(RAW_DIR, "news.csv")
SENTIMENT_CACHE = os.path.join(OUT_DIR, "sentiment_per_article.csv")

# ── Sentiment model registry ─────────────────────────────────────────────────
SENTIMENT_MODELS = {
    "finbert": {
        "hf_name": "ProsusAI/finbert",
        "type": "classifier",
    },
    "deberta": {
        "hf_name": "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis",
        "type": "classifier",
    },
    "gemma": {
        "hf_name": "google/gemma-3-1b-it",
        "type": "generative",
    },
    "qwen": {
        "hf_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "type": "generative",
    },
    "llama-finsent": {
        "hf_name": "oopere/Llama-FinSent-S",
        "type": "generative",
    },
    "llama32": {
        "hf_name": "meta-llama/Llama-3.2-1B-Instruct",
        "type": "generative",
    },
}

# 7-class sentiment scale for generative models
_7CLASS_MAP = {
    "strong positive": 1.0,
    "moderately positive": 0.66,
    "mildly positive": 0.33,
    "neutral": 0.0,
    "mildly negative": -0.33,
    "moderately negative": -0.66,
    "strong negative": -1.0,
}

_PROMPT_TEMPLATE = (
    "What is the sentiment of this news? Please choose an answer from "
    "{{strong negative/moderately negative/mildly negative/"
    "neutral/mildly positive/moderately positive/strong positive}}\n"
    "Input: {text}\n"
    "Answer:"
)

_CHAT_TEMPLATE_MODELS = {
    "google/gemma-3-1b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
}

# ── Fixed hyperparameters ──────────────────────────────────────────────────────
LOOKBACK = 10
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08  # of total (remaining 0.20 = test)
LSTM_UNITS = (32, 16)
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
SEEDS = [16, 32, 42, 64, 128]

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR = 16

# ── Feature configurations ─────────────────────────────────────────────────────
# Each config maps to a list of column names in the prepared DataFrame
FEATURE_CONFIGS = {
    "A": ["close_return"],
    "B": ["close_return", "volume"],
    "C": ["close_return", "sentiment"],
    "D": ["close_return", "volume", "sentiment"],
    "E": ["close_return", "volume", "news_count", "sentiment"],
    "F": ["close_return", "volume", "news_count",
          "sentiment",
          "count_positive", "count_negative", "count_neutral",
          "label_positive_sum", "label_negative_sum", "label_neutral_sum"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & NEWS SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

def load_price_data():
    """Load price.csv → DataFrame with date, ticker, open, close, volume."""
    df = pd.read_csv(PRICE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def load_and_split_news():
    """Load news.csv, split each article into gap or cc bucket using
    score_pipeline.py timing rules.

    Returns two DataFrames (gap_news, cc_news) with columns:
        [datetime, date_target, ticker, headline, summary, text]

    cc bucket includes gap articles (cc spans full day).
    """
    df = pd.read_csv(NEWS_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["text"] = df["headline"].fillna("") + ". " + df["summary"].fillna("")

    # Build sorted trading dates per ticker for next-day lookup
    price_df = pd.read_csv(PRICE_CSV)
    trading_dates = {}
    for ticker in TICKERS:
        dates = sorted(price_df[price_df["ticker"] == ticker]["date"].unique())
        trading_dates[ticker] = dates

    def next_trading_day(ticker, date_str):
        dates = trading_dates.get(ticker, [])
        for d in dates:
            if d > date_str:
                return d
        return None

    gap_rows = []
    cc_rows = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        if ticker not in TICKERS:
            continue

        dt = row["datetime"]
        date_str = dt.strftime("%Y-%m-%d")
        hour, minute = dt.hour, dt.minute
        article = {
            "datetime": dt,
            "ticker": ticker,
            "headline": row["headline"],
            "summary": row["summary"],
            "text": row["text"],
        }

        if hour < MARKET_OPEN_HOUR or (hour == MARKET_OPEN_HOUR and minute < MARKET_OPEN_MIN):
            # Pre-market → today's gap
            article["date_target"] = date_str
            gap_rows.append(article)
        elif hour < MARKET_CLOSE_HOUR:
            # Market hours → today's cc only
            article["date_target"] = date_str
            cc_rows.append(article)
        else:
            # After close → next trading day's gap
            target = next_trading_day(ticker, date_str)
            if target is None:
                continue
            article["date_target"] = target
            gap_rows.append(article)

    gap_df = pd.DataFrame(gap_rows)
    cc_df = pd.DataFrame(cc_rows)

    # cc includes gap articles (cc return spans full day)
    cc_full = pd.concat([gap_df, cc_df], ignore_index=True)

    print(f"News split: {len(gap_df)} gap articles, {len(cc_full)} cc articles "
          f"(cc includes {len(gap_df)} gap articles)")

    return gap_df, cc_full


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def compute_article_sentiment(news_df, model_key="finbert"):
    """Run sentiment model on each article. Returns DataFrame with sentiment columns.

    Supports classifier models (FinBERT, DeBERTa) and generative models (Gemma, Qwen, Llama).
    Columns added: finbert_score, label, prob_positive, prob_negative, prob_neutral
    """
    model_info = SENTIMENT_MODELS[model_key]
    hf_name = model_info["hf_name"]
    model_type = model_info["type"]

    texts = news_df["text"].tolist()

    if model_type == "classifier":
        scores, labels, prob_pos, prob_neg, prob_neu = _run_classifier(hf_name, texts)
    else:
        scores, labels, prob_pos, prob_neg, prob_neu = _run_generative(hf_name, texts)

    news_df = news_df.copy()
    news_df["finbert_score"] = scores
    news_df["label"] = labels
    news_df["prob_positive"] = prob_pos
    news_df["prob_negative"] = prob_neg
    news_df["prob_neutral"] = prob_neu

    return news_df


def _run_classifier(hf_name, texts):
    """Score articles with a classifier model (FinBERT, DeBERTa)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    print(f"Loading classifier: {hf_name}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                   truncation=True, max_length=512, top_k=None)

    print(f"Running classifier on {len(texts)} articles (batch_size=32)...")
    all_results = nlp(texts, batch_size=32)

    scores, labels, prob_pos, prob_neg, prob_neu = [], [], [], [], []
    for result in all_results:
        probs = {r["label"].lower(): r["score"] for r in result}
        p_pos = probs.get("positive", 0)
        p_neg = probs.get("negative", 0)
        p_neu = probs.get("neutral", 0)
        score = p_pos - p_neg
        label = max(probs, key=probs.get)
        scores.append(score)
        labels.append(label)
        prob_pos.append(p_pos)
        prob_neg.append(p_neg)
        prob_neu.append(p_neu)

    return scores, labels, prob_pos, prob_neg, prob_neu


def _run_generative(hf_name, texts):
    """Score articles with a generative model (Gemma, Qwen, Llama)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import time

    # Device detection
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    print(f"Loading generative model: {hf_name} [{dtype}] on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_name, dtype=dtype, device_map=device, trust_remote_code=True
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Running generative inference on {len(texts)} articles one-by-one...")
    scores, labels, prob_pos, prob_neg, prob_neu = [], [], [], [], []
    start = time.time()

    for i, text in enumerate(texts):
        raw_prompt = _PROMPT_TEMPLATE.format(text=text)
        if hf_name in _CHAT_TEMPLATE_MODELS:
            messages = [{"role": "user", "content": raw_prompt}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = raw_prompt

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)

        # Parse 7-class label → map to score
        gen_lower = generated_text.lower().strip()
        matched_score = 0.0
        matched_label = None
        for lbl in sorted(_7CLASS_MAP.keys(), key=len, reverse=True):
            if lbl in gen_lower:
                matched_score = _7CLASS_MAP[lbl]
                matched_label = lbl
                break

        # Map 7-class to 3-class probabilities for feature compatibility
        if matched_score > 0:
            label = "positive"
            p_pos = abs(matched_score)
            p_neg = 0.0
            p_neu = 1.0 - p_pos
        elif matched_score < 0:
            label = "negative"
            p_neg = abs(matched_score)
            p_pos = 0.0
            p_neu = 1.0 - p_neg
        else:
            label = "neutral"
            p_pos = 0.0
            p_neg = 0.0
            p_neu = 1.0

        scores.append(matched_score)
        labels.append(label)
        prob_pos.append(p_pos)
        prob_neg.append(p_neg)
        prob_neu.append(p_neu)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (len(texts) - i - 1) / rate / 60
            print(f"  [{i+1}/{len(texts)}] {rate:.1f} art/sec, ETA: {remaining:.0f} min")

    return scores, labels, prob_pos, prob_neg, prob_neu


def get_sentiment_articles(gap_news, cc_news, model_key="finbert", skip_sentiment=False):
    """Get or compute per-article sentiment for gap and cc news.

    Uses model-specific cache if available.
    Returns (gap_scored, cc_scored) where cc includes gap articles.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Model-specific cache file
    cache_path = os.path.join(OUT_DIR, f"sentiment_{model_key}.csv")
    sent_cols = ["finbert_score", "label", "prob_positive", "prob_negative", "prob_neutral"]

    if os.path.exists(cache_path):
        print(f"Loading cached {model_key} sentiment from {cache_path}")
        cached = pd.read_csv(cache_path)
        cached["datetime"] = pd.to_datetime(cached["datetime"])

        gap_scored = gap_news.merge(
            cached[["datetime", "ticker", "text"] + sent_cols],
            on=["datetime", "ticker", "text"], how="left"
        )
        cc_scored = cc_news.merge(
            cached[["datetime", "ticker", "text"] + sent_cols],
            on=["datetime", "ticker", "text"], how="left"
        )
        return gap_scored, cc_scored

    if skip_sentiment:
        raise FileNotFoundError(
            f"No cached sentiment for {model_key} at {cache_path}. "
            f"Run without --skip-sentiment first."
        )

    # Combine all unique articles, score once, then split back
    all_articles = pd.concat([gap_news, cc_news], ignore_index=True)
    all_articles = all_articles.drop_duplicates(
        subset=["datetime", "ticker", "text"]
    ).reset_index(drop=True)

    scored = compute_article_sentiment(all_articles, model_key=model_key)

    scored.to_csv(cache_path, index=False)
    print(f"Sentiment cache saved: {len(scored)} articles → {cache_path}")

    gap_scored = gap_news.merge(
        scored[["datetime", "ticker", "text"] + sent_cols],
        on=["datetime", "ticker", "text"], how="left"
    )
    cc_scored = cc_news.merge(
        scored[["datetime", "ticker", "text"] + sent_cols],
        on=["datetime", "ticker", "text"], how="left"
    )

    return gap_scored, cc_scored


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate_sentiment_features(scored_news, price_df, target_type):
    """Aggregate per-article sentiment into daily features per (ticker, date).

    Returns DataFrame aligned with price_df dates, with columns:
        sentiment, news_count, count_positive, count_negative, count_neutral,
        label_positive_sum, label_negative_sum, label_neutral_sum

    target_type: 'gap' or 'cc' — determines which price column is the target.
    """
    records = []

    for ticker in TICKERS:
        ticker_prices = price_df[price_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        ticker_news = scored_news[scored_news["ticker"] == ticker] if len(scored_news) > 0 else pd.DataFrame()

        prev_close = None
        for _, prow in ticker_prices.iterrows():
            date_str = prow["date"].strftime("%Y-%m-%d")

            # Compute returns (% change from previous close)
            if prev_close is not None and prev_close != 0:
                close_return = (prow["close"] - prev_close) / prev_close
                gap_return = (prow["open"] - prev_close) / prev_close
            else:
                close_return = 0.0
                gap_return = 0.0
            prev_close = prow["close"]

            # Match news for this (ticker, date)
            if len(ticker_news) > 0 and "date_target" in ticker_news.columns:
                day_news = ticker_news[ticker_news["date_target"] == date_str]
            else:
                day_news = pd.DataFrame()

            n = len(day_news)
            if n > 0:
                sentiment = day_news["finbert_score"].mean()
                count_pos = (day_news["label"] == "positive").sum()
                count_neg = (day_news["label"] == "negative").sum()
                count_neu = (day_news["label"] == "neutral").sum()
                label_pos_sum = day_news["prob_positive"].sum()
                label_neg_sum = day_news["prob_negative"].sum()
                label_neu_sum = day_news["prob_neutral"].sum()
            else:
                sentiment = 0.0
                count_pos = count_neg = count_neu = 0
                label_pos_sum = label_neg_sum = label_neu_sum = 0.0

            records.append({
                "date": prow["date"],
                "ticker": ticker,
                "close": prow["close"],
                "open": prow["open"],
                "volume": prow["volume"],
                "close_return": close_return,
                "gap_return": gap_return,
                "cc_return": close_return,
                "sentiment": sentiment,
                "news_count": n,
                "count_positive": count_pos,
                "count_negative": count_neg,
                "count_neutral": count_neu,
                "label_positive_sum": label_pos_sum,
                "label_negative_sum": label_neg_sum,
                "label_neutral_sum": label_neu_sum,
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LSTM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_lstm(input_shape, seed):
    """Build standardized 2-layer LSTM (32→16) with fixed hyperparameters."""
    import tensorflow as tf
    tf.random.set_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(LSTM_UNITS[0], activation="tanh",
                             return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.LSTM(LSTM_UNITS[1], activation="tanh",
                             return_sequences=False),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_sequences(data, feature_cols, target_col, lookback):
    """Build (X, y) sequences from a DataFrame.

    X shape: (n_samples, lookback, n_features)
    y shape: (n_samples,) — next-day target value
    """
    features = data[feature_cols].values
    target = data[target_col].values

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(features[i - lookback:i])
        y.append(target[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_single_experiment(ticker_data, feature_cols, target_col, seed):
    """Train and evaluate one LSTM run. Returns (mae, dir_acc, actuals, preds).

    mae: mean absolute error on returns (e.g. 0.01 = 1%)
    dir_acc: fraction of days where predicted direction matches actual
    """
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)

    n = len(ticker_data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = ticker_data.iloc[:train_end]
    val_df = ticker_data.iloc[train_end:val_end]
    test_df = ticker_data.iloc[val_end:]

    if len(test_df) < LOOKBACK + 1 or len(train_df) < LOOKBACK + 1:
        return None, None, None, None

    # Fit scalers on train only
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    train_features = train_df[feature_cols].values
    feature_scaler.fit(train_features)

    train_target = train_df[[target_col]].values
    target_scaler.fit(train_target)

    # Transform all splits
    for split_df in [train_df, val_df, test_df]:
        scaled_features = feature_scaler.transform(split_df[feature_cols].values)
        for i, col in enumerate(feature_cols):
            split_df.loc[:, f"_s_{col}"] = scaled_features[:, i]
        split_df.loc[:, f"_s_{target_col}"] = target_scaler.transform(
            split_df[[target_col]].values
        ).flatten()

    scaled_feature_cols = [f"_s_{c}" for c in feature_cols]
    scaled_target_col = f"_s_{target_col}"

    # Build sequences — use full contiguous blocks
    # For val/test, prepend lookback rows from previous split for continuity
    train_seq = train_df
    val_seq = pd.concat([train_df.iloc[-LOOKBACK:], val_df])
    test_seq = pd.concat([val_df.iloc[-LOOKBACK:], test_df])

    X_train, y_train = prepare_sequences(train_seq, scaled_feature_cols, scaled_target_col, LOOKBACK)
    X_val, y_val = prepare_sequences(val_seq, scaled_feature_cols, scaled_target_col, LOOKBACK)
    X_test, y_test = prepare_sequences(test_seq, scaled_feature_cols, scaled_target_col, LOOKBACK)

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None, None, None

    # Train
    model = build_lstm(input_shape=(LOOKBACK, len(feature_cols)), seed=seed)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(X_val) > 0 else None,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop] if len(X_val) > 0 else [],
        verbose=0,
    )

    # Predict
    preds_scaled = model.predict(X_test, verbose=0)
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    actuals = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, preds)

    # Direction accuracy: did the model predict the right sign of the return?
    actual_dir = np.sign(actuals)
    pred_dir = np.sign(preds)
    # Only count days with actual movement (exclude ~0 returns)
    mask = actual_dir != 0
    dir_acc = np.mean(pred_dir[mask] == actual_dir[mask]) if mask.any() else 0.5

    # Clean up to free memory
    tf.keras.backend.clear_session()

    return mae, dir_acc, actuals, preds


def run_config_target(prepared_data, config_name, target_type, tickers, seeds):
    """Run all seeds for one config × one target across all tickers.

    Returns list of result dicts.
    """
    feature_cols = FEATURE_CONFIGS[config_name]
    target_col = "gap_return" if target_type == "gap" else "cc_return"

    results = []

    for ticker in tickers:
        ticker_data = prepared_data[prepared_data["ticker"] == ticker].copy()
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)

        # Drop first row per ticker (return = 0 placeholder)
        ticker_data = ticker_data.iloc[1:].reset_index(drop=True)

        if len(ticker_data) < LOOKBACK + 10:
            print(f"  {ticker}: insufficient data ({len(ticker_data)} rows), skipping")
            continue

        maes = []
        dir_accs = []

        for seed in seeds:
            mae, dir_acc, _, _ = run_single_experiment(
                ticker_data, feature_cols, target_col, seed
            )
            if mae is not None:
                maes.append(mae)
                dir_accs.append(dir_acc)

        if maes:
            results.append({
                "config": config_name,
                "target": target_type,
                "ticker": ticker,
                "mae_mean": np.mean(maes),
                "mae_std": np.std(maes),
                "dir_acc_mean": np.mean(dir_accs) * 100,
                "dir_acc_std": np.std(dir_accs) * 100,
                "n_seeds": len(maes),
                "maes": maes,
                "dir_accs": [d * 100 for d in dir_accs],
            })
            print(f"  {ticker}: MAE={np.mean(maes)*100:.3f}%, "
                  f"DirAcc={np.mean(dir_accs)*100:.1f}%±{np.std(dir_accs)*100:.1f}%")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STATISTICAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def paired_ttest(results, baseline_config="A"):
    """Paired t-test comparing each config against baseline across tickers×seeds."""
    print(f"\n{'='*70}")
    print(f"Paired t-test vs Config {baseline_config}")
    print(f"{'='*70}")

    # Group by (config, target)
    by_config_target = {}
    for r in results:
        key = (r["config"], r["target"])
        if key not in by_config_target:
            by_config_target[key] = {}
        by_config_target[key][r["ticker"]] = r["maes"]

    for target in ["gap", "cc"]:
        baseline_key = (baseline_config, target)
        if baseline_key not in by_config_target:
            continue

        print(f"\n  Target: {target}")
        baseline_data = by_config_target[baseline_key]

        for config in sorted(FEATURE_CONFIGS.keys()):
            if config == baseline_config:
                continue
            comp_key = (config, target)
            if comp_key not in by_config_target:
                continue

            comp_data = by_config_target[comp_key]

            # Collect paired observations (same ticker, same seed index)
            baseline_vals = []
            comp_vals = []
            for ticker in TICKERS:
                if ticker in baseline_data and ticker in comp_data:
                    b = baseline_data[ticker]
                    c = comp_data[ticker]
                    n = min(len(b), len(c))
                    baseline_vals.extend(b[:n])
                    comp_vals.extend(c[:n])

            if len(baseline_vals) < 3:
                continue

            t_stat, p_val = stats.ttest_rel(baseline_vals, comp_vals)
            direction = "better" if np.mean(comp_vals) < np.mean(baseline_vals) else "worse"
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

            print(f"    Config {config} vs {baseline_config}: "
                  f"t={t_stat:.3f}, p={p_val:.4f} {sig} ({direction})")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REPORTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table(results):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"RESULTS SUMMARY (predicting returns)")
    print(f"{'='*90}")

    for target in ["gap", "cc"]:
        target_results = [r for r in results if r["target"] == target]
        if not target_results:
            continue

        target_label = "GAP (→ open return)" if target == "gap" else "CC (→ close return)"
        print(f"\n  Target: {target_label}")
        print(f"  {'Config':<8} {'Ticker':<7} {'MAE (%)':>12} {'DirAcc %':>14}")
        print(f"  {'-'*45}")

        for config in sorted(FEATURE_CONFIGS.keys()):
            config_results = [r for r in target_results if r["config"] == config]
            if not config_results:
                continue

            for r in sorted(config_results, key=lambda x: x["ticker"]):
                print(f"  {r['config']:<8} {r['ticker']:<7} "
                      f"{r['mae_mean']*100:>6.3f}±{r['mae_std']*100:<5.3f} "
                      f"{r['dir_acc_mean']:>6.1f}±{r['dir_acc_std']:<5.1f}")

            # Aggregate
            avg_mae = np.mean([r["mae_mean"] for r in config_results])
            avg_dir = np.mean([r["dir_acc_mean"] for r in config_results])
            print(f"  {config:<8} {'MEAN':<7} {avg_mae*100:>6.3f}{'':>6} {avg_dir:>6.1f}")
            print(f"  {'-'*45}")


def save_results_csv(results):
    """Save results to CSV for further analysis."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "experiment_results.csv")

    rows = []
    for r in results:
        rows.append({
            "sentiment_model": r.get("sentiment_model", "finbert"),
            "config": r["config"],
            "target": r["target"],
            "ticker": r["ticker"],
            "mae_mean": f"{r['mae_mean']:.6f}",
            "mae_std": f"{r['mae_std']:.6f}",
            "dir_acc_mean": f"{r['dir_acc_mean']:.2f}",
            "dir_acc_std": f"{r['dir_acc_std']:.2f}",
            "n_seeds": r["n_seeds"],
            "features": " + ".join(FEATURE_CONFIGS[r["config"]]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LSTM Feature Comparison Experiment")
    parser.add_argument("--config", nargs="+", default=list(FEATURE_CONFIGS.keys()),
                        help="Feature configs to test (A-F)")
    parser.add_argument("--target", nargs="+", default=["gap", "cc"],
                        help="Prediction targets (gap, cc)")
    parser.add_argument("--ticker", nargs="+", default=TICKERS,
                        help="Tickers to test")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Use cached sentiment scores (skip inference)")
    parser.add_argument("--sentiment-model", nargs="+",
                        default=["finbert"],
                        choices=list(SENTIMENT_MODELS.keys()),
                        help="Sentiment model(s) to use")
    args = parser.parse_args()

    print("=" * 70)
    print("LSTM Feature Comparison Experiment")
    print(f"Configs: {args.config} | Targets: {args.target} | Tickers: {args.ticker}")
    print(f"Sentiment models: {args.sentiment_model}")
    print(f"Seeds: {SEEDS} | Lookback: {LOOKBACK} | LSTM: {LSTM_UNITS}")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading price data...")
    price_df = load_price_data()
    print(f"  {len(price_df)} price rows, {price_df['ticker'].nunique()} tickers")

    # 2. Split news into gap/cc
    print("\n[2/4] Loading and splitting news into gap/cc buckets...")
    gap_news, cc_news = load_and_split_news()

    # 3-5. Loop over sentiment models
    all_results = []

    for model_key in args.sentiment_model:
        print(f"\n{'='*70}")
        print(f"SENTIMENT MODEL: {model_key} ({SENTIMENT_MODELS[model_key]['hf_name']})")
        print(f"{'='*70}")

        # 3. Sentiment scoring
        needs_sentiment = any(c in args.config for c in ["C", "D", "E", "F"])
        if needs_sentiment:
            print("\n[3/4] Computing sentiment scores...")
            gap_scored, cc_scored = get_sentiment_articles(
                gap_news, cc_news, model_key=model_key,
                skip_sentiment=args.skip_sentiment
            )
        else:
            print("\n[3/4] Skipping sentiment (not needed for configs A/B)...")
            for col in ["finbert_score", "label", "prob_positive", "prob_negative", "prob_neutral"]:
                gap_news[col] = 0.0
                cc_news[col] = 0.0
            gap_news["label"] = "neutral"
            cc_news["label"] = "neutral"
            gap_scored, cc_scored = gap_news, cc_news

        # 4. Aggregate features
        print("\n[4/4] Aggregating features...")
        gap_data = aggregate_sentiment_features(gap_scored, price_df, "gap")
        cc_data = aggregate_sentiment_features(cc_scored, price_df, "cc")
        print(f"  Gap features: {len(gap_data)} rows")
        print(f"  CC features: {len(cc_data)} rows")

        # 5. Run experiments
        prepared = {"gap": gap_data, "cc": cc_data}

        total_runs = len(args.config) * len(args.target)
        run_num = 0

        for target in args.target:
            for config in args.config:
                run_num += 1
                print(f"\n{'─'*70}")
                print(f"[{run_num}/{total_runs}] {model_key} × Config {config} × Target {target}")
                print(f"  Features: {FEATURE_CONFIGS[config]}")
                print(f"{'─'*70}")

                results = run_config_target(
                    prepared[target], config, target, args.ticker, SEEDS
                )
                # Tag results with sentiment model
                for r in results:
                    r["sentiment_model"] = model_key
                all_results.extend(results)

    # 6. Report
    print_summary_table(all_results)
    paired_ttest(all_results)
    save_results_csv(all_results)

    print(f"\nExperiment complete. {len(all_results)} ticker-config-target combinations evaluated.")


if __name__ == "__main__":
    main()

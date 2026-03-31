"""
Compare sentiment-based return prediction approaches — all small-data models.

Tests multiple approaches × 5 sentiment models × 7 tickers × 2 targets.
No standard LSTM (proven to fail in Entry 39). Instead: small-data models
+ lightweight sequential models (GRU, TCN) that may work where LSTM failed.

All approaches use same 72/8/20 split as Config C for fair comparison.
Parallel execution across sentiment models.

Key insight from literature review:
- Papers 1,4: Their LSTMs DON'T actually beat naive — they never checked.
  Entry 39 proved this. Those are the same dead end we already discarded.
- Paper 3: Genuinely different — uses LLM to generate formulaic alpha
  factors from existing features. 10-26% MSE improvement. But needs 8 years
  of data (we have 1). We adapt the CONCEPT: create interaction/formula
  features from sentiment + price, not just raw sentiment.
- Paper 4: Learned category weights across news levels help. We don't have
  categories per sentiment model, but we can weight sentiment × volatility.

Approaches:
  1. Naive (predict 0% return)
  2. Gated Naive (predict 0 unless extreme sentiment)
  3. Sentiment × Volatility (sent × σ_20d × k)
  4. Direction Nudge (calibrated step where DirAcc > 55%)
  5. Multi-Model Consensus (4/5 models agree → predict)
  6. Extreme-Day Magnitude (|sent| → |return|, direction from momentum)
  7. Ridge with interaction features (pooled, regularized)
  8. Bayesian Shrinkage (prior=naive, update with sentiment)
  9. GRU (lightweight sequential, single layer)
  10. TCN (temporal convolutional, causal, no recurrence)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
LOOKBACK = 10
SEEDS = [42, 64, 128]  # 3 seeds for speed (sequential models are slow)

SENT_MODELS = {
    "FinBERT": os.path.join(OUT_DIR, "finbert_lstm_results", "sentiment.csv"),
    "DeBERTa": os.path.join(OUT_DIR, "deberta_v3_lstm_results", "sentiment.csv"),
    "Gemma-3-1B": os.path.join(OUT_DIR, "gemma_3_1b_lstm_results", "sentiment.csv"),
    "Qwen2.5": os.path.join(OUT_DIR, "qwen25_lstm_results", "sentiment.csv"),
    "Llama-FinSent": os.path.join(OUT_DIR, "llama_finsent_lstm_results", "sentiment.csv"),
}

# Config C LSTM results for comparison (from user's data)
CONFIG_C_MAE = {
    "gap": {
        "Gemma-3-1B": {"AAPL": 0.71, "AMZN": 0.71, "GOOGL": 0.74, "META": 0.78, "MSFT": 0.56, "NVDA": 1.22, "TSLA": 1.60},
        "Qwen2.5":    {"AAPL": 0.81, "AMZN": 0.65, "GOOGL": 0.76, "META": 0.63, "MSFT": 0.62, "NVDA": 1.21, "TSLA": 1.64},
        "FinBERT":    {"AAPL": 0.74, "AMZN": 0.66, "GOOGL": 0.74, "META": 0.81, "MSFT": 0.66, "NVDA": 1.25, "TSLA": 1.58},
        "DeBERTa":    {"AAPL": 0.73, "AMZN": 0.67, "GOOGL": 0.72, "META": 0.98, "MSFT": 0.58, "NVDA": 1.18, "TSLA": 1.64},
        "Llama-FinSent": {"AAPL": 0.69, "AMZN": 0.82, "GOOGL": 0.74, "META": 0.62, "MSFT": 0.61, "NVDA": 1.21, "TSLA": 1.85},
    },
    "cc": {
        "Qwen2.5":    {"AAPL": 1.00, "AMZN": 1.21, "GOOGL": 1.05, "META": 1.23, "MSFT": 0.83, "NVDA": 2.48, "TSLA": 2.87},
        "FinBERT":    {"AAPL": 0.94, "AMZN": 1.25, "GOOGL": 1.06, "META": 1.13, "MSFT": 0.81, "NVDA": 2.56, "TSLA": 2.98},
        "DeBERTa":    {"AAPL": 0.95, "AMZN": 1.27, "GOOGL": 1.02, "META": 1.22, "MSFT": 0.81, "NVDA": 2.43, "TSLA": 3.05},
        "Gemma-3-1B": {"AAPL": 0.98, "AMZN": 1.24, "GOOGL": 1.07, "META": 1.12, "MSFT": 0.78, "NVDA": 2.56, "TSLA": 3.11},
        "Llama-FinSent": {"AAPL": 0.97, "AMZN": 1.28, "GOOGL": 1.06, "META": 1.05, "MSFT": 0.81, "NVDA": 2.46, "TSLA": 3.71},
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_price():
    """Load price data, compute returns and derived features."""
    df = pd.read_csv(os.path.join(RAW_DIR, "price.csv"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    records = []
    for ticker in TICKERS:
        t = df[df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        t["gap_return"] = (t["open"] - t["close"].shift(1)) / t["close"].shift(1)
        t["cc_return"] = t["close"].pct_change()
        t["vol_20d"] = t["cc_return"].rolling(20).std()
        t["lag1_return"] = t["cc_return"].shift(1)
        t["lag2_return"] = t["cc_return"].shift(2)
        t["momentum_5d"] = t["cc_return"].rolling(5).sum().shift(1)
        t["range_pct"] = (t["high"] - t["low"]) / t["close"]
        records.append(t.iloc[20:])  # drop warmup

    return pd.concat(records, ignore_index=True)


def load_sentiment(model_name):
    """Load one sentiment model's daily scores."""
    path = SENT_MODELS[model_name]
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.rename(columns={"finbert_score": "sentiment"})[["date", "ticker", "sentiment"]]


def load_all_sentiments():
    """Load all 5 sentiment models, merge."""
    price = load_price()
    merged = price.copy()
    for name, path in SENT_MODELS.items():
        if os.path.exists(path):
            s = pd.read_csv(path)
            s["date"] = pd.to_datetime(s["date"])
            s = s.rename(columns={"finbert_score": f"sent_{name}"})
            merged = merged.merge(s[["date", "ticker", f"sent_{name}"]],
                                  on=["date", "ticker"], how="left")
            merged[f"sent_{name}"] = merged[f"sent_{name}"].fillna(0.0)
    return merged


def merge_sentiment(price_df, sent_df):
    merged = price_df.merge(sent_df, on=["date", "ticker"], how="left")
    merged["sentiment"] = merged["sentiment"].fillna(0.0)
    return merged


def split_ticker(data, ticker):
    td = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    n = len(td)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return td.iloc[:train_end], td.iloc[train_end:val_end], td.iloc[val_end:]


def evaluate(actuals, preds):
    mae = mean_absolute_error(actuals, preds)
    mask = actuals != 0
    dir_acc = np.mean(np.sign(preds[mask]) == np.sign(actuals[mask])) if mask.sum() > 0 else 0.5
    return mae, dir_acc


# ── Approach 1: Naive ────────────────────────────────────────────────────────

def run_naive(data, target_col, ticker):
    _, _, test = split_ticker(data, ticker)
    return evaluate(test[target_col].values, np.zeros(len(test)))


# ── Approach 2: Sentiment-Gated Naive ────────────────────────────────────────

def run_gated_naive(data, target_col, ticker):
    train, val, test = split_ticker(data, ticker)
    abs_sent = train["sentiment"].abs()
    threshold = max(abs_sent.quantile(0.80), 0.01)

    extreme = abs_sent > threshold
    step = train.loc[extreme, target_col].abs().median() if extreme.sum() > 5 else train[target_col].abs().median()

    best_damp, best_mae = 0.0, float("inf")
    for damp in np.arange(0.0, 1.55, 0.05):
        preds = np.where(val["sentiment"].abs() > threshold,
                         np.sign(val["sentiment"]) * step * damp, 0.0)
        mae = mean_absolute_error(val[target_col], preds)
        if mae < best_mae:
            best_mae, best_damp = mae, damp

    test_preds = np.where(test["sentiment"].abs() > threshold,
                          np.sign(test["sentiment"]) * step * best_damp, 0.0)
    return evaluate(test[target_col].values, test_preds)


# ── Approach 3: Sentiment × Volatility ───────────────────────────────────────

def run_sent_vol(data, target_col, ticker):
    train, val, test = split_ticker(data, ticker)
    best_k, best_mae = 0.0, float("inf")
    for k in np.arange(0.0, 2.05, 0.05):
        preds = val["sentiment"].values * val["vol_20d"].fillna(0.01).values * k
        mae = mean_absolute_error(val[target_col], preds)
        if mae < best_mae:
            best_mae, best_k = mae, k

    test_preds = test["sentiment"].values * test["vol_20d"].fillna(0.01).values * best_k
    return evaluate(test[target_col].values, test_preds)


# ── Approach 4: Direction Nudge ──────────────────────────────────────────────

def run_direction_nudge(data, target_col, ticker):
    train, val, test = split_ticker(data, ticker)
    val_dir = np.sign(val["sentiment"].values)
    val_actual = np.sign(val[target_col].values)
    mask = val[target_col].values != 0
    val_dir_acc = np.mean(val_dir[mask] == val_actual[mask]) if mask.sum() > 0 else 0.5

    if val_dir_acc <= 0.55:
        return evaluate(test[target_col].values, np.zeros(len(test)))

    sent_mask = train["sentiment"].abs() > 0.01
    step = train.loc[sent_mask, target_col].abs().median() if sent_mask.sum() > 5 else train[target_col].abs().median()

    best_damp, best_mae = 0.0, float("inf")
    for damp in np.arange(0.0, 1.55, 0.05):
        preds = np.where(val["sentiment"].abs() > 0.01,
                         np.sign(val["sentiment"]) * step * damp, 0.0)
        mae = mean_absolute_error(val[target_col], preds)
        if mae < best_mae:
            best_mae, best_damp = mae, damp

    test_preds = np.where(test["sentiment"].abs() > 0.01,
                          np.sign(test["sentiment"]) * step * best_damp, 0.0)
    return evaluate(test[target_col].values, test_preds)


# ── Approach 5: Multi-Model Consensus ────────────────────────────────────────

def run_consensus(all_data, target_col, ticker):
    _, val, test = split_ticker(all_data, ticker)
    sent_cols = [c for c in all_data.columns if c.startswith("sent_")]
    n_models = len(sent_cols)

    def predict(df, k):
        signs = np.column_stack([np.sign(df[c].values) for c in sent_cols])
        agreement = np.abs(signs.sum(axis=1)) / n_models
        avg_sent = np.column_stack([df[c].values for c in sent_cols]).mean(axis=1)
        vol = df["vol_20d"].fillna(0.01).values
        return np.where(agreement >= 0.8, avg_sent * vol * k, 0.0)

    best_k, best_mae = 0.0, float("inf")
    for k in np.arange(0.0, 2.05, 0.1):
        preds = predict(val, k)
        mae = mean_absolute_error(val[target_col], preds)
        if mae < best_mae:
            best_mae, best_k = mae, k

    return evaluate(test[target_col].values, predict(test, best_k))


# ── Approach 6: Extreme-Day Magnitude ────────────────────────────────────────

def run_extreme_magnitude(data, target_col, ticker):
    train, val, test = split_ticker(data, ticker)
    abs_sent = train["sentiment"].abs().values
    abs_ret = train[target_col].abs().values
    threshold = np.percentile(abs_sent, 80)

    extreme = abs_sent > threshold
    mag_extreme = np.median(abs_ret[extreme]) if extreme.sum() > 3 else np.median(abs_ret)

    def predict(df, scale):
        is_extreme = df["sentiment"].abs().values > threshold
        direction = np.sign(df["sentiment"].values)
        mom = df["lag1_return"].fillna(0).values
        mom_agrees = np.sign(mom) == np.sign(df["sentiment"].values)
        damping = np.where(mom_agrees, scale, scale * 0.5)
        return np.where(is_extreme, direction * mag_extreme * damping, 0.0)

    best_scale, best_mae = 0.0, float("inf")
    for scale in np.arange(0.0, 2.05, 0.05):
        mae = mean_absolute_error(val[target_col], predict(val, scale))
        if mae < best_mae:
            best_mae, best_scale = mae, scale

    return evaluate(test[target_col].values, predict(test, best_scale))


# ── Approach 7: Ridge with interaction features (per-ticker) ─────────────────

def run_ridge(data, target_col, ticker):
    train, val, test = split_ticker(data, ticker)

    def feats(df):
        f = pd.DataFrame(index=df.index)
        f["sent"] = df["sentiment"].values
        f["sent_sq"] = df["sentiment"].values ** 2
        f["abs_sent"] = df["sentiment"].abs().values
        f["vol_20d"] = df["vol_20d"].fillna(0.01).values
        f["sent_x_vol"] = f["sent"] * f["vol_20d"]
        f["abs_sent_x_vol"] = f["abs_sent"] * f["vol_20d"]
        f["lag1"] = df["lag1_return"].fillna(0).values
        f["sent_x_lag1"] = f["sent"] * f["lag1"]
        f["momentum"] = df["momentum_5d"].fillna(0).values
        f["sent_x_mom"] = f["sent"] * f["momentum"]
        f["range"] = df["range_pct"].fillna(0).values
        return f

    X_fit = pd.concat([feats(train), feats(val)], ignore_index=True)
    y_fit = np.concatenate([train[target_col].values, val[target_col].values])

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    X_test_s = scaler.transform(feats(test))

    model = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100, 1000])
    model.fit(X_fit_s, y_fit)
    preds = model.predict(X_test_s)

    return evaluate(test[target_col].values, preds)


# ── Approach 8: Bayesian Shrinkage ───────────────────────────────────────────

def run_bayesian(data, target_col, ticker):
    train, val, test = split_ticker(data, ticker)
    train_var = max(train[target_col].var(), 1e-8)
    prior_prec = 1.0 / train_var

    best_k, best_ps, best_mae = 0.0, 1.0, float("inf")
    for k in np.arange(0.0, 0.55, 0.025):
        for ps in [0.1, 0.5, 1, 2, 5, 10, 20, 50]:
            lm = val["sentiment"].values * k
            lp = val["sentiment"].abs().values * ps
            tp = prior_prec + lp
            post = (lp * lm) / np.maximum(tp, 1e-10)
            mae = mean_absolute_error(val[target_col], post)
            if mae < best_mae:
                best_mae, best_k, best_ps = mae, k, ps

    lm = test["sentiment"].values * best_k
    lp = test["sentiment"].abs().values * best_ps
    tp = prior_prec + lp
    post = (lp * lm) / np.maximum(tp, 1e-10)
    return evaluate(test[target_col].values, post)


# ── Approach 9: GRU (lightweight sequential) ─────────────────────────────────

def run_gru(data, target_col, ticker, seed=42):
    """Single-layer GRU — simpler than LSTM, fewer params, better for small data."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train, val, test = split_ticker(data, ticker)
    feature_cols = ["cc_return" if "cc" in target_col else "gap_return", "sentiment"]

    # Use cc_return as the return feature for both targets
    if "gap_return" not in train.columns and "gap" in target_col:
        return None
    actual_feat = []
    for c in feature_cols:
        if c in train.columns:
            actual_feat.append(c)
    if not actual_feat:
        return None
    feature_cols = actual_feat

    feat_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()
    feat_scaler.fit(train[feature_cols].fillna(0).values)
    tgt_scaler.fit(train[[target_col]].values)

    def scale_df(df):
        d = df.copy()
        sf = feat_scaler.transform(d[feature_cols].fillna(0).values)
        for i, c in enumerate(feature_cols):
            d[f"_s_{c}"] = sf[:, i]
        d[f"_s_tgt"] = tgt_scaler.transform(d[[target_col]].values).flatten()
        return d

    train_s = scale_df(train)
    val_s = scale_df(val)
    test_s = scale_df(test)

    s_feat = [f"_s_{c}" for c in feature_cols]

    def make_seq(d):
        X, y = [], []
        feats = d[s_feat].values
        tgt = d["_s_tgt"].values
        for i in range(LOOKBACK, len(d)):
            X.append(feats[i - LOOKBACK:i])
            y.append(tgt[i])
        return np.array(X, np.float32), np.array(y, np.float32)

    X_tr, y_tr = make_seq(train_s)
    val_seq = pd.concat([train_s.iloc[-LOOKBACK:], val_s])
    X_val, y_val = make_seq(val_seq)
    test_seq = pd.concat([val_s.iloc[-LOOKBACK:], test_s])
    X_te, y_te = make_seq(test_seq)

    if len(X_tr) == 0 or len(X_te) == 0:
        return None

    # Single-layer GRU — fewer params than 2-layer LSTM
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(LOOKBACK, len(feature_cols))),
        tf.keras.layers.GRU(16, activation="tanh"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val) if len(X_val) > 0 else None,
              epochs=50, batch_size=32, callbacks=[es] if len(X_val) > 0 else [], verbose=0)

    preds_s = model.predict(X_te, verbose=0)
    preds = tgt_scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    actuals = tgt_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    tf.keras.backend.clear_session()
    return evaluate(actuals, preds)


# ── Approach 10: TCN (Temporal Convolutional Network) ────────────────────────

def run_tcn(data, target_col, ticker, seed=42):
    """Causal 1D convolutions — no recurrence, parallelizable, good for small data."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    train, val, test = split_ticker(data, ticker)

    feature_cols = []
    for c in ["cc_return" if "cc" in target_col else "gap_return", "sentiment"]:
        if c in train.columns:
            feature_cols.append(c)
    if not feature_cols:
        return None

    feat_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()
    feat_scaler.fit(train[feature_cols].fillna(0).values)
    tgt_scaler.fit(train[[target_col]].values)

    def scale_df(df):
        d = df.copy()
        sf = feat_scaler.transform(d[feature_cols].fillna(0).values)
        for i, c in enumerate(feature_cols):
            d[f"_s_{c}"] = sf[:, i]
        d["_s_tgt"] = tgt_scaler.transform(d[[target_col]].values).flatten()
        return d

    train_s = scale_df(train)
    val_s = scale_df(val)
    test_s = scale_df(test)

    s_feat = [f"_s_{c}" for c in feature_cols]

    def make_seq(d):
        X, y = [], []
        feats = d[s_feat].values
        tgt = d["_s_tgt"].values
        for i in range(LOOKBACK, len(d)):
            X.append(feats[i - LOOKBACK:i])
            y.append(tgt[i])
        return np.array(X, np.float32), np.array(y, np.float32)

    X_tr, y_tr = make_seq(train_s)
    val_seq = pd.concat([train_s.iloc[-LOOKBACK:], val_s])
    X_val, y_val = make_seq(val_seq)
    test_seq = pd.concat([val_s.iloc[-LOOKBACK:], test_s])
    X_te, y_te = make_seq(test_seq)

    if len(X_tr) == 0 or len(X_te) == 0:
        return None

    # Causal TCN: dilated causal convolutions
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(LOOKBACK, len(feature_cols))),
        tf.keras.layers.Conv1D(16, kernel_size=3, padding="causal", dilation_rate=1, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(16, kernel_size=3, padding="causal", dilation_rate=2, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(0.001))

    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val) if len(X_val) > 0 else None,
              epochs=50, batch_size=32, callbacks=[es] if len(X_val) > 0 else [], verbose=0)

    preds_s = model.predict(X_te, verbose=0)
    preds = tgt_scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    actuals = tgt_scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()

    tf.keras.backend.clear_session()
    return evaluate(actuals, preds)


# ── Runners ──────────────────────────────────────────────────────────────────

def run_simple_approaches(sent_model_name):
    """Approaches 1-4, 6-8 for one sentiment model (all per-ticker)."""
    price = load_price()
    sent = load_sentiment(sent_model_name)
    data = merge_sentiment(price, sent)

    results = {}
    for target in ["gap", "cc"]:
        tc = f"{target}_return"
        results[target] = {}
        for ticker in TICKERS:
            r = {}
            r["Naive"] = run_naive(data, tc, ticker)
            r["Gated-Naive"] = run_gated_naive(data, tc, ticker)
            r["Sent×Vol"] = run_sent_vol(data, tc, ticker)
            r["Dir-Nudge"] = run_direction_nudge(data, tc, ticker)
            r["Extreme-Mag"] = run_extreme_magnitude(data, tc, ticker)
            r["Bayesian"] = run_bayesian(data, tc, ticker)
            r["Ridge"] = run_ridge(data, tc, ticker)
            results[target][ticker] = r
    return sent_model_name, results


def run_sequential_approach(sent_model_name):
    """Approaches 9-10 (GRU, TCN) for one sentiment model. Multiple seeds."""
    price = load_price()
    sent = load_sentiment(sent_model_name)
    data = merge_sentiment(price, sent)

    results = {}
    for target in ["gap", "cc"]:
        tc = f"{target}_return"
        results[target] = {}
        for ticker in TICKERS:
            gru_maes, gru_das = [], []
            tcn_maes, tcn_das = [], []
            for seed in SEEDS:
                g = run_gru(data, tc, ticker, seed)
                if g:
                    gru_maes.append(g[0])
                    gru_das.append(g[1])
                t = run_tcn(data, tc, ticker, seed)
                if t:
                    tcn_maes.append(t[0])
                    tcn_das.append(t[1])

            r = {}
            if gru_maes:
                r["GRU"] = (np.mean(gru_maes), np.mean(gru_das))
            if tcn_maes:
                r["TCN"] = (np.mean(tcn_maes), np.mean(tcn_das))
            results[target][ticker] = r
    return sent_model_name, results


def run_consensus_approach():
    """Approach 5 — needs all models."""
    all_data = load_all_sentiments()
    results = {}
    for target in ["gap", "cc"]:
        tc = f"{target}_return"
        results[target] = {}
        for ticker in TICKERS:
            results[target][ticker] = run_consensus(all_data, tc, ticker)
    return results


# ── Print results ────────────────────────────────────────────────────────────

def print_tables(simple_results, seq_results, consensus_results):
    available = list(simple_results.keys())
    approaches = ["Naive", "Gated-Naive", "Sent×Vol", "Dir-Nudge", "Extreme-Mag",
                   "Bayesian", "Ridge", "Consensus", "GRU", "TCN"]

    for target in ["gap", "cc"]:
        label = "GAP (overnight news → open return)" if target == "gap" else "CC (all news → close return)"
        print(f"\n{'='*95}")
        print(f"  Return MAE (%) — {label}")
        print(f"{'='*95}")

        header = f"  {'Approach':<15}"
        for t in TICKERS:
            header += f" {t:>7}"
        header += f" {'MEAN':>7}  {'Best Model':>12}"
        print(header)
        print(f"  {'-'*100}")

        for approach in approaches:
            best_per_ticker = {}
            best_model_per_ticker = {}

            for ticker in TICKERS:
                best_mae, best_model = float("inf"), ""

                if approach == "Consensus":
                    if target in consensus_results and ticker in consensus_results[target]:
                        mae, _ = consensus_results[target][ticker]
                        best_mae, best_model = mae, "all"
                elif approach in ("GRU", "TCN"):
                    for sn in available:
                        if (sn in seq_results and target in seq_results[sn] and
                            ticker in seq_results[sn][target] and
                            approach in seq_results[sn][target][ticker]):
                            mae, _ = seq_results[sn][target][ticker][approach]
                            if mae < best_mae:
                                best_mae, best_model = mae, sn
                else:
                    for sn in available:
                        if (sn in simple_results and target in simple_results[sn] and
                            ticker in simple_results[sn][target] and
                            approach in simple_results[sn][target][ticker]):
                            mae, _ = simple_results[sn][target][ticker][approach]
                            if mae < best_mae:
                                best_mae, best_model = mae, sn

                best_per_ticker[ticker] = best_mae
                best_model_per_ticker[ticker] = best_model

            if all(v == float("inf") for v in best_per_ticker.values()):
                continue

            row = f"  {approach:<15}"
            for ticker in TICKERS:
                mae_pct = best_per_ticker[ticker] * 100 if best_per_ticker[ticker] < float("inf") else 0
                row += f" {mae_pct:>6.2f}%"
            valid = [v for v in best_per_ticker.values() if v < float("inf")]
            avg = np.mean(valid) * 100 if valid else 0
            from collections import Counter
            common = Counter(best_model_per_ticker.values()).most_common(1)
            bm = common[0][0] if common else ""
            row += f" {avg:>6.2f}%  {bm:>12}"
            print(row)

        # Config C LSTM comparison
        print(f"  {'-'*100}")
        if target in CONFIG_C_MAE:
            row = f"  {'LSTM-BestC':<15}"
            for ticker in TICKERS:
                best = min(CONFIG_C_MAE[target][sn][ticker] for sn in CONFIG_C_MAE[target])
                row += f" {best:>6.2f}%"
            avgs = [min(CONFIG_C_MAE[target][sn][t] for sn in CONFIG_C_MAE[target]) for t in TICKERS]
            row += f" {np.mean(avgs):>6.2f}%  {'Config-C':>12}"
            print(row)


def main():
    available = {k: v for k, v in SENT_MODELS.items() if os.path.exists(v)}
    print(f"Running 10 approaches × {len(available)} sentiment models × 7 tickers × 2 targets")
    print(f"Models: {list(available.keys())}")
    print(f"Sequential model seeds: {SEEDS}")
    print()

    simple_results = {}
    seq_results = {}

    # Run everything in parallel
    with ProcessPoolExecutor(max_workers=min(len(available) * 2, 10)) as pool:
        # Simple approaches including Ridge (per-ticker, fast)
        simple_futures = {pool.submit(run_simple_approaches, n): ("simple", n) for n in available}
        # Sequential approaches (slower — GRU + TCN with 3 seeds)
        seq_futures = {pool.submit(run_sequential_approach, n): ("seq", n) for n in available}

        all_futures = {**simple_futures, **seq_futures}
        for future in as_completed(all_futures):
            kind, name = all_futures[future]
            try:
                if kind == "simple":
                    sn, res = future.result()
                    simple_results[sn] = res
                    print(f"  Done: {sn} (Naive, Gated, SentVol, DirNudge, ExtMag, Bayesian, Ridge)")
                elif kind == "seq":
                    sn, res = future.result()
                    seq_results[sn] = res
                    print(f"  Done: {sn} GRU+TCN ({len(SEEDS)} seeds)")
            except Exception as e:
                print(f"  ERROR: {name} {kind}: {e}")

    # Consensus (needs all models loaded)
    print("  Running consensus (all 5 models)...")
    consensus_results = run_consensus_approach()
    print("  Done: Consensus")

    # Print comparison tables
    print_tables(simple_results, seq_results, consensus_results)

    # Save CSV
    rows = []
    for sn in simple_results:
        for target in ["gap", "cc"]:
            for ticker in TICKERS:
                for approach, (mae, da) in simple_results[sn][target].get(ticker, {}).items():
                    rows.append({"sent_model": sn, "target": target, "ticker": ticker,
                                 "approach": approach, "mae": mae, "dir_acc": da})
    for sn in seq_results:
        for target in ["gap", "cc"]:
            for ticker in TICKERS:
                for approach, (mae, da) in seq_results[sn].get(target, {}).get(ticker, {}).items():
                    rows.append({"sent_model": sn, "target": target, "ticker": ticker,
                                 "approach": approach, "mae": mae, "dir_acc": da})
    for target in ["gap", "cc"]:
        for ticker in TICKERS:
            if target in consensus_results and ticker in consensus_results[target]:
                mae, da = consensus_results[target][ticker]
                rows.append({"sent_model": "consensus", "target": target, "ticker": ticker,
                             "approach": "Consensus", "mae": mae, "dir_acc": da})

    out_path = os.path.join(OUT_DIR, "lstm_feature_experiment", "sentiment_baseline_comparison.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

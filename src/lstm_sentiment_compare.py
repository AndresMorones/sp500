"""
Compare Config C (return + sentiment) across all sentiment models.

Uses pre-computed daily sentiment from old pipeline runs.
Runs all models in parallel using multiprocessing.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
LOOKBACK = 10
TRAIN_RATIO = 0.72
VAL_RATIO = 0.08
LSTM_UNITS = (32, 16)
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50
PATIENCE = 10
SEEDS = [16, 32, 42, 64, 128]

# Map model key → sentiment.csv path
MODELS = {
    "FinBERT": os.path.join(OUT_DIR, "finbert_lstm_results", "sentiment.csv"),
    "DeBERTa": os.path.join(OUT_DIR, "deberta_v3_lstm_results", "sentiment.csv"),
    "Gemma-3-1B": os.path.join(OUT_DIR, "gemma_3_1b_lstm_results", "sentiment.csv"),
    "Qwen2.5": os.path.join(OUT_DIR, "qwen25_lstm_results", "sentiment.csv"),
    "Llama-FinSent": os.path.join(OUT_DIR, "llama_finsent_lstm_results", "sentiment.csv"),
}


def load_data(sentiment_path):
    """Load price + pre-computed daily sentiment, compute returns."""
    price_df = pd.read_csv(os.path.join(RAW_DIR, "price.csv"))
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    sent_df = pd.read_csv(sentiment_path)
    sent_df["date"] = pd.to_datetime(sent_df["date"])

    # Merge sentiment into price
    merged = price_df.merge(sent_df[["date", "ticker", "finbert_score"]],
                            on=["date", "ticker"], how="left")
    merged["finbert_score"] = merged["finbert_score"].fillna(0.0)

    # Compute returns per ticker
    records = []
    for ticker in TICKERS:
        t = merged[merged["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        t["close_return"] = t["close"].pct_change().fillna(0.0)
        t["gap_return"] = (t["open"] - t["close"].shift(1)) / t["close"].shift(1)
        t["gap_return"] = t["gap_return"].fillna(0.0)
        t["cc_return"] = t["close_return"]
        t["sentiment"] = t["finbert_score"]
        records.append(t.iloc[1:])  # drop first row (no return)

    return pd.concat(records, ignore_index=True)


def build_lstm(input_shape, seed):
    import tensorflow as tf
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(LSTM_UNITS[0], activation="tanh", return_sequences=True),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.LSTM(LSTM_UNITS[1], activation="tanh", return_sequences=False),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(loss="mean_squared_error",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    return model


def run_single(ticker_data, target_col, seed):
    """Train/eval one LSTM run. Returns (mae, dir_acc)."""
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)

    feature_cols = ["close_return", "sentiment"]
    n = len(ticker_data)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = ticker_data.iloc[:train_end].copy()
    val_df = ticker_data.iloc[train_end:val_end].copy()
    test_df = ticker_data.iloc[val_end:].copy()

    if len(test_df) < LOOKBACK + 1 or len(train_df) < LOOKBACK + 1:
        return None, None

    # Fit scalers on train
    feat_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()
    feat_scaler.fit(train_df[feature_cols].values)
    tgt_scaler.fit(train_df[[target_col]].values)

    for df in [train_df, val_df, test_df]:
        sf = feat_scaler.transform(df[feature_cols].values)
        for i, c in enumerate(feature_cols):
            df[f"_s_{c}"] = sf[:, i]
        df[f"_s_{target_col}"] = tgt_scaler.transform(df[[target_col]].values).flatten()

    s_feat = [f"_s_{c}" for c in feature_cols]
    s_tgt = f"_s_{target_col}"

    def make_seq(data):
        X, y = [], []
        feats = data[s_feat].values
        tgt = data[s_tgt].values
        for i in range(LOOKBACK, len(data)):
            X.append(feats[i - LOOKBACK:i])
            y.append(tgt[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    train_seq = train_df
    val_seq = pd.concat([train_df.iloc[-LOOKBACK:], val_df])
    test_seq = pd.concat([val_df.iloc[-LOOKBACK:], test_df])

    X_train, y_train = make_seq(train_seq)
    X_val, y_val = make_seq(val_seq)
    X_test, y_test = make_seq(test_seq)

    if len(X_train) == 0 or len(X_test) == 0:
        return None, None

    model = build_lstm((LOOKBACK, len(feature_cols)), seed)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val) if len(X_val) > 0 else None,
              epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[early_stop] if len(X_val) > 0 else [],
              verbose=0)

    preds_s = model.predict(X_test, verbose=0)
    preds = tgt_scaler.inverse_transform(preds_s.reshape(-1, 1)).flatten()
    actuals = tgt_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, preds)
    mask = np.sign(actuals) != 0
    dir_acc = np.mean(np.sign(preds[mask]) == np.sign(actuals[mask])) if mask.any() else 0.5

    tf.keras.backend.clear_session()
    return mae, dir_acc


def run_model(model_name, sentiment_path):
    """Run Config C for one sentiment model across all tickers/targets/seeds."""
    import tensorflow as tf

    data = load_data(sentiment_path)
    results = []

    for target in ["gap", "cc"]:
        target_col = "gap_return" if target == "gap" else "cc_return"
        for ticker in TICKERS:
            td = data[data["ticker"] == ticker].sort_values("date").reset_index(drop=True)
            maes, daccs = [], []
            for seed in SEEDS:
                mae, da = run_single(td, target_col, seed)
                if mae is not None:
                    maes.append(mae)
                    daccs.append(da)
            if maes:
                results.append({
                    "model": model_name,
                    "target": target,
                    "ticker": ticker,
                    "mae_mean": np.mean(maes),
                    "mae_std": np.std(maes),
                    "dir_acc_mean": np.mean(daccs) * 100,
                    "dir_acc_std": np.std(daccs) * 100,
                })
    return results


def main():
    # Check which models have cached sentiment
    available = {k: v for k, v in MODELS.items() if os.path.exists(v)}
    missing = {k: v for k, v in MODELS.items() if not os.path.exists(v)}
    if missing:
        print(f"Missing sentiment caches (skipping): {list(missing.keys())}")
    print(f"Running Config C for: {list(available.keys())}")
    print(f"Seeds: {SEEDS} | Tickers: {TICKERS}")
    print()

    # Run models in parallel (each in a separate process)
    all_results = []
    with ProcessPoolExecutor(max_workers=min(len(available), 3)) as pool:
        futures = {
            pool.submit(run_model, name, path): name
            for name, path in available.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            results = future.result()
            all_results.extend(results)
            # Print per-model summary
            for target in ["gap", "cc"]:
                tr = [r for r in results if r["target"] == target]
                if tr:
                    avg_dir = np.mean([r["dir_acc_mean"] for r in tr])
                    avg_mae = np.mean([r["mae_mean"] for r in tr]) * 100
                    print(f"  {name:15s} {target}: DirAcc={avg_dir:.1f}%, MAE={avg_mae:.3f}%")

    # Final comparison table
    print(f"\n{'='*80}")
    print("CONFIG C COMPARISON — Direction Accuracy by Sentiment Model")
    print(f"{'='*80}")

    for target in ["gap", "cc"]:
        label = "GAP (→ open return)" if target == "gap" else "CC (→ close return)"
        print(f"\n  {label}")
        header = f"  {'Model':15s}"
        for t in TICKERS:
            header += f" {t:>7s}"
        header += f" {'MEAN':>7s}"
        print(header)
        print(f"  {'-'*75}")

        for model_name in available:
            tr = [r for r in all_results
                  if r["model"] == model_name and r["target"] == target]
            if not tr:
                continue
            row = f"  {model_name:15s}"
            for ticker in TICKERS:
                r = next((x for x in tr if x["ticker"] == ticker), None)
                if r:
                    row += f" {r['dir_acc_mean']:6.1f}%"
                else:
                    row += f"     - "
            avg = np.mean([r["dir_acc_mean"] for r in tr])
            row += f" {avg:6.1f}%"
            print(row)

    # Save CSV
    out_path = os.path.join(OUT_DIR, "lstm_feature_experiment", "sentiment_model_comparison.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(all_results).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

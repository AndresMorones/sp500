"""
Step 5: MLP baseline model — predicts Close price from 10-day price history.

Original: Single ticker (NDX), MinMaxScaler fitted on test (data leakage).
Adapted:  Per-ticker loop, scaler fitted on train only.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from config import (STOCK_PRICE_CSV, MLP_RESULTS_CSV, TICKERS,
                    SEQUENCE_LENGTH, SPLIT_RATIO, EPOCHS, MLP_LEARNING_RATE)


def build_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X).astype(float), np.array(y).astype(float)


def create_mlp(input_shape, learning_rate):
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Dense(units=50, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(units=30, activation="relu"),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(units=20, activation="relu"),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(units=1, activation="linear"),
    ])
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model


def run_ticker(ticker, stock_df):
    df = stock_df[stock_df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    dates = df["Date"].values
    close = df[["Close"]].values

    # Train/test split
    split_idx = int(len(close) * SPLIT_RATIO)
    train_raw, test_raw = close[:split_idx], close[split_idx:]
    test_dates = dates[split_idx + SEQUENCE_LENGTH:]

    # Normalize — fit on TRAIN only (fixes original's data leakage)
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_raw)
    test = scaler.transform(test_raw)

    # Build sequences
    X_train, y_train = build_sequences(train, SEQUENCE_LENGTH)
    X_test, y_test = build_sequences(test, SEQUENCE_LENGTH)

    # Flatten for MLP: (batch, seq_len, 1) -> (batch, seq_len)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Train
    model = create_mlp(input_shape=(X_train.shape[1],), learning_rate=MLP_LEARNING_RATE)
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0)

    # Predict and inverse-transform
    preds = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, preds)
    mape = mean_absolute_percentage_error(actuals, preds)

    # Build results dataframe
    results = pd.DataFrame({
        "date": test_dates[:len(actuals)],
        "ticker": ticker,
        "actual": actuals,
        "predicted": preds,
    })

    return results, mae, mape


def main():
    import os
    stock_df = pd.read_csv(STOCK_PRICE_CSV)

    # Checkpoint: resume from existing output — skip already-completed tickers
    completed = set()
    saved_rows = []
    if os.path.exists(MLP_RESULTS_CSV):
        existing = pd.read_csv(MLP_RESULTS_CSV)
        completed = set(existing["ticker"].unique())
        saved_rows = [existing]
        if completed >= set(TICKERS):
            print("MLP: all tickers already complete — skipping.")
            return
        print(f"MLP: resuming, already done: {sorted(completed)}")

    print("=" * 60)
    print("MLP Model — Per-Ticker Results")
    print("=" * 60)

    for ticker in TICKERS:
        if ticker in completed:
            print(f"{ticker}: skipped (checkpoint)")
            continue
        results, mae, mape = run_ticker(ticker, stock_df)
        saved_rows.append(results)
        # Save immediately after each ticker — crash-safe
        pd.concat(saved_rows, ignore_index=True).to_csv(MLP_RESULTS_CSV, index=False)
        print(f"{ticker}: MAE={mae:.2f}, MAPE={mape:.4f} ({mape*100:.2f}%), Acc={1-mape:.4f}")

    combined = pd.read_csv(MLP_RESULTS_CSV)
    avg_mae = combined.groupby("ticker").apply(
        lambda g: mean_absolute_error(g["actual"], g["predicted"])).mean()
    avg_mape = combined.groupby("ticker").apply(
        lambda g: mean_absolute_percentage_error(g["actual"], g["predicted"])).mean()
    print(f"\nAggregate: MAE={avg_mae:.2f}, MAPE={avg_mape:.4f} ({avg_mape*100:.2f}%)")
    print(f"Results saved to {MLP_RESULTS_CSV}")


if __name__ == "__main__":
    main()

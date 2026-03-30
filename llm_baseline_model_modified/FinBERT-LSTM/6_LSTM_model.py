"""
Step 6: LSTM baseline model — predicts Close price from 10-day price history.

Original: Single ticker (NDX), MinMaxScaler fitted on test (data leakage).
Adapted:  Per-ticker loop, scaler fitted on train only.
Architecture preserved: 3-layer LSTM (50→30→20) with dropout.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from config import (STOCK_PRICE_CSV, LSTM_RESULTS_CSV, TICKERS,
                    SEQUENCE_LENGTH, SPLIT_RATIO, EPOCHS, LSTM_LEARNING_RATE)


def build_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X).astype(float), np.array(y).astype(float)


def create_lstm(input_shape, learning_rate):
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(units=50, activation="tanh", return_sequences=True),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.LSTM(units=30, activation="tanh", return_sequences=True),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.LSTM(units=20, activation="tanh", return_sequences=False),
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

    # LSTM expects (batch, timesteps, features=1) — already correct shape from build_sequences

    # Train
    model = create_lstm(input_shape=(X_train.shape[1], 1), learning_rate=LSTM_LEARNING_RATE)
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0)

    # Predict and inverse-transform
    preds = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae = mean_absolute_error(actuals, preds)
    mape = mean_absolute_percentage_error(actuals, preds)

    results = pd.DataFrame({
        "date": test_dates[:len(actuals)],
        "ticker": ticker,
        "actual": actuals,
        "predicted": preds,
    })

    return results, mae, mape


def main():
    stock_df = pd.read_csv(STOCK_PRICE_CSV)
    all_results = []

    print("=" * 60)
    print("LSTM Model — Per-Ticker Results")
    print("=" * 60)

    for ticker in TICKERS:
        results, mae, mape = run_ticker(ticker, stock_df)
        all_results.append(results)
        print(f"{ticker}: MAE={mae:.2f}, MAPE={mape:.4f} ({mape*100:.2f}%), Acc={1-mape:.4f}")

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(LSTM_RESULTS_CSV, index=False)

    avg_mae = combined.groupby("ticker").apply(
        lambda g: mean_absolute_error(g["actual"], g["predicted"])).mean()
    avg_mape = combined.groupby("ticker").apply(
        lambda g: mean_absolute_percentage_error(g["actual"], g["predicted"])).mean()
    print(f"\nAggregate: MAE={avg_mae:.2f}, MAPE={avg_mape:.4f} ({avg_mape*100:.2f}%)")
    print(f"Results saved to {LSTM_RESULTS_CSV}")


if __name__ == "__main__":
    main()

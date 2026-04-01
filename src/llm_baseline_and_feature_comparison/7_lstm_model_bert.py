"""
Step 7: FinBERT-LSTM model — predicts Close price using 10-day price history
plus current-day FinBERT sentiment score.

Original: Single ticker (NDX), appended sentiment as 11th timestep,
          MinMaxScaler fitted on test (data leakage).
Adapted:  Per-ticker loop, scaler fitted on train only, sentiment merged
          per (ticker, date) with missing days filled as neutral (0.0).

Architecture preserved: 3-layer LSTM (70→30→10), input shape (11, 1).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from config import (STOCK_PRICE_CSV, SENTIMENT_CSV, BERT_LSTM_RESULTS_CSV, TICKERS,
                    SEQUENCE_LENGTH, SPLIT_RATIO, EPOCHS, LSTM_LEARNING_RATE)


def create_bert_lstm(input_shape, learning_rate):
    tf.random.set_seed(1234)
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.LSTM(units=70, activation="tanh", return_sequences=True),
        tf.keras.layers.LSTM(units=30, activation="tanh", return_sequences=True),
        tf.keras.layers.LSTM(units=10, activation="tanh", return_sequences=False),
        tf.keras.layers.Dense(units=1, activation="linear"),
    ])
    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    return model


def run_ticker(ticker, stock_df, sentiment_df):
    # Filter and sort data for this ticker
    sdf = stock_df[stock_df["ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    sent = sentiment_df[sentiment_df["ticker"] == ticker].copy()

    # Merge sentiment into stock data (left join — some trading days have no news)
    sdf["date_key"] = sdf["Date"].astype(str).str[:10]
    sent["date_key"] = sent["date"].astype(str).str[:10]
    merged = sdf.merge(sent[["date_key", "finbert_score"]], on="date_key", how="left")
    merged["finbert_score"] = merged["finbert_score"].fillna(0.0)

    dates = merged["Date"].values
    close = merged[["Close"]].values
    sentiment_scores = merged["finbert_score"].values

    # Train/test split
    split_idx = int(len(close) * SPLIT_RATIO)
    train_raw, test_raw = close[:split_idx], close[split_idx:]
    train_sent, test_sent = sentiment_scores[:split_idx], sentiment_scores[split_idx:]
    test_dates = dates[split_idx + SEQUENCE_LENGTH:]

    # Normalize prices — fit on TRAIN only
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train_raw).flatten()
    test = scaler.transform(test_raw).flatten()

    # Build sequences with sentiment appended (original's approach: 10 prices + 1 sentiment = 11 steps)
    X_train, y_train = [], []
    for i in range(len(train) - SEQUENCE_LENGTH):
        seq = list(train[i:i + SEQUENCE_LENGTH])
        seq.append(train_sent[SEQUENCE_LENGTH + i])  # sentiment for the prediction day
        X_train.append(seq)
        y_train.append(train[SEQUENCE_LENGTH + i])

    X_test, y_test = [], []
    for i in range(len(test) - SEQUENCE_LENGTH):
        seq = list(test[i:i + SEQUENCE_LENGTH])
        seq.append(test_sent[SEQUENCE_LENGTH + i])
        X_test.append(seq)
        y_test.append(test[SEQUENCE_LENGTH + i])

    X_train = np.array(X_train).astype(float).reshape(-1, SEQUENCE_LENGTH + 1, 1)
    y_train = np.array(y_train).astype(float)
    X_test = np.array(X_test).astype(float).reshape(-1, SEQUENCE_LENGTH + 1, 1)
    y_test = np.array(y_test).astype(float)

    # Train
    model = create_bert_lstm(
        input_shape=(SEQUENCE_LENGTH + 1, 1),
        learning_rate=LSTM_LEARNING_RATE,
    )
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
    import os
    stock_df = pd.read_csv(STOCK_PRICE_CSV)
    sentiment_df = pd.read_csv(SENTIMENT_CSV)

    # Checkpoint: resume from existing output — skip already-completed tickers
    completed = set()
    saved_rows = []
    if os.path.exists(BERT_LSTM_RESULTS_CSV):
        existing = pd.read_csv(BERT_LSTM_RESULTS_CSV)
        completed = set(existing["ticker"].unique())
        saved_rows = [existing]
        if completed >= set(TICKERS):
            print("FinBERT-LSTM: all tickers already complete — skipping.")
            return
        print(f"FinBERT-LSTM: resuming, already done: {sorted(completed)}")

    print("=" * 60)
    print("FinBERT-LSTM Model — Per-Ticker Results")
    print("=" * 60)

    for ticker in TICKERS:
        if ticker in completed:
            print(f"{ticker}: skipped (checkpoint)")
            continue
        results, mae, mape = run_ticker(ticker, stock_df, sentiment_df)
        saved_rows.append(results)
        # Save immediately after each ticker — crash-safe
        pd.concat(saved_rows, ignore_index=True).to_csv(BERT_LSTM_RESULTS_CSV, index=False)
        print(f"{ticker}: MAE={mae:.2f}, MAPE={mape:.4f} ({mape*100:.2f}%), Acc={1-mape:.4f}")

    combined = pd.read_csv(BERT_LSTM_RESULTS_CSV)
    avg_mae = combined.groupby("ticker").apply(
        lambda g: mean_absolute_error(g["actual"], g["predicted"])).mean()
    avg_mape = combined.groupby("ticker").apply(
        lambda g: mean_absolute_percentage_error(g["actual"], g["predicted"])).mean()
    print(f"\nAggregate: MAE={avg_mae:.2f}, MAPE={avg_mape:.4f} ({avg_mape*100:.2f}%)")
    print(f"Results saved to {BERT_LSTM_RESULTS_CSV}")


if __name__ == "__main__":
    main()

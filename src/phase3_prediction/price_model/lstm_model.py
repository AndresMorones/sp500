"""Price Model — LSTM predicting returns.

Architecture: LSTM(32) → Dropout(0.3) → LSTM(16) → Dropout(0.3) → Dense(1)
Input: 10-day OHLCV sequences. Predicts return, converts to price.
Per-ticker models. Runs 5 seeds, reports mean ± std.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from config import (
    TICKERS, TARGETS, SEEDS, RESULTS_DIR,
    LSTM_HIDDEN, LSTM_DROPOUT, LSTM_LR, LSTM_BATCH,
    LSTM_EPOCHS, LSTM_PATIENCE,
)
from data_loader import (
    load_price_data, load_sp500, build_price_series,
    make_price_sequences, split_data, scale_splits,
    compute_metrics,
)


def build_lstm(input_shape, seed):
    """Build the standardized LSTM model."""
    import tensorflow as tf
    tf.random.set_seed(seed)

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(LSTM_HIDDEN[0], return_sequences=True,
                             input_shape=input_shape),
        tf.keras.layers.Dropout(LSTM_DROPOUT),
        tf.keras.layers.LSTM(LSTM_HIDDEN[1]),
        tf.keras.layers.Dropout(LSTM_DROPOUT),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LSTM_LR),
        loss="mse",
    )
    return model


def run_lstm_single_seed(ticker, target, series, seed):
    """Train and evaluate LSTM for one ticker/target/seed."""
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X, y_price, dates, prev_closes = make_price_sequences(series, target)
    n = len(y_price)
    train_end, val_end = split_data(n)

    # Convert target from price to return
    y_return = (y_price - prev_closes) / prev_closes

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val = y_return[:train_end], y_return[train_end:val_end]
    y_test_price = y_price[val_end:]
    test_dates = dates[val_end:]
    test_prev_closes = prev_closes[val_end:]

    # Scale features (OHLCV sequences)
    X_train, X_val, X_test, _ = scale_splits(X_train, X_val, X_test)

    # Returns are already small-scale, but scale for consistency
    from sklearn.preprocessing import MinMaxScaler as MMS
    y_sc = MMS()
    y_train_sc = y_sc.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_sc = y_sc.transform(y_val.reshape(-1, 1)).ravel()

    # Build and train
    model = build_lstm((X_train.shape[1], X_train.shape[2]), seed)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=LSTM_PATIENCE, restore_best_weights=True
    )

    model.fit(
        X_train, y_train_sc,
        validation_data=(X_val, y_val_sc),
        epochs=LSTM_EPOCHS,
        batch_size=LSTM_BATCH,
        callbacks=[early_stop],
        verbose=0,
    )

    # Predict returns, inverse scale, convert to prices
    preds_sc = model.predict(X_test, verbose=0).ravel()
    pred_returns = y_sc.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
    pred_prices = test_prev_closes * (1 + pred_returns)

    metrics = compute_metrics(y_test_price, pred_prices)
    return {
        **metrics,
        "dates": test_dates,
        "actuals": y_test_price.tolist(),
        "preds": pred_prices.tolist(),
        "prev_closes": test_prev_closes.tolist(),
    }


def run_lstm(ticker, target, series):
    """Run LSTM across all seeds, aggregate results."""
    seed_results = []
    for seed in SEEDS:
        r = run_lstm_single_seed(ticker, target, series, seed)
        seed_results.append(r)

    # Aggregate metrics
    maes = [r["MAE"] for r in seed_results]
    mapes = [r["MAPE"] for r in seed_results]
    rmses = [r["RMSE"] for r in seed_results]

    # Use median seed's predictions as representative
    median_idx = int(np.argmin([abs(m - np.median(mapes)) for m in mapes]))

    return {
        "ticker": ticker,
        "target": target,
        "model": "LSTM",
        "MAE": float(np.mean(maes)),
        "MAE_std": float(np.std(maes)),
        "MAPE": float(np.mean(mapes)),
        "MAPE_std": float(np.std(mapes)),
        "RMSE": float(np.mean(rmses)),
        "RMSE_std": float(np.std(rmses)),
        "dates": seed_results[median_idx]["dates"],
        "actuals": seed_results[median_idx]["actuals"],
        "preds": seed_results[median_idx]["preds"],
        "all_mapes": mapes,
    }


def main():
    print("=" * 60)
    print("  Price Model — LSTM (predict return → convert to price)")
    print("=" * 60)

    price_data = load_price_data()
    sp500 = load_sp500()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for target in TARGETS:
        print(f"\n--- Target: {target} (predict {'open' if target == 'gap' else 'close'}) ---")
        print(f"{'Ticker':<8} {'MAPE (%)':>14} {'MAE ($)':>14} {'RMSE ($)':>14}")
        print("-" * 54)

        for ticker in ["GOOGL"]:
            series = build_price_series(ticker, price_data, sp500)
            print(f"  Training {ticker}...", end=" ", flush=True)
            result = run_lstm(ticker, target, series)
            all_results.append(result)
            print(f"\r{ticker:<8} {result['MAPE']:>6.3f}±{result['MAPE_std']:.3f}%  "
                  f"{result['MAE']:>6.3f}±{result['MAE_std']:.3f}  "
                  f"{result['RMSE']:>6.3f}±{result['RMSE_std']:.3f}")

        target_results = [r for r in all_results if r["target"] == target]
        avg_mape = np.mean([r["MAPE"] for r in target_results])
        avg_mae = np.mean([r["MAE"] for r in target_results])
        print(f"{'Avg':<8} {avg_mape:>6.3f}%          {avg_mae:>6.3f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "price_lstm_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "target", "model", "MAE", "MAE_std", "MAPE", "MAPE_std", "RMSE", "RMSE_std"])
        for r in all_results:
            w.writerow([r["ticker"], r["target"], r["model"],
                        r["MAE"], r["MAE_std"], r["MAPE"], r["MAPE_std"],
                        r["RMSE"], r["RMSE_std"]])

    pred_path = os.path.join(RESULTS_DIR, "price_lstm_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted"])
        for r in all_results:
            for d, a, p in zip(r["dates"], r["actuals"], r["preds"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}"])

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

"""Price Model — Ridge regression predicting returns.

Per-ticker RidgeCV on flattened OHLCV lookback features.
Predicts return (not raw price), converts to price for evaluation.
72/8/20 chronological split.
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from config import TICKERS, TARGETS, RIDGE_ALPHAS, RESULTS_DIR
from data_loader import (
    load_price_data, load_sp500, build_price_series,
    extract_price_features, make_flat_features,
    compute_metrics, split_data,
)


def run_ridge(ticker, target, series):
    """Train and evaluate Ridge for a single ticker/target."""
    data = make_flat_features(series, target, extract_price_features)
    if data is None:
        return None

    X, y_price = data["X"], data["y"]
    dates = data["dates"]
    prev_closes = data["prev_closes"]
    n = len(y_price)
    train_end, val_end = split_data(n)

    # Convert target from price to return
    y_return = (y_price - prev_closes) / prev_closes

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:train_end])
    X_val = scaler.transform(X[train_end:val_end]) if val_end > train_end else np.array([])
    X_test = scaler.transform(X[val_end:])

    y_train = y_return[:train_end]
    y_test_price = y_price[val_end:]
    test_dates = dates[val_end:]
    test_prev_closes = prev_closes[val_end:]

    # Train RidgeCV on returns (uses train + val for alpha selection)
    X_fit = np.vstack([X_train, scaler.transform(X[train_end:val_end])]) if val_end > train_end else X_train
    y_fit = np.concatenate([y_train, y_return[train_end:val_end]]) if val_end > train_end else y_train

    model = RidgeCV(alphas=RIDGE_ALPHAS)
    model.fit(X_fit, y_fit)

    # Predict returns, convert to prices
    pred_returns = model.predict(X_test)
    pred_prices = test_prev_closes * (1 + pred_returns)

    metrics = compute_metrics(y_test_price, pred_prices)
    return {
        "ticker": ticker,
        "target": target,
        "model": "Ridge",
        "alpha": float(model.alpha_),
        **metrics,
        "dates": test_dates,
        "actuals": y_test_price.tolist(),
        "preds": pred_prices.tolist(),
    }


def main():
    print("=" * 60)
    print("  Price Model — Ridge (predict return → convert to price)")
    print("=" * 60)

    price_data = load_price_data()
    sp500 = load_sp500()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for target in TARGETS:
        print(f"\n--- Target: {target} (predict {'open' if target == 'gap' else 'close'}) ---")
        print(f"{'Ticker':<8} {'MAE ($)':>10} {'MAPE (%)':>10} {'RMSE ($)':>10} {'Alpha':>8}")
        print("-" * 50)

        for ticker in TICKERS:
            series = build_price_series(ticker, price_data, sp500)
            result = run_ridge(ticker, target, series)
            if result is None:
                print(f"{ticker:<8} insufficient data")
                continue
            all_results.append(result)
            print(f"{ticker:<8} {result['MAE']:>10.3f} {result['MAPE']:>9.3f}% {result['RMSE']:>10.3f} {result['alpha']:>8.2f}")

        target_results = [r for r in all_results if r["target"] == target]
        if target_results:
            avg_mape = np.mean([r["MAPE"] for r in target_results])
            avg_mae = np.mean([r["MAE"] for r in target_results])
            print(f"{'Avg':<8} {avg_mae:>10.3f} {avg_mape:>9.3f}%")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "price_ridge_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "target", "model", "MAE", "MAPE", "RMSE", "alpha"])
        for r in all_results:
            w.writerow([r["ticker"], r["target"], r["model"], r["MAE"], r["MAPE"], r["RMSE"], r.get("alpha", "")])

    pred_path = os.path.join(RESULTS_DIR, "price_ridge_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted"])
        for r in all_results:
            for d, a, p in zip(r["dates"], r["actuals"], r["preds"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}"])

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

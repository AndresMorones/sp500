"""Price Model — Naive baseline: predict yesterday's close.

Critical baseline per open-questions Q16. Any model that cannot beat this
adds no value regardless of absolute MAPE.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
from config import TICKERS, TARGETS, LOOKBACK, RESULTS_DIR
from data_loader import (
    load_price_data, load_sp500, build_price_series,
    make_price_sequences, compute_metrics, split_data,
)


def run_naive(ticker, target, series):
    """Run naive baseline for a single ticker/target."""
    X, y, dates, prev_closes = make_price_sequences(series, target)
    n = len(y)
    _, val_end = split_data(n)

    # Test set
    y_test = y[val_end:]
    naive_preds = prev_closes[val_end:]
    test_dates = dates[val_end:]

    metrics = compute_metrics(y_test, naive_preds)
    return {
        "ticker": ticker,
        "target": target,
        "model": "Naive",
        **metrics,
        "dates": test_dates,
        "actuals": y_test.tolist(),
        "preds": naive_preds.tolist(),
    }


def main():
    print("=" * 60)
    print("  Price Model — Naive Baseline (yesterday's close)")
    print("=" * 60)

    price_data = load_price_data()
    sp500 = load_sp500()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for target in TARGETS:
        print(f"\n--- Target: {target} (predict {'open' if target == 'gap' else 'close'}) ---")
        print(f"{'Ticker':<8} {'MAE ($)':>10} {'MAPE (%)':>10} {'RMSE ($)':>10} {'N_test':>8}")
        print("-" * 50)

        for ticker in TICKERS:
            series = build_price_series(ticker, price_data, sp500)
            result = run_naive(ticker, target, series)
            all_results.append(result)
            print(f"{ticker:<8} {result['MAE']:>10.3f} {result['MAPE']:>9.3f}% {result['RMSE']:>10.3f} {len(result['dates']):>8}")

        # Average
        target_results = [r for r in all_results if r["target"] == target]
        avg_mape = np.mean([r["MAPE"] for r in target_results])
        avg_mae = np.mean([r["MAE"] for r in target_results])
        print(f"{'Avg':<8} {avg_mae:>10.3f} {avg_mape:>9.3f}%")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "price_naive_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "target", "model", "MAE", "MAPE", "RMSE"])
        for r in all_results:
            w.writerow([r["ticker"], r["target"], r["model"], r["MAE"], r["MAPE"], r["RMSE"]])

    # Save predictions
    pred_path = os.path.join(RESULTS_DIR, "price_naive_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted"])
        for r in all_results:
            for d, a, p in zip(r["dates"], r["actuals"], r["preds"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}"])

    print(f"\nResults saved to {out_path}")
    print(f"Predictions saved to {pred_path}")


if __name__ == "__main__":
    main()

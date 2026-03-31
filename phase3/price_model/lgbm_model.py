"""Price Model — LightGBM predicting returns.

Per-ticker LightGBM on flattened OHLCV lookback features.
Predicts return (not raw price), converts to price for evaluation.
Conservative hyperparameters for small dataset. Uses val set for early stopping.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import lightgbm as lgb

from config import TICKERS, TARGETS, LGBM_PARAMS, LGBM_ROUNDS, LGBM_EARLY_STOP, RESULTS_DIR
from data_loader import (
    load_price_data, load_sp500, build_price_series,
    extract_price_features, make_flat_features,
    compute_metrics, split_data,
)


def run_lgbm(ticker, target, series):
    """Train and evaluate LightGBM for a single ticker/target."""
    data = make_flat_features(series, target, extract_price_features)
    if data is None:
        return None

    X, y_price = data["X"], data["y"]
    dates = data["dates"]
    prev_closes = data["prev_closes"]
    columns = data["columns"]
    n = len(y_price)
    train_end, val_end = split_data(n)

    # Convert target from price to return
    y_return = (y_price - prev_closes) / prev_closes

    X_train, y_train = X[:train_end], y_return[:train_end]
    X_val, y_val = X[train_end:val_end], y_return[train_end:val_end]
    X_test = X[val_end:]
    y_test_price = y_price[val_end:]
    test_dates = dates[val_end:]
    test_prev_closes = prev_closes[val_end:]

    # Train on returns
    dtrain = lgb.Dataset(X_train, y_train, feature_name=columns)
    dval = lgb.Dataset(X_val, y_val, feature_name=columns, reference=dtrain)

    callbacks = [lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)]
    model = lgb.train(
        LGBM_PARAMS,
        dtrain,
        num_boost_round=LGBM_ROUNDS,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    # Predict returns, convert to prices
    pred_returns = model.predict(X_test)
    pred_prices = test_prev_closes * (1 + pred_returns)

    metrics = compute_metrics(y_test_price, pred_prices)

    # Feature importance (top 10)
    imp = dict(zip(columns, model.feature_importance(importance_type="gain")))
    top_feats = sorted(imp.items(), key=lambda x: -x[1])[:10]

    return {
        "ticker": ticker,
        "target": target,
        "model": "LightGBM",
        "best_iter": model.best_iteration,
        "top_features": top_feats,
        **metrics,
        "dates": test_dates,
        "actuals": y_test_price.tolist(),
        "preds": pred_prices.tolist(),
    }


def main():
    print("=" * 60)
    print("  Price Model — LightGBM (predict return → convert to price)")
    print("=" * 60)

    price_data = load_price_data()
    sp500 = load_sp500()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for target in TARGETS:
        print(f"\n--- Target: {target} (predict {'open' if target == 'gap' else 'close'}) ---")
        print(f"{'Ticker':<8} {'MAE ($)':>10} {'MAPE (%)':>10} {'RMSE ($)':>10} {'Iters':>8}")
        print("-" * 50)

        for ticker in TICKERS:
            series = build_price_series(ticker, price_data, sp500)
            result = run_lgbm(ticker, target, series)
            if result is None:
                print(f"{ticker:<8} insufficient data")
                continue
            all_results.append(result)
            print(f"{ticker:<8} {result['MAE']:>10.3f} {result['MAPE']:>9.3f}% {result['RMSE']:>10.3f} {result['best_iter']:>8}")

        target_results = [r for r in all_results if r["target"] == target]
        if target_results:
            avg_mape = np.mean([r["MAPE"] for r in target_results])
            avg_mae = np.mean([r["MAE"] for r in target_results])
            print(f"{'Avg':<8} {avg_mae:>10.3f} {avg_mape:>9.3f}%")

    # Print feature importance for last target
    for target in TARGETS:
        target_results = [r for r in all_results if r["target"] == target and "top_features" in r]
        if target_results:
            print(f"\nTop features ({target}):")
            for r in target_results[:1]:  # show first ticker's
                for feat, imp in r["top_features"][:5]:
                    print(f"  {feat:<30} {imp:>10.1f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "price_lgbm_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "target", "model", "MAE", "MAPE", "RMSE"])
        for r in all_results:
            w.writerow([r["ticker"], r["target"], r["model"], r["MAE"], r["MAPE"], r["RMSE"]])

    pred_path = os.path.join(RESULTS_DIR, "price_lgbm_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted"])
        for r in all_results:
            for d, a, p in zip(r["dates"], r["actuals"], r["preds"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}"])

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

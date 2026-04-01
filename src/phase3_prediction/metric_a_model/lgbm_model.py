"""Metric A Model — LightGBM.

Predicts A score from news category features, converts to price via
inverse market model: A → zi → return → price.
GOOGL only (only ticker with news_phase2 data).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import lightgbm as lgb

from config import (
    METRIC_A_TICKERS, TARGETS, LGBM_PARAMS, LGBM_ROUNDS,
    LGBM_EARLY_STOP, LOOKBACK, RESULTS_DIR,
)
from data_loader import (
    load_price_data, load_sp500, load_scores, load_news_phase2,
    get_news_cat_columns, build_metric_a_series,
    extract_news_features, make_flat_features,
    a_to_price, compute_sp_stats, split_data,
    compute_metrics, compute_range_metrics,
)


def run_lgbm(ticker, target, series, cat_cols):
    """Train LightGBM: news features → A score, convert to price."""
    data = make_flat_features(
        series, target, extract_news_features, cat_cols=cat_cols
    )
    if data is None:
        return None

    X, y_a = data["X"], data["y"]  # y_a = A_gap or A_cc
    dates = data["dates"]
    prev_closes = data["prev_closes"]
    has_news = data.get("has_news", np.zeros(len(y_a), dtype=bool))
    columns = data["columns"]
    n = len(y_a)
    train_end, val_end = split_data(n)

    X_train, y_train = X[:train_end], y_a[:train_end]
    X_val, y_val = X[train_end:val_end], y_a[train_end:val_end]
    X_test = X[val_end:]
    y_test_a = y_a[val_end:]
    test_dates = dates[val_end:]
    test_prev_closes = prev_closes[val_end:]
    test_has_news = has_news[val_end:]

    # Train on A prediction
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

    # Predict A scores
    pred_a = model.predict(X_test)

    # A prediction metrics
    a_mae = float(np.mean(np.abs(y_test_a - pred_a)))

    # Get market model parameters for test days
    min_start = LOOKBACK + 1
    test_indices = list(range(val_end + min_start, val_end + min_start + len(test_dates)))

    if target == "gap":
        t_alphas = np.array([series[i]["alpha_gap"] for i in test_indices])
        t_betas = np.array([series[i]["beta_gap"] for i in test_indices])
        t_s0s = np.array([series[i]["s0_gap"] for i in test_indices])
        actual_prices = np.array([series[i]["open"] for i in test_indices])
    else:
        t_alphas = np.array([series[i]["alpha_cc"] for i in test_indices])
        t_betas = np.array([series[i]["beta_cc"] for i in test_indices])
        t_s0s = np.array([series[i]["s0_cc"] for i in test_indices])
        actual_prices = np.array([series[i]["close"] for i in test_indices])

    # S&P stats from training data
    sp_stats = compute_sp_stats(series, train_end + min_start)
    sp_avg = sp_stats["sp_gap_avg"] if target == "gap" else sp_stats["sp_cc_avg"]
    sp_std = sp_stats["sp_gap_std"] if target == "gap" else sp_stats["sp_cc_std"]

    # Convert predicted A → price
    pred_prices, _ = a_to_price(pred_a, t_alphas, t_betas, t_s0s, sp_avg, test_prev_closes)

    point_metrics = compute_metrics(actual_prices, pred_prices)

    # Market sensitivity
    sensitivity = {}
    for label, sp_assumption in [
        ("market_-2σ", sp_avg - 2 * sp_std),
        ("market_-1σ", sp_avg - 1 * sp_std),
        ("baseline", sp_avg),
        ("market_+1σ", sp_avg + 1 * sp_std),
        ("market_+2σ", sp_avg + 2 * sp_std),
    ]:
        prices, _ = a_to_price(pred_a, t_alphas, t_betas, t_s0s, sp_assumption, test_prev_closes)
        sensitivity[label] = {
            "sp_return": sp_assumption,
            "avg_price": float(np.mean(prices)),
            "mape": float(np.mean(np.abs(actual_prices - prices) / actual_prices) * 100),
        }

    # Range from market uncertainty
    price_at_plus1 = a_to_price(pred_a, t_alphas, t_betas, t_s0s, sp_avg + sp_std, test_prev_closes)[0]
    half_widths = np.abs(price_at_plus1 - pred_prices)

    range_metrics = compute_range_metrics(actual_prices, pred_prices, half_widths)

    # Conditional coverage
    cond = {}
    for k_str, rm in range_metrics.items():
        k = float(k_str.replace("σ", ""))
        hw_k = half_widths * k
        lower = pred_prices - hw_k
        upper = pred_prices + hw_k
        if test_has_news.sum() > 0:
            in_range = ((actual_prices[test_has_news] >= lower[test_has_news]) &
                        (actual_prices[test_has_news] <= upper[test_has_news]))
            cond[f"{k_str}_news_coverage"] = float(np.mean(in_range))
        if (~test_has_news).sum() > 0:
            in_range = ((actual_prices[~test_has_news] >= lower[~test_has_news]) &
                        (actual_prices[~test_has_news] <= upper[~test_has_news]))
            cond[f"{k_str}_no_news_coverage"] = float(np.mean(in_range))

    # Feature importance
    imp = dict(zip(columns, model.feature_importance(importance_type="gain")))
    top_feats = sorted(imp.items(), key=lambda x: -x[1])[:10]

    return {
        "ticker": ticker,
        "target": target,
        "model": "LightGBM",
        "best_iter": model.best_iteration,
        **point_metrics,
        "A_MAE": a_mae,
        "range": range_metrics,
        "conditional": cond,
        "sensitivity": sensitivity,
        "top_features": top_feats,
        "n_test": len(actual_prices),
        "n_news_days": int(test_has_news.sum()),
        "dates": test_dates,
        "actuals": actual_prices.tolist(),
        "preds": pred_prices.tolist(),
        "half_widths": half_widths.tolist(),
    }


def main():
    print("=" * 60)
    print("  Metric A Model — LightGBM (news → A → price)")
    print("=" * 60)

    price_data = load_price_data()
    sp500 = load_sp500()
    scores = load_scores()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for target in TARGETS:
        print(f"\n--- Target: {target} (predict {'open' if target == 'gap' else 'close'}) ---")

        for ticker in METRIC_A_TICKERS:
            cat_cols = get_news_cat_columns(ticker)
            news = load_news_phase2(ticker)
            series = build_metric_a_series(ticker, price_data, sp500, scores, news)
            result = run_lgbm(ticker, target, series, cat_cols)
            if result is None:
                print(f"  {ticker}: insufficient data")
                continue

            all_results.append(result)
            print(f"\n  {ticker} — {result['n_test']} test days ({result['n_news_days']} with news)")
            print(f"  A prediction MAE: {result['A_MAE']:.4f} (vs naive 0-pred)")
            print(f"  Price: MAE=${result['MAE']:.3f}  MAPE={result['MAPE']:.3f}%  Iters={result['best_iter']}")

            print(f"\n  Market sensitivity:")
            print(f"  {'Scenario':<14} {'S&P ret':>10} {'Avg Price':>10} {'MAPE':>8}")
            print(f"  {'-'*44}")
            for label, s in result["sensitivity"].items():
                print(f"  {label:<14} {s['sp_return']*100:>9.4f}% ${s['avg_price']:>8.2f} {s['mape']:>7.3f}%")

            print(f"\n  Range coverage (market uncertainty):")
            print(f"  {'Level':<8} {'Coverage':>10} {'Width %':>10}")
            print(f"  {'-'*30}")
            for level, rm in result["range"].items():
                print(f"  {level:<8} {rm['coverage']:>9.1%} {rm['width_pct']:>9.2f}%")

            if result["conditional"]:
                print(f"\n  Conditional coverage:")
                for k, v in result["conditional"].items():
                    print(f"    {k}: {v:.1%}")

            print(f"\n  Top features (by gain):")
            for feat, imp_val in result["top_features"][:5]:
                print(f"    {feat:<40} {imp_val:>10.1f}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "metric_a_lgbm_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "target", "model", "MAE", "MAPE", "RMSE", "A_MAE",
                     "cov_1.0σ", "cov_1.5σ", "cov_2.0σ",
                     "width_1.0σ", "width_1.5σ", "width_2.0σ"])
        for r in all_results:
            rm = r["range"]
            w.writerow([r["ticker"], r["target"], r["model"], r["MAE"], r["MAPE"], r["RMSE"],
                        r["A_MAE"],
                        rm["1.0σ"]["coverage"], rm["1.5σ"]["coverage"], rm["2.0σ"]["coverage"],
                        rm["1.0σ"]["width_pct"], rm["1.5σ"]["width_pct"], rm["2.0σ"]["width_pct"]])

    pred_path = os.path.join(RESULTS_DIR, "metric_a_lgbm_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted", "half_width_1sigma"])
        for r in all_results:
            for d, a, p, hw in zip(r["dates"], r["actuals"], r["preds"], r["half_widths"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}", f"{hw:.4f}"])

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

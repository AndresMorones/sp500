"""Metric A Model — Naive baseline.

Predicts A = 0 (no abnormal return expected).
Price = prev_close * (1 + alpha + beta * avg_sp_return)
Range: varies S&P assumption within ±1σ/±2σ for market sensitivity.

This is the pure market model baseline — any model that cannot beat this
shows that news features add no predictive value for abnormal returns.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
from config import METRIC_A_TICKERS, TARGETS, LOOKBACK, RESULTS_DIR
from data_loader import (
    load_price_data, load_sp500, load_scores, load_news_phase2,
    build_metric_a_series, split_data, a_to_price, compute_sp_stats,
    compute_metrics, compute_range_metrics,
)


def run_naive(ticker, target, series):
    """Naive: predict A=0, use market model for price."""
    min_start = LOOKBACK + 1
    if len(series) < min_start + 10:
        return None

    # Collect all samples first, then split
    dates, actuals, prev_closes = [], [], []
    alphas, betas, s0s, sp_rets = [], [], [], []
    has_news_arr = []

    for idx in range(min_start, len(series)):
        day = series[idx]

        if target == "gap":
            actual = day["open"]
            has_news = bool(day["has_gap_news"])
            alpha = day["alpha_gap"]
            beta = day["beta_gap"]
            s0 = day["s0_gap"]
            sp_ret = day["sp_gap"]
        else:
            actual = day["close"]
            has_news = bool(day["has_gap_news"] or day["has_cc_news"])
            alpha = day["alpha_cc"]
            beta = day["beta_cc"]
            s0 = day["s0_cc"]
            sp_ret = day["sp_cc"]

        dates.append(day["date"])
        actuals.append(actual)
        prev_closes.append(day["prev_close"])
        alphas.append(alpha)
        betas.append(beta)
        s0s.append(s0)
        sp_rets.append(sp_ret)
        has_news_arr.append(has_news)

    n = len(actuals)
    train_end, val_end = split_data(n)

    # Compute S&P stats from training window
    sp_stats = compute_sp_stats(series, train_end + min_start)
    if target == "gap":
        sp_avg = sp_stats["sp_gap_avg"]
        sp_std = sp_stats["sp_gap_std"]
    else:
        sp_avg = sp_stats["sp_cc_avg"]
        sp_std = sp_stats["sp_cc_std"]

    # Test set
    t_dates = dates[val_end:]
    t_actuals = np.array(actuals[val_end:])
    t_prev_closes = np.array(prev_closes[val_end:])
    t_alphas = np.array(alphas[val_end:])
    t_betas = np.array(betas[val_end:])
    t_s0s = np.array(s0s[val_end:])
    t_news = np.array(has_news_arr[val_end:])

    # Naive: predict A = 0 → price from market model only
    predicted_a = np.zeros(len(t_actuals))
    t_preds, t_pred_returns = a_to_price(
        predicted_a, t_alphas, t_betas, t_s0s, sp_avg, t_prev_closes
    )

    # Point metrics
    point_metrics = compute_metrics(t_actuals, t_preds)

    # A prediction metrics (naive always predicts 0)
    actual_a = np.array([
        series[val_end + min_start + i]["A_gap" if target == "gap" else "A_cc"]
        for i in range(len(t_actuals))
    ])
    a_mae = float(np.mean(np.abs(actual_a)))

    # Market sensitivity range: vary S&P assumption
    sensitivity = {}
    for label, sp_assumption in [
        ("market_-2σ", sp_avg - 2 * sp_std),
        ("market_-1σ", sp_avg - 1 * sp_std),
        ("baseline", sp_avg),
        ("market_+1σ", sp_avg + 1 * sp_std),
        ("market_+2σ", sp_avg + 2 * sp_std),
    ]:
        prices, _ = a_to_price(predicted_a, t_alphas, t_betas, t_s0s, sp_assumption, t_prev_closes)
        sensitivity[label] = {
            "sp_return": sp_assumption,
            "avg_price": float(np.mean(prices)),
            "mape": float(np.mean(np.abs(t_actuals - prices) / t_actuals) * 100),
        }

    # Range based on market uncertainty (±1σ, ±1.5σ, ±2σ of S&P)
    # Half-width at 1σ: difference between baseline and +1σ price
    price_at_plus1 = a_to_price(predicted_a, t_alphas, t_betas, t_s0s, sp_avg + sp_std, t_prev_closes)[0]
    half_widths = np.abs(price_at_plus1 - t_preds)

    range_metrics = compute_range_metrics(t_actuals, t_preds, half_widths)

    # Conditional coverage
    cond = {}
    for k_str, rm in range_metrics.items():
        k = float(k_str.replace("σ", ""))
        hw_k = half_widths * k
        lower = t_preds - hw_k
        upper = t_preds + hw_k
        if t_news.sum() > 0:
            in_range = ((t_actuals[t_news] >= lower[t_news]) &
                        (t_actuals[t_news] <= upper[t_news]))
            cond[f"{k_str}_news_coverage"] = float(np.mean(in_range))
        if (~t_news).sum() > 0:
            in_range = ((t_actuals[~t_news] >= lower[~t_news]) &
                        (t_actuals[~t_news] <= upper[~t_news]))
            cond[f"{k_str}_no_news_coverage"] = float(np.mean(in_range))

    return {
        "ticker": ticker,
        "target": target,
        "model": "Naive",
        **point_metrics,
        "A_MAE": a_mae,
        "range": range_metrics,
        "conditional": cond,
        "sensitivity": sensitivity,
        "sp_avg": sp_avg,
        "sp_std": sp_std,
        "n_test": len(t_actuals),
        "n_news_days": int(t_news.sum()),
        "dates": t_dates,
        "actuals": t_actuals.tolist(),
        "preds": t_preds.tolist(),
        "half_widths": half_widths.tolist(),
    }


def main():
    print("=" * 60)
    print("  Metric A Model — Naive (A=0, market model only)")
    print("=" * 60)

    price_data = load_price_data()
    sp500 = load_sp500()
    scores = load_scores()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    for target in TARGETS:
        print(f"\n--- Target: {target} (predict {'open' if target == 'gap' else 'close'}) ---")

        for ticker in METRIC_A_TICKERS:
            news = load_news_phase2(ticker)
            series = build_metric_a_series(ticker, price_data, sp500, scores, news)
            result = run_naive(ticker, target, series)
            if result is None:
                print(f"  {ticker}: insufficient data")
                continue

            all_results.append(result)
            print(f"\n  {ticker} — {result['n_test']} test days ({result['n_news_days']} with news)")
            print(f"  A prediction MAE: {result['A_MAE']:.4f} (naive predicts 0)")
            print(f"  Price: MAE=${result['MAE']:.3f}  MAPE={result['MAPE']:.3f}%")

            print(f"\n  Market sensitivity (S&P avg={result['sp_avg']*100:.4f}%, std={result['sp_std']*100:.3f}%):")
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

    # Save results
    out_path = os.path.join(RESULTS_DIR, "metric_a_naive_results.csv")
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

    pred_path = os.path.join(RESULTS_DIR, "metric_a_naive_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted", "half_width_1sigma"])
        for r in all_results:
            for d, a, p, hw in zip(r["dates"], r["actuals"], r["preds"], r["half_widths"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}", f"{hw:.4f}"])

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

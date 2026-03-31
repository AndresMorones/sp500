"""Phase 3 comparison — cross-track model comparison.

Loads all result CSVs from results/ and produces:
1. Price model comparison (GOOGL × 4 models × 2 targets)
2. Metric A model comparison — A prediction + price MAPE (GOOGL × 4 models)
3. Cross-track ranking
4. Range coverage comparison (market uncertainty)
"""

import csv
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_results(filename):
    """Load a results CSV from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in r:
                if k not in ("ticker", "target", "model"):
                    try:
                        r[k] = float(r[k])
                    except (ValueError, TypeError):
                        pass
            rows.append(r)
    return rows


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    print("\n" + "█" * 70)
    print("  PHASE 3 — MODEL COMPARISON REPORT (Corrected)")
    print("█" * 70)

    # Load all results
    price_files = {
        "Naive": "price_naive_results.csv",
        "Ridge": "price_ridge_results.csv",
        "LightGBM": "price_lgbm_results.csv",
        "LSTM": "price_lstm_results.csv",
    }
    metric_a_files = {
        "Naive": "metric_a_naive_results.csv",
        "Ridge": "metric_a_ridge_results.csv",
        "LightGBM": "metric_a_lgbm_results.csv",
        "LSTM": "metric_a_lstm_results.csv",
    }

    all_price = []
    for model, fname in price_files.items():
        all_price.extend(load_results(fname))

    all_metric_a = []
    for model, fname in metric_a_files.items():
        all_metric_a.extend(load_results(fname))

    # ─── Track 1: Price Models (GOOGL only) ───
    if all_price:
        print_section("TRACK 1: Price Model (OHLCV features → predict return → price)")
        print("  Method: predict return, convert via prev_close × (1 + pred_return)")
        for target in ["gap", "cc"]:
            googl = [r for r in all_price if r["target"] == target and r["ticker"] == "GOOGL"]
            if not googl:
                continue
            print(f"\n  {target} target — GOOGL")
            print(f"  {'Model':<12} {'MAE ($)':>10} {'MAPE (%)':>12} {'RMSE ($)':>10}")
            print(f"  {'-' * 46}")
            for m in ["Naive", "Ridge", "LightGBM", "LSTM"]:
                match = [r for r in googl if r["model"] == m]
                if match:
                    r = match[0]
                    mape_str = f"{r['MAPE']:.3f}%"
                    if "MAPE_std" in r and r["MAPE_std"] > 0:
                        mape_str = f"{r['MAPE']:.3f}±{r['MAPE_std']:.2f}%"
                    print(f"  {m:<12} {r['MAE']:>10.3f} {mape_str:>12} {r['RMSE']:>10.3f}")

    # ─── Track 2: Metric A Models ───
    if all_metric_a:
        print_section("TRACK 2: Metric A Model (news categories → predict A → price)")
        print("  Method: predict A from news, then A → zi → return → price")
        print("  Price uses avg S&P daily return as market assumption")

        for target in ["gap", "cc"]:
            target_results = [r for r in all_metric_a if r["target"] == target]
            if not target_results:
                continue

            print(f"\n  {target} target — GOOGL")
            print(f"  {'Model':<12} {'A MAE':>8} {'Price MAE':>10} {'Price MAPE':>12} {'Price RMSE':>11}")
            print(f"  {'-' * 55}")
            for m in ["Naive", "Ridge", "LightGBM", "LSTM"]:
                match = [r for r in target_results if r["model"] == m]
                if match:
                    r = match[0]
                    a_mae = r.get("A_MAE", 0)
                    a_str = f"{a_mae:.4f}"
                    if "A_MAE_std" in r and r["A_MAE_std"] > 0:
                        a_str = f"{a_mae:.3f}±{r['A_MAE_std']:.2f}"
                    mape_str = f"{r['MAPE']:.3f}%"
                    if "MAPE_std" in r and r["MAPE_std"] > 0:
                        mape_str = f"{r['MAPE']:.3f}±{r['MAPE_std']:.2f}%"
                    print(f"  {m:<12} {a_str:>8} ${r['MAE']:>9.3f} {mape_str:>12} ${r['RMSE']:>10.3f}")

        # Range coverage
        print_section("TRACK 2: Range Coverage (market uncertainty)")
        for target in ["gap", "cc"]:
            target_results = [r for r in all_metric_a if r["target"] == target]
            if not target_results:
                continue
            print(f"\n  {target} target — GOOGL")
            print(f"  {'Model':<12} {'1σ Cov':>8} {'1.5σ Cov':>10} {'2σ Cov':>8} {'1σ Width':>10}")
            print(f"  {'-' * 50}")
            for m in ["Naive", "Ridge", "LightGBM", "LSTM"]:
                match = [r for r in target_results if r["model"] == m]
                if match:
                    r = match[0]
                    c1 = r.get("cov_1.0σ", 0)
                    c15 = r.get("cov_1.5σ", 0)
                    c2 = r.get("cov_2.0σ", 0)
                    w1 = r.get("width_1.0σ", 0)
                    print(f"  {m:<12} {c1:>7.1%} {c15:>9.1%} {c2:>7.1%} {w1:>9.2f}%")

    # ─── Cross-Track Ranking ───
    if all_price and all_metric_a:
        print_section("CROSS-TRACK RANKING — GOOGL (sorted by MAPE)")
        for target in ["gap", "cc"]:
            googl_price = [r for r in all_price if r["ticker"] == "GOOGL" and r["target"] == target]
            googl_metric = [r for r in all_metric_a if r["ticker"] == "GOOGL" and r["target"] == target]

            combined = []
            for r in googl_price:
                combined.append(("Price", r))
            for r in googl_metric:
                combined.append(("MetricA", r))

            if combined:
                combined.sort(key=lambda x: x[1].get("MAPE", 999))
                print(f"\n  {target} target")
                print(f"  {'#':>3} {'Track':<10} {'Model':<12} {'MAPE (%)':>12} {'MAE ($)':>10}")
                print(f"  {'-' * 50}")
                for i, (track, r) in enumerate(combined, 1):
                    mape_str = f"{r['MAPE']:.3f}%"
                    if "MAPE_std" in r and r["MAPE_std"] > 0:
                        mape_str = f"{r['MAPE']:.3f}±{r['MAPE_std']:.2f}%"
                    print(f"  {i:>3} {track:<10} {r['model']:<12} {mape_str:>12} {r['MAE']:>10.3f}")

    # ─── Summary ───
    print_section("SUMMARY")
    if all_price:
        for target in ["gap", "cc"]:
            googl = [r for r in all_price if r["target"] == target and r["ticker"] == "GOOGL"]
            if googl:
                best = min(googl, key=lambda x: x.get("MAPE", 999))
                print(f"  Best price model ({target}): {best['model']} — MAPE {best['MAPE']:.3f}%")

    if all_metric_a:
        for target in ["gap", "cc"]:
            target_results = [r for r in all_metric_a if r["target"] == target]
            if target_results:
                best = min(target_results, key=lambda x: x.get("MAPE", 999))
                print(f"  Best metric A model ({target}): {best['model']} — MAPE {best['MAPE']:.3f}%")

    print()


if __name__ == "__main__":
    main()

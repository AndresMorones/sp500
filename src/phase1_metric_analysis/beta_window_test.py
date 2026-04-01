"""
Beta window size test: 30 vs 60 vs 120 days, single vs asymmetric beta.
Uses close-to-close returns. Out-of-sample backtest with no look-ahead bias.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    RAW_DIR, STOCK_HISTORY_FILES,
    mean, var, std, cov, median, ols, ols_asymmetric,
    load_sp500_close, load_price_csv_close, load_stock_history,
)

WINDOWS = [30, 60, 120]

# --- Helpers specific to this script ---

def predict_single(alpha, beta, rm):
    return alpha + beta * rm

def predict_asymmetric(alpha, bu, bd, rm):
    return alpha + bu * max(rm, 0) + bd * min(rm, 0)

# --- 1. Load data ---

sp500 = load_sp500_close()
price_dates, price_data = load_price_csv_close()
load_stock_history(sp500, price_data, include_ohlcv=False)

# --- 2. Build per-ticker close-to-close return series ---

all_tickers = sorted(price_dates.keys())

stocks = {}  # ticker -> list of (date, stock_return, sp_return)
for ticker in all_tickers:
    all_dates = sorted(set(d for (t, d) in price_data if t == ticker and d in sp500))
    returns = []
    for i in range(1, len(all_dates)):
        prev_date, curr_date = all_dates[i - 1], all_dates[i]
        prev_close = price_data[(ticker, prev_date)]
        curr_close = price_data[(ticker, curr_date)]
        prev_sp = sp500[prev_date]
        curr_sp = sp500[curr_date]
        if prev_close > 0 and prev_sp > 0:
            rs = (curr_close - prev_close) / prev_close
            rm = (curr_sp - prev_sp) / prev_sp
            returns.append((curr_date, rs, rm))
    stocks[ticker] = returns

# --- 3. Run backtest ---

def run_backtest(stocks, price_dates, all_tickers, windows, mode):
    """Run backtest.
    mode='market': predict using alpha + beta * actual_rm (you know S&P)
    mode='mean': predict using rolling mean of stock returns (no market info)
    mode='median': predict using rolling median of stock returns (E/Ev baseline)
    """
    all_results = []

    for ticker in all_tickers:
        ret = stocks[ticker]
        n_total = len(ret)
        target_dates = price_dates[ticker]

        for window in windows:
            beta_types = ["single", "asymmetric"] if mode == "market" else ["single"]
            for beta_type in beta_types:
                errors = []
                for i in range(window, n_total):
                    date_i = ret[i][0]
                    if date_i not in target_dates:
                        continue

                    w_rs = [ret[j][1] for j in range(i - window, i)]
                    w_rm = [ret[j][2] for j in range(i - window, i)]

                    actual_rs = ret[i][1]
                    actual_rm = ret[i][2]

                    if mode == "mean":
                        predicted = mean(w_rs)
                    elif mode == "median":
                        predicted = median(w_rs)
                    elif beta_type == "single":
                        alpha, beta = ols(w_rm, w_rs)
                        predicted = predict_single(alpha, beta, actual_rm)
                    else:
                        alpha, bu, bd = ols_asymmetric(w_rm, w_rs)
                        predicted = predict_asymmetric(alpha, bu, bd, actual_rm)

                    errors.append(actual_rs - predicted)

                if not errors:
                    continue

                abs_errors = [abs(e) for e in errors]
                mae = mean(abs_errors)
                rmse = math.sqrt(mean([e * e for e in errors]))
                medae = sorted(abs_errors)[len(abs_errors) // 2]
                mean_error = mean(errors)

                all_results.append({
                    "ticker": ticker,
                    "window": window,
                    "beta_type": beta_type,
                    "n": len(errors),
                    "mae": mae,
                    "rmse": rmse,
                    "medae": medae,
                    "bias": mean_error,
                })

    return all_results

def print_results(all_results, all_tickers, windows, title):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    beta_types_in_results = sorted(set(r["beta_type"] for r in all_results))

    # Per-ticker tables
    for ticker in all_tickers:
        ticker_res = [r for r in all_results if r["ticker"] == ticker]
        if not ticker_res:
            continue
        print(f"\n{'-' * 90}")
        print(f"  {ticker} ({ticker_res[0]['n']} target days at window={windows[0]})")
        print(f"{'-' * 90}")
        print(f"  {'Window':>8} {'Beta Type':>12} {'N':>6} {'MAE':>10} {'RMSE':>10} {'MedAE':>10} {'Bias':>10}")
        print(f"  {'-'*8} {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        for r in ticker_res:
            print(f"  {r['window']:>8} {r['beta_type']:>12} {r['n']:>6} "
                  f"{r['mae']*100:>9.4f}% {r['rmse']*100:>9.4f}% "
                  f"{r['medae']*100:>9.4f}% {r['bias']*100:>+9.5f}%")

    # Cross-ticker summary
    print(f"\n{'=' * 90}")
    print(f"  CROSS-TICKER SUMMARY (average MAE across all tickers)")
    print(f"{'=' * 90}")
    print(f"  {'Window':>8} {'Beta Type':>12} {'Avg MAE':>10} {'Avg RMSE':>10} {'Avg MedAE':>10} {'Avg Bias':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for window in windows:
        for beta_type in beta_types_in_results:
            subset = [r for r in all_results if r["window"] == window and r["beta_type"] == beta_type]
            if not subset:
                continue
            avg_mae = mean([r["mae"] for r in subset])
            avg_rmse = mean([r["rmse"] for r in subset])
            avg_medae = mean([r["medae"] for r in subset])
            avg_bias = mean([r["bias"] for r in subset])
            print(f"  {window:>8} {beta_type:>12} {avg_mae*100:>9.4f}% {avg_rmse*100:>9.4f}% "
                  f"{avg_medae*100:>9.4f}% {avg_bias*100:>+9.5f}%")

    # Winner per ticker
    print(f"\n{'=' * 90}")
    print(f"  WINNER PER TICKER (lowest MAE)")
    print(f"{'=' * 90}")
    for ticker in all_tickers:
        ticker_res = [r for r in all_results if r["ticker"] == ticker]
        if not ticker_res:
            continue
        best = min(ticker_res, key=lambda r: r["mae"])
        print(f"  {ticker}: window={best['window']}, beta={best['beta_type']}, "
              f"MAE={best['mae']*100:.4f}%")

    # Asymmetric vs single (only if both exist)
    if "asymmetric" in beta_types_in_results:
        print(f"\n{'=' * 90}")
        print(f"  ASYMMETRIC vs SINGLE BETA (MAE difference, same window)")
        print(f"{'=' * 90}")
        print(f"  {'Ticker':>8} {'Window':>8} {'Single MAE':>12} {'Asym MAE':>12} {'Diff':>10} {'Better':>10}")
        for ticker in all_tickers:
            for window in windows:
                s = [r for r in all_results if r["ticker"] == ticker and r["window"] == window and r["beta_type"] == "single"]
                a = [r for r in all_results if r["ticker"] == ticker and r["window"] == window and r["beta_type"] == "asymmetric"]
                if not s or not a:
                    continue
                s, a = s[0], a[0]
                diff = a["mae"] - s["mae"]
                better = "asym" if diff < 0 else "single" if diff > 0 else "tie"
                print(f"  {ticker:>8} {window:>8} {s['mae']*100:>11.4f}% {a['mae']*100:>11.4f}% "
                      f"{diff*100:>+9.5f}% {better:>10}")


if __name__ == "__main__":
    # ============================================================
    # TEST B: Predict with known S&P return (alpha + beta * rm)
    # ============================================================
    results_b = run_backtest(stocks, price_dates, all_tickers, WINDOWS, mode="market")
    print_results(results_b, all_tickers, WINDOWS,
                  "TEST B: Predict stock return GIVEN S&P return (alpha + beta * rm)")

    # ============================================================
    # TEST A1: Predict with NO market info — rolling mean
    # ============================================================
    results_a1 = run_backtest(stocks, price_dates, all_tickers, WINDOWS, mode="mean")
    print_results(results_a1, all_tickers, WINDOWS,
                  "TEST A1: Predict with NO market info (rolling MEAN — like score A baseline)")

    # ============================================================
    # TEST A2: Predict with NO market info — rolling median (E/Ev baseline)
    # ============================================================
    results_a2 = run_backtest(stocks, price_dates, all_tickers, WINDOWS, mode="median")
    print_results(results_a2, all_tickers, WINDOWS,
                  "TEST A2: Predict with NO market info (rolling MEDIAN — like score E/Ev baseline)")

    # ============================================================
    # Compare all three approaches
    # ============================================================
    print(f"\n{'=' * 120}")
    print(f"  COMPARISON: All 3 models side by side (single beta, MAE)")
    print(f"  B = alpha+beta*rm | A1 = rolling mean | A2 = rolling median (E/Ev)")
    print(f"{'=' * 120}")
    print(f"  {'Ticker':>8} {'Window':>8} {'B (market)':>12} {'A1 (mean)':>12} {'A2 (median)':>12} "
          f"{'B vs A1':>10} {'B vs A2':>10} {'A1 vs A2':>10}")

    for ticker in all_tickers:
        for window in WINDOWS:
            rb = [r for r in results_b if r["ticker"] == ticker and r["window"] == window and r["beta_type"] == "single"][0]
            ra1 = [r for r in results_a1 if r["ticker"] == ticker and r["window"] == window][0]
            ra2 = [r for r in results_a2 if r["ticker"] == ticker and r["window"] == window][0]
            b_vs_a1 = (ra1["mae"] - rb["mae"]) / ra1["mae"] * 100
            b_vs_a2 = (ra2["mae"] - rb["mae"]) / ra2["mae"] * 100
            a1_vs_a2 = (ra2["mae"] - ra1["mae"]) / ra1["mae"] * 100
            print(f"  {ticker:>8} {window:>8} {rb['mae']*100:>11.4f}% {ra1['mae']*100:>11.4f}% {ra2['mae']*100:>11.4f}% "
                  f"{b_vs_a1:>+9.1f}% {b_vs_a2:>+9.1f}% {a1_vs_a2:>+9.1f}%")

    # Cross-ticker summary of best config per model
    print(f"\n{'=' * 120}")
    print(f"  BEST WINDOW PER MODEL (cross-ticker avg MAE)")
    print(f"{'=' * 120}")
    for label, results in [("B (market model)", results_b), ("A1 (mean)", results_a1), ("A2 (median/E-Ev)", results_a2)]:
        print(f"\n  {label}:")
        for window in WINDOWS:
            subset = [r for r in results if r["window"] == window and r["beta_type"] == "single"]
            if not subset:
                continue
            avg_mae = mean([r["mae"] for r in subset])
            print(f"    window={window:>3}: avg MAE = {avg_mae*100:.4f}%")

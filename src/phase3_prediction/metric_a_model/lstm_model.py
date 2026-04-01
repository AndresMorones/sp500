"""Metric A Model — LSTM predicting A from news sequences.

Architecture: LSTM(32) → Dropout(0.3) → LSTM(16) → Dropout(0.3) → Dense(1)
Input: 10-day sequences of news category features (zero-filled for no-news days).
Predicts A score, converts to price via inverse market model.
GOOGL only. Runs 5 seeds, reports mean ± std.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from config import (
    METRIC_A_TICKERS, TARGETS, SEEDS, LOOKBACK, RESULTS_DIR,
    LSTM_HIDDEN, LSTM_DROPOUT, LSTM_LR, LSTM_BATCH,
    LSTM_EPOCHS, LSTM_PATIENCE,
)
from data_loader import (
    load_price_data, load_sp500, load_scores, load_news_phase2,
    get_news_cat_columns, build_metric_a_series,
    split_data, scale_splits, a_to_price, compute_sp_stats,
    compute_metrics, compute_range_metrics,
)


def make_news_sequences(series, target, cat_cols):
    """Create LSTM sequences of news features → A target.

    Each timestep has: news category scores + distinct_events + has_news.
    Zero-filled for days with no news.
    Returns (X, y_a, dates, prev_closes, has_news, indices).
    """
    X, y_vals, dates, prev_closes, has_news_arr, indices = [], [], [], [], [], []

    for idx in range(LOOKBACK + 1, len(series)):
        day = series[idx]

        # Target: A score
        if target == "gap":
            y_val = day["A_gap"]
            has_news_val = day["has_gap_news"]
        else:
            y_val = day["A_cc"]
            has_news_val = day["has_gap_news"] or day["has_cc_news"]

        # Sequence: lookback window of news features per day
        seq = []
        for lag in range(LOOKBACK, 0, -1):
            j = idx - lag
            step = [
                series[j]["has_gap_news"],
                series[j]["gap_distinct_events"],
            ]
            for c in cat_cols:
                step.append(series[j].get(f"gap_{c}", 0))
            if target == "cc":
                step.append(series[j]["has_cc_news"])
                step.append(series[j]["cc_distinct_events"])
                for c in cat_cols:
                    step.append(series[j].get(f"cc_{c}", 0))
            seq.append(step)

        # Prediction day's news as final timestep
        pred_step = [
            day["has_gap_news"],
            day["gap_distinct_events"],
        ]
        for c in cat_cols:
            pred_step.append(day.get(f"gap_{c}", 0))
        if target == "cc":
            pred_step.append(day["has_cc_news"])
            pred_step.append(day["cc_distinct_events"])
            for c in cat_cols:
                pred_step.append(day.get(f"cc_{c}", 0))
        seq.append(pred_step)

        X.append(seq)
        y_vals.append(y_val)
        dates.append(day["date"])
        prev_closes.append(day["prev_close"])
        has_news_arr.append(has_news_val)
        indices.append(idx)

    return (
        np.array(X, dtype=np.float64),
        np.array(y_vals),
        dates,
        np.array(prev_closes),
        np.array(has_news_arr, dtype=bool),
        indices,
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


def run_lstm_single_seed(ticker, target, series, cat_cols, seed, sp_stats):
    """Train and evaluate LSTM for one seed."""
    import tensorflow as tf
    np.random.seed(seed)
    tf.random.set_seed(seed)

    X, y_a, dates, prev_closes, has_news, indices = make_news_sequences(
        series, target, cat_cols
    )
    n = len(y_a)
    train_end, val_end = split_data(n)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val = y_a[:train_end], y_a[train_end:val_end]
    y_test_a = y_a[val_end:]
    test_dates = dates[val_end:]
    test_prev_closes = prev_closes[val_end:]
    test_has_news = has_news[val_end:]
    test_indices = indices[val_end:]

    # Scale features
    X_train, X_val, X_test, _ = scale_splits(X_train, X_val, X_test)

    # Scale target (A scores)
    from sklearn.preprocessing import MinMaxScaler
    y_sc = MinMaxScaler()
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

    # Predict A scores
    pred_a_sc = model.predict(X_test, verbose=0).ravel()
    pred_a = y_sc.inverse_transform(pred_a_sc.reshape(-1, 1)).ravel()

    # A prediction metrics
    a_mae = float(np.mean(np.abs(y_test_a - pred_a)))

    # Get market model parameters for test days
    if target == "gap":
        t_alphas = np.array([series[i]["alpha_gap"] for i in test_indices])
        t_betas = np.array([series[i]["beta_gap"] for i in test_indices])
        t_s0s = np.array([series[i]["s0_gap"] for i in test_indices])
        actual_prices = np.array([series[i]["open"] for i in test_indices])
        sp_avg = sp_stats["sp_gap_avg"]
        sp_std = sp_stats["sp_gap_std"]
    else:
        t_alphas = np.array([series[i]["alpha_cc"] for i in test_indices])
        t_betas = np.array([series[i]["beta_cc"] for i in test_indices])
        t_s0s = np.array([series[i]["s0_cc"] for i in test_indices])
        actual_prices = np.array([series[i]["close"] for i in test_indices])
        sp_avg = sp_stats["sp_cc_avg"]
        sp_std = sp_stats["sp_cc_std"]

    # Convert predicted A → price
    pred_prices, _ = a_to_price(pred_a, t_alphas, t_betas, t_s0s, sp_avg, test_prev_closes)

    point_metrics = compute_metrics(actual_prices, pred_prices)

    # Range from market uncertainty
    price_at_plus1 = a_to_price(pred_a, t_alphas, t_betas, t_s0s, sp_avg + sp_std, test_prev_closes)[0]
    half_widths = np.abs(price_at_plus1 - pred_prices)

    range_metrics = compute_range_metrics(actual_prices, pred_prices, half_widths)

    return {
        **point_metrics,
        "A_MAE": a_mae,
        "range": range_metrics,
        "dates": test_dates,
        "actuals": actual_prices.tolist(),
        "preds": pred_prices.tolist(),
        "half_widths": half_widths.tolist(),
        "has_news": test_has_news.tolist(),
    }


def run_lstm(ticker, target, series, cat_cols, sp_stats):
    """Run LSTM across all seeds, aggregate results."""
    seed_results = []
    for seed in SEEDS:
        r = run_lstm_single_seed(ticker, target, series, cat_cols, seed, sp_stats)
        seed_results.append(r)

    maes = [r["MAE"] for r in seed_results]
    mapes = [r["MAPE"] for r in seed_results]
    rmses = [r["RMSE"] for r in seed_results]
    a_maes = [r["A_MAE"] for r in seed_results]

    # Use median seed's predictions
    median_idx = int(np.argmin([abs(m - np.median(mapes)) for m in mapes]))
    rep = seed_results[median_idx]

    # Aggregate range metrics
    avg_range = {}
    for k_str in ["1.0σ", "1.5σ", "2.0σ"]:
        covs = [r["range"][k_str]["coverage"] for r in seed_results]
        widths = [r["range"][k_str]["width_pct"] for r in seed_results]
        winklers = [r["range"][k_str]["winkler"] for r in seed_results]
        avg_range[k_str] = {
            "coverage": float(np.mean(covs)),
            "coverage_std": float(np.std(covs)),
            "width_pct": float(np.mean(widths)),
            "winkler": float(np.mean(winklers)),
        }

    # Conditional coverage from representative seed
    test_has_news = np.array(rep["has_news"])
    actual_prices = np.array(rep["actuals"])
    pred_prices = np.array(rep["preds"])
    half_widths = np.array(rep["half_widths"])
    cond = {}
    for k_str in ["1.0σ", "1.5σ", "2.0σ"]:
        k = float(k_str.replace("σ", ""))
        hw_k = half_widths * k
        lower = pred_prices - hw_k
        upper = pred_prices + hw_k
        news_mask = test_has_news
        if news_mask.sum() > 0:
            in_range = ((actual_prices[news_mask] >= lower[news_mask]) &
                        (actual_prices[news_mask] <= upper[news_mask]))
            cond[f"{k_str}_news_coverage"] = float(np.mean(in_range))
        if (~news_mask).sum() > 0:
            in_range = ((actual_prices[~news_mask] >= lower[~news_mask]) &
                        (actual_prices[~news_mask] <= upper[~news_mask]))
            cond[f"{k_str}_no_news_coverage"] = float(np.mean(in_range))

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
        "A_MAE": float(np.mean(a_maes)),
        "A_MAE_std": float(np.std(a_maes)),
        "range": avg_range,
        "conditional": cond,
        "n_test": len(rep["actuals"]),
        "n_news_days": int(np.sum(test_has_news)),
        "dates": rep["dates"],
        "actuals": rep["actuals"],
        "preds": rep["preds"],
        "half_widths": rep["half_widths"],
    }


def main():
    print("=" * 60)
    print("  Metric A Model — LSTM (news → A → price, 5 seeds)")
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

            # Compute S&P stats from training window
            min_start = LOOKBACK + 1
            n_samples = len(series) - min_start
            train_end_idx, _ = split_data(n_samples)
            sp_stats = compute_sp_stats(series, train_end_idx + min_start)

            print(f"  Training {ticker} ({len(SEEDS)} seeds)...", flush=True)
            result = run_lstm(ticker, target, series, cat_cols, sp_stats)
            all_results.append(result)

            print(f"\n  {ticker} — {result['n_test']} test days ({result['n_news_days']} with news)")
            print(f"  A prediction MAE: {result['A_MAE']:.4f}±{result['A_MAE_std']:.4f}")
            print(f"  Price: MAPE={result['MAPE']:.3f}±{result['MAPE_std']:.3f}%  "
                  f"MAE=${result['MAE']:.3f}±{result['MAE_std']:.3f}")

            print(f"\n  Range coverage (market uncertainty, mean across seeds):")
            print(f"  {'Level':<8} {'Coverage':>14} {'Width %':>10}")
            print(f"  {'-'*34}")
            for level, rm in result["range"].items():
                print(f"  {level:<8} {rm['coverage']:>6.1%}±{rm.get('coverage_std',0):.1%}  "
                      f"{rm['width_pct']:>9.2f}%")

            if result["conditional"]:
                print(f"\n  Conditional coverage:")
                for k, v in result["conditional"].items():
                    print(f"    {k}: {v:.1%}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "metric_a_lstm_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "target", "model", "MAE", "MAE_std", "MAPE", "MAPE_std",
                     "RMSE", "RMSE_std", "A_MAE", "A_MAE_std",
                     "cov_1.0σ", "cov_1.5σ", "cov_2.0σ",
                     "width_1.0σ", "width_1.5σ", "width_2.0σ"])
        for r in all_results:
            rm = r["range"]
            w.writerow([r["ticker"], r["target"], r["model"],
                        r["MAE"], r["MAE_std"], r["MAPE"], r["MAPE_std"],
                        r["RMSE"], r["RMSE_std"], r["A_MAE"], r["A_MAE_std"],
                        rm["1.0σ"]["coverage"], rm["1.5σ"]["coverage"], rm["2.0σ"]["coverage"],
                        rm["1.0σ"]["width_pct"], rm["1.5σ"]["width_pct"], rm["2.0σ"]["width_pct"]])

    pred_path = os.path.join(RESULTS_DIR, "metric_a_lstm_predictions.csv")
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "ticker", "target", "actual", "predicted", "half_width_1sigma"])
        for r in all_results:
            for d, a, p, hw in zip(r["dates"], r["actuals"], r["preds"], r["half_widths"]):
                w.writerow([d, r["ticker"], r["target"], f"{a:.4f}", f"{p:.4f}", f"{hw:.4f}"])

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

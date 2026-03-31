"""Phase 3 shared data loading and feature engineering.

Loads price, scores, and news data. Builds per-ticker time series.
Provides sequence creation and train/val/test splitting utilities.
"""

import csv
import math
import os
from datetime import datetime

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import (
    PRICE_CSV, SP500_CSV, SCORES_CSV, OUT_DIR,
    LOOKBACK, SPLIT_RATIOS,
)

# ─── Raw data loaders ───────────────────────────────────────────────────────

def load_price_data():
    """Load price.csv → dict keyed by (ticker, date) with OHLCV."""
    rows = {}
    with open(PRICE_CSV) as f:
        for r in csv.DictReader(f):
            rows[(r["ticker"], r["date"])] = {
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": int(float(r["volume"])),
            }
    return rows


def load_sp500():
    """Load S&P 500 data → dict: date_str → {open, close}."""
    sp = {}
    with open(SP500_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            dt = datetime.strptime(r["Date"], "%m/%d/%Y")
            ds = dt.strftime("%Y-%m-%d")
            sp[ds] = {
                "open": float(r["Open"].replace(",", "")),
                "close": float(r["Price"].replace(",", "")),
            }
    return sp


def load_scores():
    """Load scores_output.csv → dict: (ticker, date) → score fields."""
    scores = {}
    with open(SCORES_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            scores[(r["ticker"], r["date"])] = {
                "A_gap": float(r["A_gap"]),
                "A_cc": float(r["A_cc"]),
                "zv": float(r["zv"]),
                "beta_gap": float(r["beta_gap"]),
                "beta_cc": float(r["beta_cc"]),
                "alpha_gap": float(r["alpha_gap"]),
                "alpha_cc": float(r["alpha_cc"]),
                "s0_gap": float(r["s0_gap"]),
                "s0_cc": float(r["s0_cc"]),
                "stock_gap": float(r["stock_gap"]),
                "stock_cc": float(r["stock_cc"]),
                "sp_gap": float(r["sp_gap"]),
                "sp_cc": float(r["sp_cc"]),
            }
    return scores


def load_news_phase2(ticker):
    """Load news_phase2_{ticker}.csv → dict: (date, period) → category scores.

    Returns dict keyed by (date, period) where period is 'gap' or 'cc'.
    Values are dicts with all cat_* columns + distinct_events.
    """
    path = os.path.join(OUT_DIR, f"news_phase2_{ticker}.csv")
    if not os.path.exists(path):
        return {}
    news = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cat_cols = [c for c in reader.fieldnames if c.startswith("cat_")]
        for r in reader:
            key = (r["date"], r["period"])
            record = {"distinct_events": int(r["distinct_events"])}
            for c in cat_cols:
                record[c] = int(r[c])
            news[key] = record
    return news


def get_news_cat_columns(ticker):
    """Return the list of cat_* column names from news_phase2 CSV."""
    path = os.path.join(OUT_DIR, f"news_phase2_{ticker}.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [c for c in reader.fieldnames if c.startswith("cat_")]


# ─── Per-ticker series builders ─────────────────────────────────────────────

def build_price_series(ticker, price_data, sp500):
    """Build chronological list of day dicts with OHLCV + returns for a ticker.

    Each day dict contains: date, open, high, low, close, volume,
    ret_cc (close-to-close return), ret_gap (prev close → open).
    """
    dates = sorted(d for (t, d) in price_data if t == ticker)
    series = []
    for i, date in enumerate(dates):
        d = price_data[(ticker, date)]
        day = {
            "date": date,
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d["volume"],
        }
        if i > 0:
            prev_close = price_data[(ticker, dates[i - 1])]["close"]
            day["prev_close"] = prev_close
            day["ret_cc"] = (d["close"] - prev_close) / prev_close
            day["ret_gap"] = (d["open"] - prev_close) / prev_close
        else:
            day["prev_close"] = d["close"]
            day["ret_cc"] = 0.0
            day["ret_gap"] = 0.0

        # S&P 500 returns
        if date in sp500 and i > 0 and dates[i - 1] in sp500:
            sp_prev = sp500[dates[i - 1]]["close"]
            day["sp_ret_cc"] = (sp500[date]["close"] - sp_prev) / sp_prev
            day["sp_ret_gap"] = (sp500[date]["open"] - sp_prev) / sp_prev
        else:
            day["sp_ret_cc"] = 0.0
            day["sp_ret_gap"] = 0.0

        series.append(day)
    return series


def build_metric_a_series(ticker, price_data, sp500, scores, news_phase2):
    """Build chronological series with metric A features + news scores.

    Returns list of day dicts. Each contains:
    - A_gap, A_cc, zv, beta_gap, beta_cc (from scores)
    - stock_gap, stock_cc, sp_gap, sp_cc (returns from scores)
    - News category scores for gap and cc periods (zero-filled if no news)
    - prev_close, close, open (for price reconstruction)
    """
    cat_cols = get_news_cat_columns(ticker)
    dates = sorted(d for (t, d) in price_data if t == ticker)
    series = []

    for i, date in enumerate(dates):
        d = price_data[(ticker, date)]
        sc = scores.get((ticker, date))
        if sc is None:
            continue  # no score data for this day

        day = {
            "date": date,
            "open": d["open"],
            "close": d["close"],
            "volume": d["volume"],
            # Metric A and related
            "A_gap": sc["A_gap"],
            "A_cc": sc["A_cc"],
            "zv": sc["zv"],
            "beta_gap": sc["beta_gap"],
            "beta_cc": sc["beta_cc"],
            # Market model parameters (for inverse A → price conversion)
            "alpha_gap": sc["alpha_gap"],
            "alpha_cc": sc["alpha_cc"],
            "s0_gap": sc["s0_gap"],
            "s0_cc": sc["s0_cc"],
            # Returns (from scores pipeline)
            "stock_gap": sc["stock_gap"],
            "stock_cc": sc["stock_cc"],
            "sp_gap": sc["sp_gap"],
            "sp_cc": sc["sp_cc"],
        }

        if i > 0:
            day["prev_close"] = price_data[(ticker, dates[i - 1])]["close"]
        else:
            day["prev_close"] = d["close"]

        # Gap-period news scores
        gap_news = news_phase2.get((date, "gap"), {})
        day["has_gap_news"] = 1 if gap_news else 0
        day["gap_distinct_events"] = gap_news.get("distinct_events", 0)
        for c in cat_cols:
            day[f"gap_{c}"] = gap_news.get(c, 0)

        # CC-period news scores
        cc_news = news_phase2.get((date, "cc"), {})
        day["has_cc_news"] = 1 if cc_news else 0
        day["cc_distinct_events"] = cc_news.get("distinct_events", 0)
        for c in cat_cols:
            day[f"cc_{c}"] = cc_news.get(c, 0)

        series.append(day)
    return series


# ─── Feature extraction ─────────────────────────────────────────────────────

def extract_price_features(series, idx, target):
    """Extract OHLCV features for predicting day[idx]'s target.

    Uses LOOKBACK days of history ending at idx-1 (no look-ahead).
    target: 'gap' (predict open) or 'cc' (predict close).
    Returns (feature_dict, target_value) or None if insufficient history.
    """
    if idx < LOOKBACK + 1:
        return None

    # Target value
    if target == "gap":
        y = series[idx]["open"]
    else:
        y = series[idx]["close"]

    feats = {}

    # Reference price for normalizing all dollar values to ratios
    ref_close = series[idx - 1]["close"]

    # Lookback window: [idx - LOOKBACK, idx) — prices as ratios to yesterday's close
    for lag in range(1, LOOKBACK + 1):
        j = idx - lag
        feats[f"close_ratio_lag{lag}"] = series[j]["close"] / ref_close
        feats[f"open_ratio_lag{lag}"] = series[j]["open"] / ref_close
        feats[f"high_ratio_lag{lag}"] = series[j]["high"] / ref_close
        feats[f"low_ratio_lag{lag}"] = series[j]["low"] / ref_close
        feats[f"volume_lag{lag}"] = math.log(max(series[j]["volume"], 1))

    # Returns (lagged)
    for lag in range(1, min(6, LOOKBACK + 1)):
        j = idx - lag
        feats[f"ret_cc_lag{lag}"] = series[j]["ret_cc"]
        feats[f"ret_gap_lag{lag}"] = series[j]["ret_gap"]

    # Rolling volatility (from lookback window)
    recent_rets = [series[idx - k]["ret_cc"] for k in range(1, min(21, idx) + 1)]
    if len(recent_rets) >= 5:
        feats["vol_20d"] = float(np.std(recent_rets[:20])) if len(recent_rets) >= 20 else float(np.std(recent_rets))
    else:
        feats["vol_20d"] = 0.0

    # Momentum
    feats["momentum_5d"] = sum(series[idx - k]["ret_cc"] for k in range(1, min(6, idx) + 1))
    feats["momentum_10d"] = sum(series[idx - k]["ret_cc"] for k in range(1, min(11, idx) + 1))

    # Range (high-low as % of close)
    feats["range_pct_lag1"] = (series[idx - 1]["high"] - series[idx - 1]["low"]) / series[idx - 1]["close"]

    # Day of week
    feats["dow"] = datetime.strptime(series[idx]["date"], "%Y-%m-%d").weekday()

    return feats, y


def extract_metric_a_features(series, idx, target, cat_cols):
    """Extract metric A + news features for predicting day[idx]'s target.

    For gap target: uses lagged A-metric + today's gap-period news.
    For cc target: uses lagged A-metric + today's gap + cc news.
    Returns (feature_dict, target_value) or None if insufficient history.
    """
    if idx < LOOKBACK + 1:
        return None

    day = series[idx]

    # Target: actual return (model predicts return, converted to price later)
    if target == "gap":
        y = day["stock_gap"]  # gap return
    else:
        y = day["stock_cc"]   # close-to-close return

    feats = {}

    # Lagged metric A features (lookback window)
    for lag in range(1, LOOKBACK + 1):
        j = idx - lag
        feats[f"A_gap_lag{lag}"] = series[j]["A_gap"]
        feats[f"A_cc_lag{lag}"] = series[j]["A_cc"]
        feats[f"abs_A_cc_lag{lag}"] = abs(series[j]["A_cc"])
        if lag <= 5:
            feats[f"zv_lag{lag}"] = series[j]["zv"]
            feats[f"stock_cc_lag{lag}"] = series[j]["stock_cc"]

    # Beta (lagged 1 day)
    feats["beta_gap"] = series[idx - 1]["beta_gap"]
    feats["beta_cc"] = series[idx - 1]["beta_cc"]

    # Rolling |A_cc| stats (regime detectors)
    abs_a_5d = [abs(series[idx - k]["A_cc"]) for k in range(1, min(6, idx) + 1)]
    abs_a_20d = [abs(series[idx - k]["A_cc"]) for k in range(1, min(21, idx) + 1)]
    feats["mean_abs_A_cc_5d"] = float(np.mean(abs_a_5d))
    feats["max_abs_A_cc_5d"] = float(np.max(abs_a_5d))
    feats["mean_abs_A_cc_20d"] = float(np.mean(abs_a_20d))

    # Historical volatility (for range prediction)
    recent_rets = [series[idx - k]["stock_cc"] for k in range(1, min(21, idx) + 1)]
    feats["vol_20d"] = float(np.std(recent_rets)) if len(recent_rets) >= 5 else 0.0

    # S&P 500 return (lagged)
    feats["sp_cc_lag1"] = series[idx - 1]["sp_cc"]

    # News features — gap period (known before market open)
    feats["has_gap_news"] = day["has_gap_news"]
    feats["gap_distinct_events"] = day["gap_distinct_events"]
    for c in cat_cols:
        feats[f"gap_{c}"] = day.get(f"gap_{c}", 0)

    # For CC target, also include CC period news
    if target == "cc":
        feats["has_cc_news"] = day["has_cc_news"]
        feats["cc_distinct_events"] = day["cc_distinct_events"]
        for c in cat_cols:
            feats[f"cc_{c}"] = day.get(f"cc_{c}", 0)

    return feats, y


# ─── Sequence creation and splitting ────────────────────────────────────────

def make_price_sequences(series, target):
    """Create LSTM sequences from price series.

    Returns (X, y, dates, prev_closes) where:
    - X: array (n_samples, LOOKBACK, 5) — OHLCV per timestep
    - y: array (n_samples,) — target price (open or close)
    - dates: list of prediction dates
    - prev_closes: array for naive baseline
    """
    X, y, dates, prev_closes = [], [], [], []

    for idx in range(LOOKBACK + 1, len(series)):
        # Target
        if target == "gap":
            target_val = series[idx]["open"]
        else:
            target_val = series[idx]["close"]

        # Sequence: LOOKBACK days ending at idx-1
        seq = []
        for lag in range(LOOKBACK, 0, -1):
            j = idx - lag
            seq.append([
                series[j]["open"],
                series[j]["high"],
                series[j]["low"],
                series[j]["close"],
                series[j]["volume"],
            ])

        X.append(seq)
        y.append(target_val)
        dates.append(series[idx]["date"])
        prev_closes.append(series[idx - 1]["close"])

    return np.array(X, dtype=np.float64), np.array(y), dates, np.array(prev_closes)


def make_metric_a_sequences(series, target, cat_cols):
    """Create LSTM sequences from metric A series.

    Returns (X, y, dates, prev_closes, has_news) where:
    - X: array (n_samples, LOOKBACK, n_features)
    - y: array (n_samples,) — target return
    - dates: list of prediction dates
    - prev_closes: for price reconstruction
    - has_news: boolean array — whether prediction day has news
    """
    # Determine feature columns per timestep
    # For each timestep in lookback: A_gap, A_cc, zv
    # For the prediction day (appended): news features
    # We use a simpler approach: flat A features per timestep + news on last step

    X, y_vals, dates, prev_closes, has_news_arr = [], [], [], [], []

    for idx in range(LOOKBACK + 1, len(series)):
        day = series[idx]

        # Target return
        if target == "gap":
            y_val = day["stock_gap"]
            has_news_val = day["has_gap_news"]
        else:
            y_val = day["stock_cc"]
            has_news_val = day["has_gap_news"] or day["has_cc_news"]

        # Build sequence: for each lookback day, include A features
        seq = []
        for lag in range(LOOKBACK, 0, -1):
            j = idx - lag
            step_feats = [
                series[j]["A_gap"],
                series[j]["A_cc"],
                abs(series[j]["A_cc"]),
                series[j]["zv"],
                series[j]["stock_cc"],
                series[j]["stock_gap"],
            ]
            # Pad with zeros for news features (news only on prediction day)
            n_news = len(cat_cols) + 1  # cat cols + distinct_events
            step_feats.extend([0.0] * n_news)
            seq.append(step_feats)

        # Append prediction day's news as an extra timestep
        news_step = [
            day["A_gap"] if target == "cc" else 0.0,  # A_gap known if predicting cc
            0.0,  # A_cc not known yet
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        # Gap news (known before open)
        news_vals = [day["gap_distinct_events"]]
        for c in cat_cols:
            news_vals.append(day.get(f"gap_{c}", 0))
        news_step.extend(news_vals)
        seq.append(news_step)

        X.append(seq)
        y_vals.append(y_val)
        dates.append(day["date"])
        prev_closes.append(day["prev_close"])
        has_news_arr.append(has_news_val)

    # X shape: (n_samples, LOOKBACK + 1, n_features_per_step)
    return (
        np.array(X, dtype=np.float64),
        np.array(y_vals),
        dates,
        np.array(prev_closes),
        np.array(has_news_arr, dtype=bool),
    )


def make_flat_features(series, target, extractor, **kwargs):
    """Create flat feature matrices for Ridge/LightGBM.

    extractor: extract_price_features or extract_metric_a_features
    Returns (X, y, dates, prev_closes) + optional has_news for metric_a.
    """
    feat_dicts = []
    y_vals, dates, prev_closes = [], [], []
    has_news_arr = []

    for idx in range(LOOKBACK + 1, len(series)):
        result = extractor(series, idx, target, **kwargs)
        if result is None:
            continue
        feats, y_val = result
        feat_dicts.append(feats)
        y_vals.append(y_val)
        dates.append(series[idx]["date"])
        prev_closes.append(series[idx - 1]["close"] if "prev_close" not in series[idx]
                           else series[idx]["prev_close"])

        # Track news presence for metric_a
        if "has_gap_news" in series[idx]:
            if target == "gap":
                has_news_arr.append(bool(series[idx]["has_gap_news"]))
            else:
                has_news_arr.append(
                    bool(series[idx]["has_gap_news"] or series[idx]["has_cc_news"])
                )

    if not feat_dicts:
        return None

    # Convert to arrays with consistent column ordering
    columns = sorted(feat_dicts[0].keys())
    X = np.array([[fd[c] for c in columns] for fd in feat_dicts], dtype=np.float64)
    y = np.array(y_vals)
    prev_closes = np.array(prev_closes)

    result = {
        "X": X, "y": y, "dates": dates, "prev_closes": prev_closes,
        "columns": columns,
    }
    if has_news_arr:
        result["has_news"] = np.array(has_news_arr, dtype=bool)
    return result


def split_data(n, ratios=SPLIT_RATIOS):
    """Return (train_end, val_end) indices for chronological split.

    Data[0:train_end] = train
    Data[train_end:val_end] = val
    Data[val_end:] = test
    """
    train_end = int(n * ratios[0])
    val_end = int(n * (ratios[0] + ratios[1]))
    return train_end, val_end


def scale_splits(X_train, X_val, X_test):
    """Fit MinMaxScaler on train, transform all splits. Works for 2D and 3D arrays."""
    scaler = MinMaxScaler()
    original_shape = X_train.shape

    if X_train.ndim == 3:
        n_tr, seq, feat = X_train.shape
        scaler.fit(X_train.reshape(-1, feat))
        X_train = scaler.transform(X_train.reshape(-1, feat)).reshape(n_tr, seq, feat)
        if len(X_val) > 0:
            X_val = scaler.transform(X_val.reshape(-1, feat)).reshape(X_val.shape[0], seq, feat)
        X_test = scaler.transform(X_test.reshape(-1, feat)).reshape(X_test.shape[0], seq, feat)
    else:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        if len(X_val) > 0:
            X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, scaler


# ─── Evaluation helpers ─────────────────────────────────────────────────────

def compute_metrics(actuals, preds):
    """Compute MAE, MAPE, RMSE for price predictions."""
    actuals, preds = np.array(actuals), np.array(preds)
    errors = actuals - preds
    abs_errors = np.abs(errors)
    mae = float(np.mean(abs_errors))
    mape = float(np.mean(abs_errors / np.abs(actuals)) * 100)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}


def compute_range_metrics(actuals, centers, half_widths, sigma_levels=(1.0, 1.5, 2.0)):
    """Compute coverage, sharpness, and Winkler scores for range predictions.

    actuals: actual prices
    centers: predicted center prices
    half_widths: predicted half-width at 1 sigma
    """
    actuals = np.array(actuals)
    centers = np.array(centers)
    half_widths = np.array(half_widths)

    results = {}
    for k in sigma_levels:
        hw = half_widths * k
        lower = centers - hw
        upper = centers + hw
        in_range = (actuals >= lower) & (actuals <= upper)
        coverage = float(np.mean(in_range))
        width_pct = float(np.mean(2 * hw / actuals) * 100)

        # Winkler score: width + penalty for misses
        alpha = 2 * (1 - coverage) if coverage < 1 else 0.1
        width = upper - lower
        penalty = np.where(actuals < lower, (2.0 / max(alpha, 0.01)) * (lower - actuals), 0.0)
        penalty += np.where(actuals > upper, (2.0 / max(alpha, 0.01)) * (actuals - upper), 0.0)
        winkler = float(np.mean(width + penalty))

        results[f"{k:.1f}σ"] = {
            "coverage": coverage,
            "width_pct": width_pct,
            "winkler": winkler,
        }
    return results


def print_metrics_table(results, title=""):
    """Print a formatted table of per-ticker metrics."""
    if title:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")

    # Get all tickers and models
    tickers = sorted(set(r["ticker"] for r in results))
    models = sorted(set(r["model"] for r in results))

    # Header
    print(f"\n{'Model':<20}", end="")
    for t in tickers:
        print(f"  {t:>8}", end="")
    print(f"  {'Avg':>8}")
    print("-" * (20 + (len(tickers) + 1) * 10))

    for model in models:
        print(f"{model:<20}", end="")
        mapes = []
        for t in tickers:
            match = [r for r in results if r["model"] == model and r["ticker"] == t]
            if match:
                mape = match[0]["MAPE"]
                print(f"  {mape:>7.2f}%", end="")
                mapes.append(mape)
            else:
                print(f"  {'N/A':>8}", end="")
        if mapes:
            print(f"  {np.mean(mapes):>7.2f}%")
        else:
            print()
    print()


# ─── News → A prediction utilities ────────────────────────────────────────

def extract_news_features(series, idx, target, cat_cols):
    """Extract news category features for predicting day[idx]'s A score.

    Features: news category scores only (no lagged A, no price data).
    Target: A_gap or A_cc for that day.
    Returns (feature_dict, target_value) or None if insufficient history.
    """
    if idx < LOOKBACK + 1:
        return None

    day = series[idx]

    # Target: A score for this day
    if target == "gap":
        y = day["A_gap"]
    else:
        y = day["A_cc"]

    feats = {}

    # Gap-period news (known before market open — used for both targets)
    feats["has_gap_news"] = day["has_gap_news"]
    feats["gap_distinct_events"] = day["gap_distinct_events"]
    for c in cat_cols:
        feats[f"gap_{c}"] = day.get(f"gap_{c}", 0)

    # For CC target, also include CC-period news
    if target == "cc":
        feats["has_cc_news"] = day["has_cc_news"]
        feats["cc_distinct_events"] = day["cc_distinct_events"]
        for c in cat_cols:
            feats[f"cc_{c}"] = day.get(f"cc_{c}", 0)

    return feats, y


def a_to_price(predicted_a, alpha, beta, s0, sp_return, prev_close):
    """Convert predicted A score to predicted price via inverse market model.

    A = sign(zi) * zi²  →  zi = sign(A) * sqrt(|A|)
    predicted_return = alpha + beta * sp_return + zi * s0
    predicted_price = prev_close * (1 + predicted_return)

    Works with scalars or numpy arrays.
    """
    predicted_a = np.asarray(predicted_a, dtype=np.float64)
    zi = np.sign(predicted_a) * np.sqrt(np.abs(predicted_a))
    predicted_return = alpha + beta * sp_return + zi * s0
    predicted_price = np.asarray(prev_close) * (1 + predicted_return)
    return predicted_price, predicted_return


def compute_sp_stats(series, train_end):
    """Compute S&P 500 daily return stats from training data.

    Returns (avg_daily_return, std_daily_return) for both gap and cc.
    Used as baseline market assumption and for sensitivity analysis.
    """
    gap_rets = [series[i]["sp_gap"] for i in range(train_end) if "sp_gap" in series[i]]
    cc_rets = [series[i]["sp_cc"] for i in range(train_end) if "sp_cc" in series[i]]
    return {
        "sp_gap_avg": float(np.mean(gap_rets)) if gap_rets else 0.0,
        "sp_gap_std": float(np.std(gap_rets)) if gap_rets else 0.01,
        "sp_cc_avg": float(np.mean(cc_rets)) if cc_rets else 0.0,
        "sp_cc_std": float(np.std(cc_rets)) if cc_rets else 0.01,
    }

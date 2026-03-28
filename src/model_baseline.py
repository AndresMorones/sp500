"""Model 1: Price-only baseline for next-day return prediction.

Pooled across 7 tickers. Walk-forward expanding window with 3-day embargo.
Models: Naive (historical mean), Ridge, LASSO, LightGBM.
Monthly refit to prevent overfitting on small dataset.

CRITICAL: All features use ONLY data available before the prediction day.
- Lagged returns: t-1, t-2, ..., t-5 (known at t-1 close)
- A scores, zv, beta: use t-1 values (lagged 1 day)
- Range, volume: use t-1 values (lagged 1 day)
- Rolling stats (momentum, volatility): computed from t-1 and earlier

Literature:
- Gu, Kelly & Xiu (2020) "Empirical Asset Pricing via Machine Learning"
- Campbell & Thompson (2008) OOS R-squared with expanding historical mean
- Clark & West (2007) test for nested model comparison
- de Prado (2018) purged cross-validation with embargo
"""

import csv
import math
import os
import warnings
from datetime import datetime

import numpy as np
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
SCORES_FILE = os.path.join(OUT_DIR, "scores_output.csv")

TRAIN_MIN = 150       # minimum training days before first prediction
EMBARGO_DAYS = 3      # de Prado (2018): gap between train and test to prevent leakage
REFIT_EVERY = 21      # refit models monthly (~21 trading days), not daily
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Feature config
LAGS = [1, 2, 3, 4, 5]
VOL_WINDOW_SHORT = 20
VOL_WINDOW_LONG = 60
MOMENTUM_WINDOWS = [5, 20]

# --- 1. Load raw data ---

print("Loading data...")

price_rows = {}  # (ticker, date) -> {open, high, low, close, volume}
with open(os.path.join(RAW_DIR, "price.csv")) as f:
    for row in csv.DictReader(f):
        key = (row["ticker"], row["date"])
        price_rows[key] = {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": int(float(row["volume"])),
        }

sp500 = {}  # date -> {open, close}
with open(os.path.join(RAW_DIR, "S&P 500 Historical Data.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        dt = datetime.strptime(row["Date"], "%m/%d/%Y")
        date_str = dt.strftime("%Y-%m-%d")
        sp500[date_str] = {
            "open": float(row["Open"].replace(",", "")),
            "close": float(row["Price"].replace(",", "")),
        }

scores_data = {}  # (ticker, date) -> {A_gap, A_cc, zv, beta_cc, beta_gap}
with open(SCORES_FILE, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["ticker"], row["date"])
        scores_data[key] = {
            "A_gap": float(row["A_gap"]),
            "A_cc": float(row["A_cc"]),
            "zv": float(row["zv"]),
            "beta_cc": float(row["beta_cc"]),
            "beta_gap": float(row["beta_gap"]),
        }

dates_per_ticker = {}
for (ticker, date) in price_rows:
    if ticker not in dates_per_ticker:
        dates_per_ticker[ticker] = []
    dates_per_ticker[ticker].append(date)
for t in dates_per_ticker:
    dates_per_ticker[t].sort()

all_dates = sorted(set(d for dates in dates_per_ticker.values() for d in dates))
print(f"  {len(all_dates)} trading days, {len(TICKERS)} tickers")

# --- 2. Build per-ticker time series ---

def build_ticker_series(ticker):
    """Build time series for a ticker. Returns list of day dicts sorted by date."""
    dates = dates_per_ticker[ticker]
    series = []
    for i, date in enumerate(dates):
        d = price_rows[(ticker, date)]
        day = {
            "date": date,
            "ticker": ticker,
            "open": d["open"],
            "high": d["high"],
            "low": d["low"],
            "close": d["close"],
            "volume": d["volume"],
            "dow": datetime.strptime(date, "%Y-%m-%d").weekday(),
        }

        if i > 0:
            prev = price_rows[(ticker, dates[i - 1])]
            day["ret_cc"] = (d["close"] - prev["close"]) / prev["close"]
            day["ret_gap"] = (d["open"] - prev["close"]) / prev["close"]
        else:
            day["ret_cc"] = 0.0
            day["ret_gap"] = 0.0

        if date in sp500 and i > 0 and dates[i - 1] in sp500:
            sp_prev = sp500[dates[i - 1]]["close"]
            day["sp_ret_cc"] = (sp500[date]["close"] - sp_prev) / sp_prev
        else:
            day["sp_ret_cc"] = 0.0

        # Intraday range
        day["range_pct"] = (d["high"] - d["low"]) / d["close"] if d["close"] > 0 else 0.0

        # Scores from pipeline (these are SAME-DAY values — must be lagged in features)
        sc = scores_data.get((ticker, date))
        day["A_cc"] = sc["A_cc"] if sc else 0.0
        day["A_gap"] = sc["A_gap"] if sc else 0.0
        day["zv"] = sc["zv"] if sc else 0.0
        day["beta_cc"] = sc["beta_cc"] if sc else 1.0

        series.append(day)
    return series


def compute_features(series, idx):
    """Compute feature vector for predicting day[idx]'s return.

    CRITICAL: All features use data from day[idx-1] and earlier.
    We are predicting what happens on day idx using information
    available at the close of day idx-1.
    """
    if idx < max(VOL_WINDOW_LONG, max(LAGS)) + 1:
        return None

    # prev = idx-1 (last known day), day = idx (prediction target)
    prev = series[idx - 1]
    feat = {}

    # Lagged returns: ret at t-1, t-2, ..., t-5 (all known at t-1 close)
    for lag in LAGS:
        j = idx - lag
        if j >= 0:
            feat[f"ret_cc_{lag}"] = series[j]["ret_cc"]
            feat[f"ret_gap_{lag}"] = series[j]["ret_gap"]
            feat[f"sp_ret_cc_{lag}"] = series[j]["sp_ret_cc"]
        else:
            feat[f"ret_cc_{lag}"] = 0.0
            feat[f"ret_gap_{lag}"] = 0.0
            feat[f"sp_ret_cc_{lag}"] = 0.0

    # Momentum: cumulative return ending at t-1
    for w in MOMENTUM_WINDOWS:
        cum = 1.0
        for j in range(1, w + 1):
            k = idx - j
            if k >= 0:
                cum *= (1 + series[k]["ret_cc"])
        feat[f"momentum_{w}d"] = cum - 1.0

    # Volatility: rolling std of cc returns ending at t-1
    cc_returns_20 = [series[idx - j]["ret_cc"] for j in range(1, VOL_WINDOW_SHORT + 1) if idx - j >= 0]
    cc_returns_60 = [series[idx - j]["ret_cc"] for j in range(1, VOL_WINDOW_LONG + 1) if idx - j >= 0]

    vol_20 = np.std(cc_returns_20, ddof=1) if len(cc_returns_20) > 1 else 0.01
    vol_60 = np.std(cc_returns_60, ddof=1) if len(cc_returns_60) > 1 else 0.01
    feat["vol_20d"] = vol_20
    feat["vol_ratio"] = vol_20 / max(vol_60, 1e-8)

    # Range features: LAGGED (t-1 range, t-1 to t-5 avg range)
    feat["range_pct_1"] = prev["range_pct"]
    range_5d = [series[idx - j]["range_pct"] for j in range(1, 6) if idx - j >= 0]
    feat["range_pct_5d_avg"] = np.mean(range_5d)

    # Volume features: LAGGED
    vol_20d_avg = np.mean([series[idx - j]["volume"] for j in range(1, 21) if idx - j >= 0])
    feat["vol_z_1"] = prev["zv"]  # yesterday's volume z-score
    vol_5d_avg = np.mean([series[idx - j]["volume"] for j in range(1, 6) if idx - j >= 0])
    feat["vol_ratio_5d"] = vol_5d_avg / max(vol_20d_avg, 1) if vol_20d_avg > 0 else 1.0

    # Market-relative: ALL LAGGED by 1 day
    feat["beta_cc_1"] = prev["beta_cc"]
    feat["A_cc_1"] = prev["A_cc"]      # yesterday's A_cc, NOT today's
    feat["A_gap_1"] = prev["A_gap"]    # yesterday's A_gap, NOT today's

    # Day of week (of the prediction day — this is known in advance)
    feat["dow"] = series[idx]["dow"]

    # Ticker encoding
    for t in TICKERS:
        feat[f"ticker_{t}"] = 1.0 if series[idx]["ticker"] == t else 0.0

    return feat


# --- 3. Build feature matrix ---

print("Building feature matrix...")

feature_names = None
all_samples = []

for ticker in TICKERS:
    series = build_ticker_series(ticker)
    for idx in range(1, len(series)):
        feat = compute_features(series, idx)
        if feat is None:
            continue
        target_cc = series[idx]["ret_cc"]
        prev_close = series[idx - 1]["close"]
        actual_close = series[idx]["close"]

        if feature_names is None:
            feature_names = sorted(feat.keys())

        all_samples.append({
            "date": series[idx]["date"],
            "ticker": ticker,
            "features": feat,
            "target_cc": target_cc,
            "prev_close": prev_close,
            "actual_close": actual_close,
        })

all_samples.sort(key=lambda s: (s["date"], s["ticker"]))

unique_dates = sorted(set(s["date"] for s in all_samples))
date_to_idx = {d: i for i, d in enumerate(unique_dates)}

print(f"  {len(all_samples)} total samples, {len(feature_names)} features, {len(unique_dates)} unique dates")
print(f"  Date range: {unique_dates[0]} to {unique_dates[-1]}")
print(f"  Features: {feature_names}")

X_all = np.array([[s["features"][f] for f in feature_names] for s in all_samples])
y_all = np.array([s["target_cc"] for s in all_samples])
dates_all = np.array([date_to_idx[s["date"]] for s in all_samples])

# --- 4. Walk-forward evaluation with embargo and monthly refit ---

print(f"\nWalk-forward evaluation (min train={TRAIN_MIN} dates, embargo={EMBARGO_DAYS}, refit every {REFIT_EVERY} days)...")

test_date_indices = sorted(set(di for di in dates_all if di >= TRAIN_MIN))

predictions = {name: [] for name in ["naive", "ridge", "lasso", "lgbm"]}
actuals = []
metadata = []
expanding_means = []  # for OOS R² (Campbell & Thompson 2008)

# Track current fitted models for monthly refit
current_models = {}
last_refit_di = -999

for test_di in test_date_indices:
    # Embargo: train on dates < (test_di - EMBARGO_DAYS)
    train_cutoff = test_di - EMBARGO_DAYS
    train_mask = dates_all <= train_cutoff
    test_day_mask = dates_all == test_di

    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    X_test = X_all[test_day_mask]
    y_test = y_all[test_day_mask]

    if len(X_train) < 50 or len(X_test) == 0:
        continue

    # Expanding historical mean for naive baseline (Campbell & Thompson 2008)
    hist_mean = np.mean(y_train)

    # Refit models monthly (or on first iteration)
    need_refit = (test_di - last_refit_di) >= REFIT_EVERY or not current_models

    if need_refit:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)

        # Ridge with CV over alpha
        ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
        ridge.fit(X_train_s, y_train)

        # LASSO with CV over alpha
        lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1], max_iter=10000, cv=5)
        lasso.fit(X_train_s, y_train)

        # LightGBM
        lgb_train_ds = lgb.Dataset(X_train_s, y_train, free_raw_data=False)
        lgb_params = {
            "objective": "regression",
            "metric": "mae",
            "max_depth": 3,
            "num_leaves": 8,
            "min_child_samples": 30,
            "learning_rate": 0.05,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "seed": 42,
        }
        lgb_model = lgb.train(lgb_params, lgb_train_ds, num_boost_round=150,
                              callbacks=[lgb.log_evaluation(0)])

        current_models = {
            "scaler": scaler,
            "ridge": ridge,
            "lasso": lasso,
            "lgbm": lgb_model,
        }
        last_refit_di = test_di

    # Predict using current models
    scaler = current_models["scaler"]
    X_test_s = scaler.transform(X_test)

    pred_naive = np.full(len(X_test), hist_mean)
    pred_ridge = current_models["ridge"].predict(X_test_s)
    pred_lasso = current_models["lasso"].predict(X_test_s)
    pred_lgbm = current_models["lgbm"].predict(X_test_s)

    for i in range(len(X_test)):
        sample_idx = np.where(test_day_mask)[0][i]
        s = all_samples[sample_idx]
        predictions["naive"].append(pred_naive[i])
        predictions["ridge"].append(pred_ridge[i])
        predictions["lasso"].append(pred_lasso[i])
        predictions["lgbm"].append(pred_lgbm[i])
        actuals.append(y_test[i])
        expanding_means.append(hist_mean)
        metadata.append((s["date"], s["ticker"], s["prev_close"], s["actual_close"]))

actuals = np.array(actuals)
expanding_means = np.array(expanding_means)
for k in predictions:
    predictions[k] = np.array(predictions[k])

print(f"  {len(actuals)} out-of-sample predictions ({len(test_date_indices)} test dates)")

# --- 5. Evaluation metrics ---

def oos_r2(pred, actual, hist_means):
    """Campbell & Thompson (2008) OOS R-squared.
    Benchmark is the expanding historical mean, not the global mean.
    R²_OOS > 0 means the model beats the naive historical average."""
    ss_res = np.sum((actual - pred) ** 2)
    ss_bench = np.sum((actual - hist_means) ** 2)
    return 1 - ss_res / ss_bench if ss_bench > 0 else 0


def clark_west_test(pred_model, pred_bench, actual):
    """Clark & West (2007) test for nested model comparison.
    Tests H0: model has no predictive advantage over benchmark.
    Returns test statistic and approximate p-value.
    Positive t-stat with p<0.05 means model significantly beats benchmark."""
    e_bench = actual - pred_bench
    e_model = actual - pred_model
    adj = (pred_bench - pred_model) ** 2
    f_t = e_bench ** 2 - (e_model ** 2 - adj)
    t_stat = np.mean(f_t) / (np.std(f_t, ddof=1) / np.sqrt(len(f_t)))
    # One-sided test: model beats benchmark
    from scipy.stats import norm
    p_value = 1 - norm.cdf(t_stat)
    return t_stat, p_value


def evaluate(pred, actual, meta, hist_means):
    """Compute evaluation metrics for return predictions."""
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))

    # Directional accuracy (exclude near-zero predictions)
    correct_dir = np.sum(np.sign(pred) == np.sign(actual))
    dir_acc = correct_dir / len(actual) * 100

    # OOS R² (Campbell & Thompson 2008)
    r2_oos = oos_r2(pred, actual, hist_means)

    # Standard R²
    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # Price-space metrics
    price_errors = []
    for i, (date, ticker, prev_close, actual_close) in enumerate(meta):
        pred_close = prev_close * (1 + pred[i])
        price_errors.append(abs(pred_close - actual_close))
    price_mae = np.mean(price_errors)
    price_rmse = np.sqrt(np.mean(np.array(price_errors) ** 2))

    return {
        "MAE_ret": mae,
        "RMSE_ret": rmse,
        "Dir_Acc": dir_acc,
        "R2": r2,
        "R2_OOS": r2_oos,
        "MAE_price": price_mae,
        "RMSE_price": price_rmse,
    }


# --- 6. Print results ---

print(f"\n{'=' * 110}")
print("MODEL COMPARISON -- Out-of-sample (walk-forward, 3-day embargo, monthly refit)")
print(f"{'=' * 110}")

header = (f"{'Model':<12} {'MAE(ret)':>10} {'RMSE(ret)':>10} {'Dir Acc%':>10} "
          f"{'R2':>8} {'R2_OOS':>8} {'MAE($)':>10} {'RMSE($)':>10}")
print(header)
print("-" * len(header))

model_results = {}
for model_name in ["naive", "ridge", "lasso", "lgbm"]:
    res = evaluate(predictions[model_name], actuals, metadata, expanding_means)
    model_results[model_name] = res
    print(f"{model_name:<12} {res['MAE_ret']:>10.6f} {res['RMSE_ret']:>10.6f} "
          f"{res['Dir_Acc']:>9.1f}% {res['R2']:>8.4f} {res['R2_OOS']:>8.4f} "
          f"{res['MAE_price']:>9.2f}$ {res['RMSE_price']:>9.2f}$")

# Clark-West tests: each model vs naive
print(f"\n  Clark-West tests (model vs naive historical mean):")
for model_name in ["ridge", "lasso", "lgbm"]:
    t_stat, p_val = clark_west_test(predictions[model_name], predictions["naive"], actuals)
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
    print(f"    {model_name:>8} vs naive: t={t_stat:>6.3f}, p={p_val:.4f} {sig}")

# --- 7. Per-ticker breakdown ---

print(f"\n{'=' * 110}")
print("PER-TICKER BREAKDOWN")
print(f"{'=' * 110}")

for ticker in TICKERS:
    ticker_mask = np.array([m[1] == ticker for m in metadata])
    if ticker_mask.sum() == 0:
        continue

    print(f"\n  {ticker} ({ticker_mask.sum()} predictions):")
    header = f"  {'Model':<12} {'MAE(ret)':>10} {'Dir Acc%':>10} {'R2_OOS':>8} {'MAE($)':>10}"
    print(header)

    for model_name in ["naive", "ridge", "lasso", "lgbm"]:
        pred_t = predictions[model_name][ticker_mask]
        act_t = actuals[ticker_mask]
        hm_t = expanding_means[ticker_mask]
        meta_t = [m for m, mask in zip(metadata, ticker_mask) if mask]
        res = evaluate(pred_t, act_t, meta_t, hm_t)
        print(f"  {model_name:<12} {res['MAE_ret']:>10.6f} {res['Dir_Acc']:>9.1f}% "
              f"{res['R2_OOS']:>8.4f} {res['MAE_price']:>9.2f}$")

# --- 8. Feature importance ---

print(f"\n{'=' * 110}")
print("FEATURE IMPORTANCE (final refit on all training data)")
print(f"{'=' * 110}")

scaler_final = StandardScaler()
X_all_s = scaler_final.fit_transform(X_all)

# Ridge
ridge_final = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
ridge_final.fit(X_all_s, y_all)
ridge_coefs = list(zip(feature_names, ridge_final.coef_))
ridge_coefs.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Ridge (best alpha={ridge_final.alpha_:.2f}) -- Top 15 features by |coefficient|:")
for i, (name, coef) in enumerate(ridge_coefs[:15]):
    print(f"    {i+1:>2}. {name:<20} {coef:>+10.6f}")

# LASSO
lasso_final = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1], max_iter=10000, cv=5)
lasso_final.fit(X_all_s, y_all)
lasso_coefs = [(n, c) for n, c in zip(feature_names, lasso_final.coef_) if abs(c) > 1e-8]
lasso_coefs.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n  LASSO (best alpha={lasso_final.alpha_:.4f}) -- Non-zero features ({len(lasso_coefs)}/{len(feature_names)}):")
for i, (name, coef) in enumerate(lasso_coefs[:15]):
    print(f"    {i+1:>2}. {name:<20} {coef:>+10.6f}")

# LightGBM
lgb_final_ds = lgb.Dataset(X_all_s, y_all)
lgb_params_final = {
    "objective": "regression",
    "metric": "mae",
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 30,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": 42,
}
lgb_final = lgb.train(lgb_params_final, lgb_final_ds, num_boost_round=150,
                       callbacks=[lgb.log_evaluation(0)])
lgb_imp = list(zip(feature_names, lgb_final.feature_importance(importance_type="gain")))
lgb_imp.sort(key=lambda x: x[1], reverse=True)

print("\n  LightGBM -- Top 15 features by gain:")
for i, (name, imp) in enumerate(lgb_imp[:15]):
    print(f"    {i+1:>2}. {name:<20} {imp:>10.1f}")

# --- 9. Save predictions ---

out_path = os.path.join(OUT_DIR, "baseline_predictions.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "ticker", "prev_close", "actual_close", "actual_ret",
                     "pred_naive", "pred_ridge", "pred_lasso", "pred_lgbm",
                     "expanding_mean"])
    for i, (date, ticker, prev_close, actual_close) in enumerate(metadata):
        writer.writerow([
            date, ticker, f"{prev_close:.4f}", f"{actual_close:.4f}",
            f"{actuals[i]:.6f}",
            f"{predictions['naive'][i]:.6f}",
            f"{predictions['ridge'][i]:.6f}",
            f"{predictions['lasso'][i]:.6f}",
            f"{predictions['lgbm'][i]:.6f}",
            f"{expanding_means[i]:.6f}",
        ])

print(f"\nPredictions saved to {out_path}")
print(f"Total out-of-sample predictions: {len(actuals)}")

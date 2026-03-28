"""Model 1: Price-only baseline for excess return prediction.

Goal: predict the magnitude of stock-specific excess returns (stock - SP500),
not direction. Evaluation focuses on return prediction accuracy (MAE, RMSE,
R²_OOS) and reconstructed price accuracy ($MAE), not directional accuracy.

Pooled across 7 tickers. Walk-forward expanding window with embargo.
Models: Naive (expanding mean), Ridge (MSE), Ridge (Huber), LASSO, LightGBM (Huber).
Monthly refit to prevent overfitting on small dataset.

Three prediction targets (all excess returns = stock - SP500):
1. cc_excess: next-day close-to-close excess return (embargo=3)
2. gap_excess: next-day gap excess return, prev close -> open (embargo=3)
3. cum3d_excess: cumulative 3-day excess return (embargo=5, prevents overlap)

Price reconstruction (cc_excess, gap_excess):
  pred_close = prev_close * (1 + pred_excess + sp_ret)
  This converts return predictions back to dollar prices for practical use.

Huber loss (epsilon=1.35): robust to fat-tailed return distributions. Stock returns
follow power-law tails — a single 8% drop generates 16x the squared error of a 2% drop.
MSE-trained models distort coefficients chasing these rare extremes. Huber switches from
quadratic to linear loss above the threshold, focusing on the predictable ~95% of days.

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
- kristina969 GKX replication: Huber loss for fat-tail robustness
"""

import csv
import math
import os
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import RidgeCV, LassoCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
SCORES_FILE = os.path.join(OUT_DIR, "scores_output.csv")

TRAIN_MIN = 150       # minimum training days before first prediction
REFIT_EVERY = 21      # refit models monthly (~21 trading days), not daily
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Target configs: (name, target_key, embargo_days, horizon)
# horizon = forecast steps ahead. Used for Newey-West lags (h-1) to correct
# overlapping observation bias (Hansen & Hodrick 1980).
TARGETS = [
    ("cc_excess",   "target_cc_excess",   3, 1),   # next-day close-to-close excess
    ("gap_excess",  "target_gap_excess",  3, 1),   # next-day gap (prev close -> open) excess
    ("cum3d_excess","target_cum3d_excess", 5, 3),   # cumulative 3-day excess (wider embargo)
]

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
            day["sp_ret_gap"] = (sp500[date]["open"] - sp_prev) / sp_prev
        else:
            day["sp_ret_cc"] = 0.0
            day["sp_ret_gap"] = 0.0

        # Excess returns: stock - market
        day["ret_excess"] = day["ret_cc"] - day["sp_ret_cc"]
        day["ret_gap_excess"] = day["ret_gap"] - day["sp_ret_gap"]

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
            feat[f"ret_excess_{lag}"] = series[j]["ret_excess"]
        else:
            feat[f"ret_cc_{lag}"] = 0.0
            feat[f"ret_gap_{lag}"] = 0.0
            feat[f"sp_ret_cc_{lag}"] = 0.0
            feat[f"ret_excess_{lag}"] = 0.0

    # Momentum: cumulative return ending at t-1
    for w in MOMENTUM_WINDOWS:
        cum = 1.0
        cum_excess = 1.0
        for j in range(1, w + 1):
            k = idx - j
            if k >= 0:
                cum *= (1 + series[k]["ret_cc"])
                cum_excess *= (1 + series[k]["ret_excess"])
        feat[f"momentum_{w}d"] = cum - 1.0
        feat[f"momentum_excess_{w}d"] = cum_excess - 1.0

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

        if feature_names is None:
            feature_names = sorted(feat.keys())

        # Cumulative 3-day excess return: sum of ret_excess[t], ret_excess[t+1], ret_excess[t+2]
        cum3d = None
        if idx + 2 < len(series):
            cum3d = sum(series[idx + j]["ret_excess"] for j in range(3))

        all_samples.append({
            "date": series[idx]["date"],
            "ticker": ticker,
            "features": feat,
            "target_cc_excess": series[idx]["ret_excess"],
            "target_gap_excess": series[idx]["ret_gap_excess"],
            "target_cum3d_excess": cum3d,
            "target_cc": series[idx]["ret_cc"],
            "target_gap": series[idx]["ret_gap"],
            "sp_ret_cc": series[idx]["sp_ret_cc"],
            "sp_ret_gap": series[idx]["sp_ret_gap"],
            "prev_close": series[idx - 1]["close"],
            "actual_close": series[idx]["close"],
            "actual_open": series[idx]["open"],
        })

all_samples.sort(key=lambda s: (s["date"], s["ticker"]))

unique_dates = sorted(set(s["date"] for s in all_samples))
date_to_idx = {d: i for i, d in enumerate(unique_dates)}

print(f"  {len(all_samples)} total samples, {len(feature_names)} features, {len(unique_dates)} unique dates")
print(f"  Date range: {unique_dates[0]} to {unique_dates[-1]}")
print(f"  Features: {feature_names}")

X_all = np.array([[s["features"][f] for f in feature_names] for s in all_samples])
dates_all = np.array([date_to_idx[s["date"]] for s in all_samples])

# --- 4. Walk-forward evaluation with embargo and monthly refit ---

MODEL_NAMES = ["naive", "ridge", "ridge_huber", "lasso", "lgbm"]


def run_walk_forward(target_name, target_key, embargo_days):
    """Run walk-forward for a single target. Returns results dict."""
    # Build target array, skipping samples where target is None (cum3d at end of series)
    valid_mask = np.array([s[target_key] is not None for s in all_samples])
    y_target = np.array([s[target_key] if s[target_key] is not None else 0.0 for s in all_samples])

    test_date_indices = sorted(set(di for di in dates_all if di >= TRAIN_MIN))

    predictions = {name: [] for name in MODEL_NAMES}
    actuals_target = []
    sample_indices = []
    expanding_means = []

    current_models = {}
    last_refit_di = -999

    for test_di in test_date_indices:
        train_cutoff = test_di - embargo_days
        train_mask = (dates_all <= train_cutoff) & valid_mask
        test_day_mask = (dates_all == test_di) & valid_mask

        X_train = X_all[train_mask]
        y_train = y_target[train_mask]
        X_test = X_all[test_day_mask]
        y_test = y_target[test_day_mask]

        if len(X_train) < 50 or len(X_test) == 0:
            continue

        hist_mean = np.mean(y_train)

        need_refit = (test_di - last_refit_di) >= REFIT_EVERY or not current_models

        if need_refit:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)

            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
            ridge.fit(X_train_s, y_train)

            ridge_huber = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
            ridge_huber.fit(X_train_s, y_train)

            lasso = LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1], max_iter=10000, cv=5)
            lasso.fit(X_train_s, y_train)

            lgb_train_ds = lgb.Dataset(X_train_s, y_train, free_raw_data=False)
            lgb_params = {
                "objective": "huber",
                "huber_delta": 1.35,
                "metric": "huber",
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
                "ridge_huber": ridge_huber,
                "lasso": lasso,
                "lgbm": lgb_model,
            }
            last_refit_di = test_di

        scaler = current_models["scaler"]
        X_test_s = scaler.transform(X_test)

        pred_naive = np.full(len(X_test), hist_mean)
        pred_ridge = current_models["ridge"].predict(X_test_s)
        pred_ridge_huber = current_models["ridge_huber"].predict(X_test_s)
        pred_lasso = current_models["lasso"].predict(X_test_s)
        pred_lgbm = current_models["lgbm"].predict(X_test_s)

        for i in range(len(X_test)):
            si = np.where(test_day_mask)[0][i]
            predictions["naive"].append(pred_naive[i])
            predictions["ridge"].append(pred_ridge[i])
            predictions["ridge_huber"].append(pred_ridge_huber[i])
            predictions["lasso"].append(pred_lasso[i])
            predictions["lgbm"].append(pred_lgbm[i])
            actuals_target.append(y_test[i])
            sample_indices.append(si)
            expanding_means.append(hist_mean)

    actuals_target = np.array(actuals_target)
    expanding_means = np.array(expanding_means)
    for k in predictions:
        predictions[k] = np.array(predictions[k])

    return {
        "target_name": target_name,
        "target_key": target_key,
        "embargo_days": embargo_days,
        "predictions": predictions,
        "actuals_target": actuals_target,
        "sample_indices": sample_indices,
        "expanding_means": expanding_means,
    }


print(f"\nWalk-forward evaluation (min train={TRAIN_MIN}, refit every {REFIT_EVERY} days)")
print(f"  Huber loss for fat-tail robustness. All targets are excess returns (stock - SP500).")

target_results = {}
for tgt_name, tgt_key, tgt_embargo, tgt_horizon in TARGETS:
    print(f"\n  Running target: {tgt_name} (embargo={tgt_embargo}, horizon={tgt_horizon})...")
    res = run_walk_forward(tgt_name, tgt_key, tgt_embargo)
    res["horizon"] = tgt_horizon
    target_results[tgt_name] = res
    print(f"    {len(res['actuals_target'])} OOS predictions")

# --- 5. Evaluation metrics ---

def oos_r2(pred, actual, hist_means):
    """Campbell & Thompson (2008) OOS R-squared.
    Benchmark is the expanding historical mean, not the global mean.
    R²_OOS > 0 means the model beats the naive historical average."""
    ss_res = np.sum((actual - pred) ** 2)
    ss_bench = np.sum((actual - hist_means) ** 2)
    return 1 - ss_res / ss_bench if ss_bench > 0 else 0


def newey_west_se(x, n_lags):
    """Newey-West (1987) HAC standard error for the mean of x.
    Corrects for serial correlation up to n_lags (use h-1 for h-step forecasts).
    Hansen & Hodrick (1980): overlapping multi-period returns create MA(h-1)
    structure in forecast errors — standard SEs are too small without this."""
    n = len(x)
    x_dm = x - np.mean(x)
    # Autocovariance at lag 0
    gamma_0 = np.sum(x_dm ** 2) / n
    # Bartlett kernel weighted autocovariances
    weighted_sum = 0.0
    for lag in range(1, n_lags + 1):
        weight = 1 - lag / (n_lags + 1)  # Bartlett kernel
        gamma_lag = np.sum(x_dm[lag:] * x_dm[:-lag]) / n
        weighted_sum += 2 * weight * gamma_lag
    var_hat = (gamma_0 + weighted_sum) / n
    return np.sqrt(max(var_hat, 1e-16))


def clark_west_test(pred_model, pred_bench, actual, nw_lags=0):
    """Clark & West (2007) test for nested model comparison.
    Tests H0: model has no predictive advantage over benchmark.
    Returns test statistic and approximate p-value.
    Positive t-stat with p<0.05 means model significantly beats benchmark.

    nw_lags: Newey-West lags for HAC standard errors. Use h-1 for h-step
    ahead forecasts to correct for overlapping observation bias
    (Hansen & Hodrick 1980). 0 = no correction (iid errors)."""
    e_bench = actual - pred_bench
    e_model = actual - pred_model
    adj = (pred_bench - pred_model) ** 2
    f_t = e_bench ** 2 - (e_model ** 2 - adj)
    if nw_lags > 0:
        se = newey_west_se(f_t, nw_lags)
    else:
        se = np.std(f_t, ddof=1) / np.sqrt(len(f_t))
    t_stat = np.mean(f_t) / se
    p_value = 1 - norm.cdf(t_stat)
    return t_stat, p_value


def evaluate_target(res):
    """Evaluate a single target's results. Returns dict of per-model metrics.

    Primary metrics (return prediction quality):
    - MAE_bps: mean absolute error in basis points
    - RMSE_bps: root mean squared error in basis points
    - R2_OOS: Campbell & Thompson (2008) out-of-sample R-squared vs naive mean
    - Price_MAE: mean absolute error in $ (cc_excess/gap_excess only)
    - Bias_bps: mean prediction error (should be ~0)
    """
    preds = res["predictions"]
    actuals = res["actuals_target"]
    hm = res["expanding_means"]
    sidxs = res["sample_indices"]
    target_name = res["target_name"]

    # Price reconstruction for cc_excess and gap_excess
    can_reconstruct_price = target_name in ("cc_excess", "gap_excess")
    if can_reconstruct_price:
        prev_closes = np.array([all_samples[si]["prev_close"] for si in sidxs])
        if target_name == "cc_excess":
            sp_rets = np.array([all_samples[si]["sp_ret_cc"] for si in sidxs])
            actual_prices = np.array([all_samples[si]["actual_close"] for si in sidxs])
        else:  # gap_excess
            sp_rets = np.array([all_samples[si]["sp_ret_gap"] for si in sidxs])
            actual_prices = np.array([all_samples[si]["actual_open"] for si in sidxs])

    metrics = {}
    for model_name in MODEL_NAMES:
        pred = preds[model_name]
        mae = np.mean(np.abs(pred - actuals))
        rmse = np.sqrt(np.mean((pred - actuals) ** 2))
        r2 = oos_r2(pred, actuals, hm)
        bias = np.mean(pred - actuals)

        m = {
            "MAE_bps": mae * 10000,
            "RMSE_bps": rmse * 10000,
            "R2_OOS": r2,
            "Bias_bps": bias * 10000,
        }

        if can_reconstruct_price:
            pred_prices = prev_closes * (1 + pred + sp_rets)
            price_errors = np.abs(pred_prices - actual_prices)
            m["Price_MAE"] = np.mean(price_errors)
            m["Price_MAPE"] = np.mean(price_errors / actual_prices) * 100

        metrics[model_name] = m

    return metrics


# --- 6. Print results per target ---

for tgt_name, tgt_key, tgt_embargo, tgt_horizon in TARGETS:
    res = target_results[tgt_name]
    preds = res["predictions"]
    actuals = res["actuals_target"]
    hm = res["expanding_means"]
    sidxs = res["sample_indices"]
    n_preds = len(actuals)
    nw_lags = tgt_horizon - 1  # Hansen & Hodrick (1980): h-1 lags for h-step forecasts

    print(f"\n{'=' * 100}")
    print(f"TARGET: {tgt_name} (embargo={tgt_embargo}, horizon={tgt_horizon}, {n_preds} OOS predictions)")
    if nw_lags > 0:
        print(f"  Newey-West correction: {nw_lags} lags (overlapping {tgt_horizon}-day returns)")
    print(f"{'=' * 100}")

    metrics = evaluate_target(res)
    has_price = "Price_MAE" in metrics["naive"]

    if has_price:
        header = f"{'Model':<16} {'MAE(bps)':>10} {'RMSE(bps)':>11} {'R2_OOS':>8} {'Bias(bps)':>10} {'$MAE':>8} {'MAPE%':>7}"
    else:
        header = f"{'Model':<16} {'MAE(bps)':>10} {'RMSE(bps)':>11} {'R2_OOS':>8} {'Bias(bps)':>10}"
    print(header)
    print("-" * len(header))

    for model_name in MODEL_NAMES:
        m = metrics[model_name]
        line = (f"{model_name:<16} {m['MAE_bps']:>10.1f} {m['RMSE_bps']:>11.1f} "
                f"{m['R2_OOS']:>8.4f} {m['Bias_bps']:>+10.2f}")
        if has_price:
            line += f" {m['Price_MAE']:>8.2f} {m['Price_MAPE']:>6.2f}%"
        print(line)

    # Clark-West tests with Newey-West correction for multi-step targets
    nw_label = f", NW({nw_lags})" if nw_lags > 0 else ""
    print(f"\n  Clark-West tests (vs naive{nw_label}):")
    for model_name in MODEL_NAMES:
        if model_name == "naive":
            continue
        t_stat, p_val = clark_west_test(preds[model_name], preds["naive"], actuals, nw_lags=nw_lags)
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"    {model_name:>14} vs naive: t={t_stat:>6.3f}, p={p_val:.4f} {sig}")

    # Per-ticker breakdown: MAE and R² (return prediction quality, not direction)
    print(f"\n  Per-ticker breakdown (MAE in bps, R²_OOS):")
    ticker_labels = [all_samples[si]["ticker"] for si in sidxs]
    for ticker in TICKERS:
        tmask = np.array([t == ticker for t in ticker_labels])
        if tmask.sum() == 0:
            continue
        print(f"    {ticker} ({tmask.sum()} preds):", end="")
        for model_name in ["ridge_huber", "lgbm"]:
            pred_t = preds[model_name][tmask]
            act_t = actuals[tmask]
            hm_t = hm[tmask]
            mae_t = np.mean(np.abs(pred_t - act_t)) * 10000
            r2 = oos_r2(pred_t, act_t, hm_t)
            print(f"  {model_name} MAE={mae_t:.0f}bps R²={r2:.3f}", end="")
        print()

# --- 7. Return prediction quality diagnostics ---

print(f"\n{'=' * 100}")
print("RETURN PREDICTION DIAGNOSTICS (best model per target)")
print(f"{'=' * 100}")

for tgt_name, tgt_key, tgt_embargo, tgt_horizon in TARGETS:
    res = target_results[tgt_name]
    preds = res["predictions"]
    actuals = res["actuals_target"]

    # Pick best model by R²_OOS (excluding naive)
    metrics = evaluate_target(res)
    best_model = max((m for m in MODEL_NAMES if m != "naive"),
                     key=lambda m: metrics[m]["R2_OOS"])
    pred = preds[best_model]

    print(f"\n  {tgt_name} — best model: {best_model} (R²_OOS={metrics[best_model]['R2_OOS']:.4f})")

    # Tail performance: how well does the model predict extreme return days?
    abs_actuals = np.abs(actuals)
    p80 = np.percentile(abs_actuals, 80)
    tail_mask = abs_actuals >= p80
    core_mask = ~tail_mask

    tail_mae = np.mean(np.abs(pred[tail_mask] - actuals[tail_mask])) * 10000
    core_mae = np.mean(np.abs(pred[core_mask] - actuals[core_mask])) * 10000
    print(f"    Core days (bottom 80%):  MAE = {core_mae:.1f} bps ({core_mask.sum()} days)")
    print(f"    Tail days (top 20%):     MAE = {tail_mae:.1f} bps ({tail_mask.sum()} days)")
    print(f"    Tail/Core MAE ratio:     {tail_mae/core_mae:.1f}x")

    # Prediction range vs actual range
    print(f"    Actual return range:     [{actuals.min()*10000:+.0f}, {actuals.max()*10000:+.0f}] bps")
    print(f"    Predicted return range:  [{pred.min()*10000:+.0f}, {pred.max()*10000:+.0f}] bps")
    print(f"    Prediction std / Actual std: {np.std(pred)/np.std(actuals):.3f}")

# --- 8. Feature importance (cc_excess target, same features for all) ---

print(f"\n{'=' * 100}")
print("FEATURE IMPORTANCE (final refit on cc_excess target)")
print(f"{'=' * 100}")

# Use cc_excess target for feature importance (primary target)
y_fi = np.array([s["target_cc_excess"] if s["target_cc_excess"] is not None else 0.0 for s in all_samples])

scaler_final = StandardScaler()
X_all_s = scaler_final.fit_transform(X_all)

# Ridge (Huber) — primary model
huber_final = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
huber_final.fit(X_all_s, y_fi)
huber_coefs = list(zip(feature_names, huber_final.coef_))
huber_coefs.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\n  Ridge Huber -- Top 15 features by |coefficient|:")
for i, (name, coef) in enumerate(huber_coefs[:15]):
    print(f"    {i+1:>2}. {name:<25} {coef:>+10.6f}")

# LightGBM (Huber)
lgb_final_ds = lgb.Dataset(X_all_s, y_fi)
lgb_params_final = {
    "objective": "huber",
    "huber_delta": 1.35,
    "metric": "huber",
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

print("\n  LightGBM Huber -- Top 15 features by gain:")
for i, (name, imp) in enumerate(lgb_imp[:15]):
    print(f"    {i+1:>2}. {name:<25} {imp:>10.1f}")

# --- 9. Save predictions (one CSV per target) ---

for tgt_name, tgt_key, tgt_embargo, tgt_horizon in TARGETS:
    res = target_results[tgt_name]
    preds = res["predictions"]
    actuals = res["actuals_target"]
    sidxs = res["sample_indices"]
    hm = res["expanding_means"]

    # Determine if we can reconstruct prices
    can_price = tgt_name in ("cc_excess", "gap_excess")

    out_path = os.path.join(OUT_DIR, f"baseline_{tgt_name}_predictions.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["date", "ticker", "prev_close",
                  f"actual_{tgt_name}", "sp_ret",
                  "pred_naive", "pred_ridge", "pred_ridge_huber",
                  "pred_lasso", "pred_lgbm", "expanding_mean"]
        if can_price:
            header.extend(["actual_price", "pred_price_lgbm", "price_error_lgbm"])
        writer.writerow(header)

        for i, si in enumerate(sidxs):
            s = all_samples[si]
            if tgt_name == "gap_excess":
                sp_ret = s["sp_ret_gap"]
                actual_price = s["actual_open"]
            else:
                sp_ret = s["sp_ret_cc"]
                actual_price = s["actual_close"]

            row_data = [
                s["date"], s["ticker"],
                f"{s['prev_close']:.4f}",
                f"{actuals[i]:.6f}", f"{sp_ret:.6f}",
                f"{preds['naive'][i]:.6f}",
                f"{preds['ridge'][i]:.6f}",
                f"{preds['ridge_huber'][i]:.6f}",
                f"{preds['lasso'][i]:.6f}",
                f"{preds['lgbm'][i]:.6f}",
                f"{hm[i]:.6f}",
            ]
            if can_price:
                pred_price = s["prev_close"] * (1 + preds["lgbm"][i] + sp_ret)
                price_err = pred_price - actual_price
                row_data.extend([
                    f"{actual_price:.4f}",
                    f"{pred_price:.4f}",
                    f"{price_err:.4f}",
                ])
            writer.writerow(row_data)
    print(f"\nSaved {len(actuals)} predictions to {out_path}")

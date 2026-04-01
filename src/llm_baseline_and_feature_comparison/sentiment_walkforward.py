"""
Walk-forward evaluation of sentiment-enhanced models (per-ticker).

Each ticker gets its own model — no cross-ticker data mixing.

Methodology:
  - Walk-forward expanding window, per ticker
  - Daily refit (train on all data up to embargo cutoff)
  - embargo=1 for cc/gap, embargo=3 for cum3d (3-day overlap)
  - Clark-West test for statistical significance

Adds sentiment features on top of the price-only features:
  - sentiment (raw score from each LLM model)
  - |sentiment|
  - sentiment² (non-linear response)
  - sentiment × vol_20d (volatility-scaled)
  - sentiment × lag1_return (momentum interaction)
  - sentiment × A_cc_lag1 (abnormal return interaction)

Tests each sentiment model separately + a consensus feature (mean of all 5).
Compares: naive, price-only Ridge, price+sentiment Ridge, price+sentiment LightGBM.

This directly answers: does sentiment add statistically significant value
beyond price-only features under rigorous walk-forward evaluation?
"""

import csv
import math
import os
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import RidgeCV, LassoCV, HuberRegressor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import pandas as pd

warnings.filterwarnings("ignore")

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
SCORES_FILE = os.path.join(OUT_DIR, "scores_output.csv")

TRAIN_MIN = 100
REFIT_EVERY = 21  # Monthly refit — reduces from ~41k to ~2k model fits
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

TARGETS = [
    ("cc_excess",   "target_cc_excess",   1, 1),   # Embargo=1: train up to yesterday, predict today
    ("gap_excess",  "target_gap_excess",  1, 1),
    ("cum3d_excess","target_cum3d_excess", 3, 3),
]

LAGS = [1, 2, 3, 4, 5]
VOL_WINDOW_SHORT = 20
VOL_WINDOW_LONG = 60
MOMENTUM_WINDOWS = [5, 20]

SENT_MODELS = {
    "FinBERT": os.path.join(OUT_DIR, "finbert_lstm_results", "sentiment.csv"),
    "DeBERTa": os.path.join(OUT_DIR, "deberta_v3_lstm_results", "sentiment.csv"),
    "Gemma-3-1B": os.path.join(OUT_DIR, "gemma_3_1b_lstm_results", "sentiment.csv"),
    "Qwen2.5": os.path.join(OUT_DIR, "qwen25_lstm_results", "sentiment.csv"),
    "Llama-FinSent": os.path.join(OUT_DIR, "llama_finsent_lstm_results", "sentiment.csv"),
}


# --- 1. Load raw data (same as model_baseline.py) ---

print("Loading data...")

price_rows = {}
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

sp500 = {}
with open(os.path.join(RAW_DIR, "S&P 500 Historical Data.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        dt = datetime.strptime(row["Date"], "%m/%d/%Y")
        date_str = dt.strftime("%Y-%m-%d")
        sp500[date_str] = {
            "open": float(row["Open"].replace(",", "")),
            "close": float(row["Price"].replace(",", "")),
        }

scores_data = {}
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

# Load all sentiment models
sent_data = {}  # model_name → {(ticker, date) → score}
for name, path in SENT_MODELS.items():
    if os.path.exists(path):
        sent_data[name] = {}
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            key = (row["ticker"], str(pd.to_datetime(row["date"]).date()))
            sent_data[name][key] = float(row["finbert_score"])
        print(f"  Loaded {name}: {len(sent_data[name])} sentiment scores")

dates_per_ticker = {}
for (ticker, date) in price_rows:
    if ticker not in dates_per_ticker:
        dates_per_ticker[ticker] = []
    dates_per_ticker[ticker].append(date)
for t in dates_per_ticker:
    dates_per_ticker[t].sort()

all_dates = sorted(set(d for dates in dates_per_ticker.values() for d in dates))
print(f"  {len(all_dates)} trading days, {len(TICKERS)} tickers, {len(sent_data)} sentiment models")


# --- 2. Build per-ticker time series (same as model_baseline.py) ---

def build_ticker_series(ticker):
    dates = dates_per_ticker[ticker]
    series = []
    for i, date in enumerate(dates):
        d = price_rows[(ticker, date)]
        day = {
            "date": date, "ticker": ticker,
            "open": d["open"], "high": d["high"],
            "low": d["low"], "close": d["close"],
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

        day["ret_excess"] = day["ret_cc"] - day["sp_ret_cc"]
        day["ret_gap_excess"] = day["ret_gap"] - day["sp_ret_gap"]
        day["range_pct"] = (d["high"] - d["low"]) / d["close"] if d["close"] > 0 else 0.0

        sc = scores_data.get((ticker, date))
        day["A_cc"] = sc["A_cc"] if sc else 0.0
        day["A_gap"] = sc["A_gap"] if sc else 0.0
        day["zv"] = sc["zv"] if sc else 0.0
        day["beta_cc"] = sc["beta_cc"] if sc else 1.0

        # Add sentiment from all models
        for sname in sent_data:
            day[f"sent_{sname}"] = sent_data[sname].get((ticker, date), 0.0)

        series.append(day)
    return series


def compute_features(series, idx, include_sentiment=False, sent_model=None):
    """Compute feature vector. Same as model_baseline + optional sentiment features."""
    if idx < max(VOL_WINDOW_LONG, max(LAGS)) + 1:
        return None

    prev = series[idx - 1]
    feat = {}

    # === PRICE-ONLY FEATURES (same as model_baseline.py) ===

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

    cc_returns_20 = [series[idx - j]["ret_cc"] for j in range(1, VOL_WINDOW_SHORT + 1) if idx - j >= 0]
    cc_returns_60 = [series[idx - j]["ret_cc"] for j in range(1, VOL_WINDOW_LONG + 1) if idx - j >= 0]
    vol_20 = np.std(cc_returns_20, ddof=1) if len(cc_returns_20) > 1 else 0.01
    vol_60 = np.std(cc_returns_60, ddof=1) if len(cc_returns_60) > 1 else 0.01
    feat["vol_20d"] = vol_20
    feat["vol_ratio"] = vol_20 / max(vol_60, 1e-8)

    feat["range_pct_1"] = prev["range_pct"]
    range_5d = [series[idx - j]["range_pct"] for j in range(1, 6) if idx - j >= 0]
    feat["range_pct_5d_avg"] = np.mean(range_5d)

    vol_20d_avg = np.mean([series[idx - j]["volume"] for j in range(1, 21) if idx - j >= 0])
    feat["vol_z_1"] = prev["zv"]
    vol_5d_avg = np.mean([series[idx - j]["volume"] for j in range(1, 6) if idx - j >= 0])
    feat["vol_ratio_5d"] = vol_5d_avg / max(vol_20d_avg, 1) if vol_20d_avg > 0 else 1.0

    feat["beta_cc_1"] = prev["beta_cc"]
    feat["A_cc_1"] = prev["A_cc"]
    feat["A_gap_1"] = prev["A_gap"]
    feat["dow"] = series[idx]["dow"]

    # === SENTIMENT FEATURES (same-day: news available before close) ===
    if include_sentiment and sent_model:
        cur = series[idx]
        if sent_model == "consensus":
            s = np.mean([cur.get(f"sent_{sn}", 0.0) for sn in sent_data])
        else:
            s = cur.get(f"sent_{sent_model}", 0.0)

        feat["sent"] = s
        feat["abs_sent"] = abs(s)
        feat["sent_sq"] = s ** 2
        feat["sent_x_vol"] = s * vol_20
        feat["sent_x_lag1"] = s * prev["ret_cc"]
        feat["sent_x_A_cc"] = s * prev["A_cc"]

    return feat


# --- 3. Build per-ticker feature matrices ---

print("Building per-ticker feature matrices...")

SENT_CONFIGS = list(sent_data.keys()) + ["consensus"]

# Build samples per ticker (no cross-ticker mixing)
ticker_data = {}  # ticker → {samples, dates, X_price, X_sent, ...}

for ticker in TICKERS:
    series = build_ticker_series(ticker)
    samples = []
    for idx in range(1, len(series)):
        feat_price = compute_features(series, idx, include_sentiment=False)
        if feat_price is None:
            continue

        sent_feats = {}
        for sconf in SENT_CONFIGS:
            sf = compute_features(series, idx, include_sentiment=True, sent_model=sconf)
            if sf:
                sent_feats[sconf] = sf

        cum3d = None
        if idx + 2 < len(series):
            cum3d = sum(series[idx + j]["ret_excess"] for j in range(3))

        samples.append({
            "date": series[idx]["date"],
            "ticker": ticker,
            "features_price": feat_price,
            "features_sent": sent_feats,
            "target_cc_excess": series[idx]["ret_excess"],
            "target_gap_excess": series[idx]["ret_gap_excess"],
            "target_cum3d_excess": cum3d,
            "sp_ret_cc": series[idx]["sp_ret_cc"],
            "sp_ret_gap": series[idx]["sp_ret_gap"],
            "prev_close": series[idx - 1]["close"],
            "actual_close": series[idx]["close"],
            "actual_open": series[idx]["open"],
        })

    samples.sort(key=lambda s: s["date"])
    unique_dates = sorted(set(s["date"] for s in samples))
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    dates_arr = np.array([date_to_idx[s["date"]] for s in samples])

    # Get feature names from first sample
    feature_names_price = sorted(samples[0]["features_price"].keys())
    feature_names_sent = None
    for sconf in SENT_CONFIGS:
        if sconf in samples[0]["features_sent"]:
            feature_names_sent = sorted(samples[0]["features_sent"][sconf].keys())
            break

    X_price = np.array([[s["features_price"][f] for f in feature_names_price] for s in samples])
    X_sent = {}
    for sconf in SENT_CONFIGS:
        X_sent[sconf] = np.array([
            [s["features_sent"].get(sconf, s["features_price"]).get(f, 0.0) for f in feature_names_sent]
            for s in samples
        ])

    ticker_data[ticker] = {
        "samples": samples,
        "dates": dates_arr,
        "X_price": X_price,
        "X_sent": X_sent,
        "feature_names_price": feature_names_price,
        "feature_names_sent": feature_names_sent,
    }

    print(f"  {ticker}: {len(samples)} samples, {len(unique_dates)} dates")

n_price = len(ticker_data[TICKERS[0]]["feature_names_price"])
n_sent = len(ticker_data[TICKERS[0]]["feature_names_sent"]) if ticker_data[TICKERS[0]]["feature_names_sent"] else 0
print(f"  {n_price} price features, {n_sent} price+sent features")
print(f"  Sentiment configs: {SENT_CONFIGS}")


# --- 4. Walk-forward evaluation ---

def clark_west_test(pred_model, pred_bench, actual, nw_lags=0):
    e_bench = actual - pred_bench
    e_model = actual - pred_model
    adj = (pred_bench - pred_model) ** 2
    f_t = e_bench ** 2 - (e_model ** 2 - adj)
    if nw_lags > 0:
        n = len(f_t)
        x_dm = f_t - np.mean(f_t)
        gamma_0 = np.sum(x_dm ** 2) / n
        weighted_sum = 0.0
        for lag in range(1, nw_lags + 1):
            weight = 1 - lag / (nw_lags + 1)
            gamma_lag = np.sum(x_dm[lag:] * x_dm[:-lag]) / n
            weighted_sum += 2 * weight * gamma_lag
        var_hat = (gamma_0 + weighted_sum) / n
        se = np.sqrt(max(var_hat, 1e-16))
    else:
        se = np.std(f_t, ddof=1) / np.sqrt(len(f_t))
    t_stat = np.mean(f_t) / se
    p_value = 1 - norm.cdf(t_stat)
    return t_stat, p_value


def oos_r2(pred, actual, hist_means):
    ss_res = np.sum((actual - pred) ** 2)
    ss_bench = np.sum((actual - hist_means) ** 2)
    return 1 - ss_res / ss_bench if ss_bench > 0 else 0


# Model names: naive, price_ridge, then per-sentiment-model Ridge and LightGBM
def get_model_names():
    names = ["naive", "price_ridge", "price_lgbm"]
    for sconf in SENT_CONFIGS:
        short = sconf.replace("-", "").replace(".", "")[:8]
        names.append(f"ridge_{short}")
        names.append(f"lgbm_{short}")
    return names


MODEL_NAMES = get_model_names()


def run_walk_forward_ticker(ticker, target_name, target_key, embargo_days):
    """Walk-forward for a single ticker — no cross-ticker data."""
    td = ticker_data[ticker]
    samples = td["samples"]
    dates = td["dates"]
    X_price = td["X_price"]
    X_sent = td["X_sent"]

    valid_mask = np.array([s[target_key] is not None for s in samples])
    y_target = np.array([s[target_key] if s[target_key] is not None else 0.0 for s in samples])

    test_date_indices = sorted(set(di for di in dates if di >= TRAIN_MIN))

    predictions = {name: [] for name in MODEL_NAMES}
    actuals_list = []
    sample_indices = []
    expanding_means = []

    current_models = {}
    last_refit_di = -999

    for test_di in test_date_indices:
        train_cutoff = test_di - embargo_days
        train_mask = (dates <= train_cutoff) & valid_mask
        test_day_mask = (dates == test_di) & valid_mask

        X_train_p = X_price[train_mask]
        y_train = y_target[train_mask]
        X_test_p = X_price[test_day_mask]
        y_test = y_target[test_day_mask]

        if len(X_train_p) < 30 or len(X_test_p) == 0:
            continue

        hist_mean = np.mean(y_train)
        need_refit = (test_di - last_refit_di) >= REFIT_EVERY or not current_models

        if need_refit:
            models = {}

            scaler_p = StandardScaler()
            X_tr_ps = scaler_p.fit_transform(X_train_p)
            X_tr_ps = np.nan_to_num(X_tr_ps, nan=0.0, posinf=0.0, neginf=0.0)

            ridge_p = RidgeCV(alphas=[1, 10, 100, 1000])
            ridge_p.fit(X_tr_ps, y_train)
            models["price_scaler"] = scaler_p
            models["price_ridge"] = ridge_p

            lgb_ds = lgb.Dataset(X_tr_ps, y_train, free_raw_data=False)
            lgb_params = {
                "objective": "huber", "huber_delta": 1.35, "metric": "huber",
                "max_depth": 3, "num_leaves": 8, "min_child_samples": 30,
                "learning_rate": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
                "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "seed": 42,
            }
            models["price_lgbm"] = lgb.train(lgb_params, lgb_ds, num_boost_round=150,
                                              callbacks=[lgb.log_evaluation(0)])

            for sconf in SENT_CONFIGS:
                short = sconf.replace("-", "").replace(".", "")[:8]
                X_tr_s = X_sent[sconf][train_mask]
                scaler_s = StandardScaler()
                X_tr_ss = scaler_s.fit_transform(X_tr_s)
                X_tr_ss = np.nan_to_num(X_tr_ss, nan=0.0, posinf=0.0, neginf=0.0)

                ridge_s = RidgeCV(alphas=[1, 10, 100, 1000])
                ridge_s.fit(X_tr_ss, y_train)
                models[f"scaler_{short}"] = scaler_s
                models[f"ridge_{short}"] = ridge_s

                lgb_ds_s = lgb.Dataset(X_tr_ss, y_train, free_raw_data=False)
                models[f"lgbm_{short}"] = lgb.train(lgb_params, lgb_ds_s, num_boost_round=150,
                                                     callbacks=[lgb.log_evaluation(0)])

            current_models = models
            last_refit_di = test_di

        X_te_ps = np.nan_to_num(current_models["price_scaler"].transform(X_test_p), nan=0.0, posinf=0.0, neginf=0.0)
        pred_naive = np.full(len(X_test_p), hist_mean)
        pred_price_ridge = np.nan_to_num(current_models["price_ridge"].predict(X_te_ps), nan=hist_mean)
        pred_price_lgbm = current_models["price_lgbm"].predict(X_te_ps)

        sent_preds = {}
        for sconf in SENT_CONFIGS:
            short = sconf.replace("-", "").replace(".", "")[:8]
            X_te_s = X_sent[sconf][test_day_mask]
            X_te_ss = np.nan_to_num(current_models[f"scaler_{short}"].transform(X_te_s), nan=0.0, posinf=0.0, neginf=0.0)
            sent_preds[f"ridge_{short}"] = np.nan_to_num(current_models[f"ridge_{short}"].predict(X_te_ss), nan=hist_mean)
            sent_preds[f"lgbm_{short}"] = current_models[f"lgbm_{short}"].predict(X_te_ss)

        for i in range(len(X_test_p)):
            si = np.where(test_day_mask)[0][i]
            predictions["naive"].append(pred_naive[i])
            predictions["price_ridge"].append(pred_price_ridge[i])
            predictions["price_lgbm"].append(pred_price_lgbm[i])
            for k, v in sent_preds.items():
                predictions[k].append(v[i])
            actuals_list.append(y_test[i])
            sample_indices.append(si)
            expanding_means.append(hist_mean)

    actuals_arr = np.array(actuals_list)
    expanding_arr = np.array(expanding_means)
    for k in predictions:
        predictions[k] = np.array(predictions[k])

    return {
        "ticker": ticker,
        "predictions": predictions,
        "actuals": actuals_arr,
        "sample_indices": sample_indices,
        "expanding_means": expanding_arr,
    }


# --- 5. Run and evaluate (per-ticker) ---

print(f"\nWalk-forward PER-TICKER (min_train={TRAIN_MIN}, refit={REFIT_EVERY}d, embargo per target)")

for tgt_name, tgt_key, tgt_embargo, tgt_horizon in TARGETS:
    print(f"\n{'='*110}")
    print(f"  TARGET: {tgt_name} (embargo={tgt_embargo}, horizon={tgt_horizon})")
    print(f"{'='*110}")

    nw_lags = tgt_horizon - 1
    can_price = tgt_name in ("cc_excess", "gap_excess")

    # Collect per-ticker results, then aggregate
    all_preds = {name: [] for name in MODEL_NAMES}
    all_actuals = []
    all_hm = []
    all_tickers_label = []
    ticker_n_preds = {}

    for ticker in TICKERS:
        td = ticker_data[ticker]
        res = run_walk_forward_ticker(ticker, tgt_name, tgt_key, tgt_embargo)
        n_preds = len(res["actuals"])
        ticker_n_preds[ticker] = n_preds

        if n_preds == 0:
            continue

        for mname in MODEL_NAMES:
            all_preds[mname].append(res["predictions"][mname])
        all_actuals.append(res["actuals"])
        all_hm.append(res["expanding_means"])
        all_tickers_label.extend([ticker] * n_preds)

    # Concatenate all tickers
    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k]) if all_preds[k] else np.array([])
    actuals = np.concatenate(all_actuals) if all_actuals else np.array([])
    hm = np.concatenate(all_hm) if all_hm else np.array([])
    ticker_labels = np.array(all_tickers_label)

    total_preds = len(actuals)
    print(f"  {total_preds} OOS predictions ({', '.join(f'{t}={ticker_n_preds[t]}' for t in TICKERS)})")

    if total_preds == 0:
        continue

    # Aggregate table
    print(f"\n  {'Model':<20} {'MAE(bps)':>9} {'RMSE(bps)':>10} {'R²_OOS':>8} {'CW p':>7}")
    print(f"  {'-'*65}")

    for mname in MODEL_NAMES:
        pred = all_preds[mname]
        if len(pred) == 0:
            continue
        mae = np.mean(np.abs(pred - actuals)) * 10000
        rmse = np.sqrt(np.mean((pred - actuals) ** 2)) * 10000
        r2 = oos_r2(pred, actuals, hm)

        if mname == "naive":
            cw_str = "—"
        else:
            _, p = clark_west_test(pred, all_preds["naive"], actuals, nw_lags)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            cw_str = f"{p:.3f}{sig}"

        print(f"  {mname:<20} {mae:>9.1f} {rmse:>10.1f} {r2:>+8.4f} {cw_str:>7}")

    # Per-ticker breakdown
    print(f"\n  Per-ticker MAE (bps):")
    print(f"    {'Ticker':<8} {'N':>4} {'Naive':>8} {'PriceRidge':>11} {'PriceLGBM':>10}", end="")
    # Find best sentiment model
    best_sent_name, best_sent_mae = "", float("inf")
    for mname in MODEL_NAMES:
        if mname.startswith("ridge_") or mname.startswith("lgbm_"):
            if mname in ("price_ridge", "price_lgbm"):
                continue
            mae = np.mean(np.abs(all_preds[mname] - actuals)) * 10000
            if mae < best_sent_mae:
                best_sent_mae = mae
                best_sent_name = mname
    if best_sent_name:
        print(f" {best_sent_name:>15} {'Δ%':>6}")
    else:
        print()

    for ticker in TICKERS:
        tmask = ticker_labels == ticker
        if tmask.sum() == 0:
            continue
        n_mae = np.mean(np.abs(all_preds["naive"][tmask] - actuals[tmask])) * 10000
        pr_mae = np.mean(np.abs(all_preds["price_ridge"][tmask] - actuals[tmask])) * 10000
        pl_mae = np.mean(np.abs(all_preds["price_lgbm"][tmask] - actuals[tmask])) * 10000
        row = f"    {ticker:<8} {tmask.sum():>4} {n_mae:>8.1f} {pr_mae:>11.1f} {pl_mae:>10.1f}"
        if best_sent_name:
            bs_mae = np.mean(np.abs(all_preds[best_sent_name][tmask] - actuals[tmask])) * 10000
            delta = (bs_mae / n_mae - 1) * 100
            row += f" {bs_mae:>15.1f} {delta:>+5.1f}%"
        print(row)

    # Summary
    naive_mae = np.mean(np.abs(all_preds["naive"] - actuals)) * 10000
    price_ridge_mae = np.mean(np.abs(all_preds["price_ridge"] - actuals)) * 10000
    print(f"\n  Summary:")
    print(f"    Naive:        {naive_mae:.1f} bps")
    print(f"    Price Ridge:  {price_ridge_mae:.1f} bps ({(price_ridge_mae/naive_mae - 1)*100:+.1f}% vs naive)")
    if best_sent_name:
        print(f"    Best Sent:    {best_sent_mae:.1f} bps ({(best_sent_mae/naive_mae - 1)*100:+.1f}% vs naive) [{best_sent_name}]")

print("\nDone.")

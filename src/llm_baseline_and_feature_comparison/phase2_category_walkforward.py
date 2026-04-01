"""
Walk-forward evaluation of Phase 2 news category features for GOOGL.

Uses 29 granular LLM-extracted category scores (0-10) from news_phase2_GOOGL.csv
instead of single sentiment scores. Same walk-forward methodology as
sentiment_walkforward.py but adapted for:
  - GOOGL only (Phase 2 data exists only for this ticker)
  - Period-matched: gap-period rows for gap_excess, cc-period for cc/cum3d
  - 29 category features + derived interactions instead of 6 sentiment features
  - Smaller dataset → TRAIN_MIN=60, more aggressive regularization

Feature groups:
  - Price (~27 features): lagged returns, momentum, vol, range, volume, scores, dow
  - Raw categories (29): all cat_* columns, no sparsity filtering
  - Derived (5): centered sentiment direction, direction change, signed strength,
                 materiality×urgency, distinct_events
  - Interactions (5): direction×vol, direction×lag1, direction×A_cc,
                      surprise×vol, conviction×clarity
  - Category group means (6): sentiment, impact, information, strategic, googl, risk

Models: naive, price_ridge, all_cats_ridge, all_cats_elasticnet, all_cats_lasso,
        groups_ridge, pca_ridge, all_cats_lgbm, groups_lgbm

Tests: Clark-West vs naive, OOS R², feature importance analysis.
"""

import csv
import os
import warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lightgbm as lgb
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# --- Config ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
SCORES_FILE = os.path.join(OUT_DIR, "scores_output.csv")
PHASE2_FILE = os.path.join(OUT_DIR, "news_phase2_GOOGL.csv")

TICKER = "GOOGL"
TRAIN_MIN = 60
REFIT_EVERY = 1  # Daily refit — match sentiment_walkforward.py

TARGETS = [
    ("cc_excess",   "target_cc_excess",   1, 1),   # embargo=1 for single-day
    ("cum3d_excess","target_cum3d_excess", 3, 3),   # embargo=3: cum3d covers t,t+1,t+2
    # gap_excess removed: same-day news (gap+cc) predicts close, not gap (already realized)
]

LAGS = [1, 2, 3, 4, 5]
VOL_WINDOW_SHORT = 20
VOL_WINDOW_LONG = 60
MOMENTUM_WINDOWS = [5, 20]

# All 29 category columns
CAT_COLUMNS = [
    "cat_ai_strategic_investments", "cat_analyst_consensus_signals",
    "cat_capital_allocation_signal", "cat_catalyst_immediacy",
    "cat_cloud_enterprise_growth", "cat_competitive_impact",
    "cat_datacenter_energy_infrastructure", "cat_digital_platform_regulation",
    "cat_directional_clarity", "cat_earnings_financial_results",
    "cat_financial_result_surprise", "cat_gemini_ai_innovation",
    "cat_impact_timeline_duration", "cat_information_density",
    "cat_management_credibility", "cat_market_sector_sentiment",
    "cat_materiality", "cat_narrative_shift", "cat_regulatory_risk",
    "cat_scope_breadth", "cat_search_ad_antitrust",
    "cat_sentiment_direction", "cat_sentiment_strength",
    "cat_surprise_unexpectedness", "cat_systemic_contagion_risk",
    "cat_user_data_privacy", "cat_valuation_growth_outlook",
    "cat_waymo_autonomous_mobility", "cat_youtube_platform_dynamics",
]

CATEGORY_GROUPS = {
    "grp_sentiment": ["cat_sentiment_direction", "cat_sentiment_strength",
                      "cat_surprise_unexpectedness", "cat_narrative_shift"],
    "grp_impact": ["cat_materiality", "cat_catalyst_immediacy",
                   "cat_impact_timeline_duration", "cat_directional_clarity",
                   "cat_scope_breadth"],
    "grp_information": ["cat_information_density", "cat_management_credibility",
                        "cat_analyst_consensus_signals"],
    "grp_strategic": ["cat_competitive_impact", "cat_regulatory_risk",
                      "cat_valuation_growth_outlook", "cat_capital_allocation_signal",
                      "cat_market_sector_sentiment"],
    "grp_googl": ["cat_gemini_ai_innovation", "cat_ai_strategic_investments",
                  "cat_cloud_enterprise_growth", "cat_search_ad_antitrust",
                  "cat_digital_platform_regulation",
                  "cat_datacenter_energy_infrastructure"],
    "grp_risk": ["cat_systemic_contagion_risk", "cat_user_data_privacy"],
}

# LightGBM — very conservative for small data
LGB_PARAMS = {
    "objective": "huber", "huber_delta": 1.35, "metric": "huber",
    "max_depth": 2, "num_leaves": 4, "min_child_samples": 20,
    "learning_rate": 0.03, "subsample": 0.6, "colsample_bytree": 0.5,
    "reg_alpha": 0.5, "reg_lambda": 5.0, "verbose": -1, "seed": 42,
}
LGB_ROUNDS = 100


# --- 1. Load data ---

print("Loading data...")

price_rows = {}
with open(os.path.join(RAW_DIR, "price.csv")) as f:
    for row in csv.DictReader(f):
        if row["ticker"] != TICKER:
            continue
        key = row["date"]
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
        if row["ticker"] != TICKER:
            continue
        scores_data[row["date"]] = {
            "A_gap": float(row["A_gap"]),
            "A_cc": float(row["A_cc"]),
            "zv": float(row["zv"]),
            "beta_cc": float(row["beta_cc"]),
        }

# Load Phase 2 categories: keyed by (date, period)
phase2_data = {}  # (date, period) → {cat_col: value, ..., distinct_events: int}
phase2_df = pd.read_csv(PHASE2_FILE)
for _, row in phase2_df.iterrows():
    date_str = str(pd.to_datetime(row["date"]).date())
    period = row["period"]
    cats = {c: float(row[c]) for c in CAT_COLUMNS}
    cats["distinct_events"] = int(row["distinct_events"])
    phase2_data[(date_str, period)] = cats

dates_sorted = sorted(price_rows.keys())
print(f"  {len(dates_sorted)} trading days for {TICKER}")
print(f"  {len(phase2_data)} Phase 2 news rows ({sum(1 for k in phase2_data if k[1]=='cc')} cc, "
      f"{sum(1 for k in phase2_data if k[1]=='gap')} gap)")


# --- 2. Build GOOGL time series ---

def build_googl_series():
    series = []
    for i, date in enumerate(dates_sorted):
        d = price_rows[date]
        day = {
            "date": date,
            "open": d["open"], "high": d["high"],
            "low": d["low"], "close": d["close"],
            "volume": d["volume"],
            "dow": datetime.strptime(date, "%Y-%m-%d").weekday(),
        }

        if i > 0:
            prev = price_rows[dates_sorted[i - 1]]
            day["ret_cc"] = (d["close"] - prev["close"]) / prev["close"]
            day["ret_gap"] = (d["open"] - prev["close"]) / prev["close"]
        else:
            day["ret_cc"] = 0.0
            day["ret_gap"] = 0.0

        if date in sp500 and i > 0 and dates_sorted[i - 1] in sp500:
            sp_prev = sp500[dates_sorted[i - 1]]["close"]
            day["sp_ret_cc"] = (sp500[date]["close"] - sp_prev) / sp_prev
            day["sp_ret_gap"] = (sp500[date]["open"] - sp_prev) / sp_prev
        else:
            day["sp_ret_cc"] = 0.0
            day["sp_ret_gap"] = 0.0

        day["ret_excess"] = day["ret_cc"] - day["sp_ret_cc"]
        day["ret_gap_excess"] = day["ret_gap"] - day["sp_ret_gap"]
        day["range_pct"] = (d["high"] - d["low"]) / d["close"] if d["close"] > 0 else 0.0

        sc = scores_data.get(date)
        day["A_cc"] = sc["A_cc"] if sc else 0.0
        day["A_gap"] = sc["A_gap"] if sc else 0.0
        day["zv"] = sc["zv"] if sc else 0.0
        day["beta_cc"] = sc["beta_cc"] if sc else 1.0

        series.append(day)
    return series


def compute_price_features(series, idx):
    """Price-only features (same as sentiment_walkforward.py minus ticker dummies)."""
    if idx < max(VOL_WINDOW_LONG, max(LAGS)) + 1:
        return None

    prev = series[idx - 1]
    feat = {}

    for lag in LAGS:
        j = idx - lag
        if j >= 0:
            feat[f"price:ret_cc_{lag}"] = series[j]["ret_cc"]
            feat[f"price:ret_gap_{lag}"] = series[j]["ret_gap"]
            feat[f"price:sp_ret_cc_{lag}"] = series[j]["sp_ret_cc"]
            feat[f"price:ret_excess_{lag}"] = series[j]["ret_excess"]
        else:
            feat[f"price:ret_cc_{lag}"] = 0.0
            feat[f"price:ret_gap_{lag}"] = 0.0
            feat[f"price:sp_ret_cc_{lag}"] = 0.0
            feat[f"price:ret_excess_{lag}"] = 0.0

    for w in MOMENTUM_WINDOWS:
        cum = 1.0
        cum_excess = 1.0
        for j in range(1, w + 1):
            k = idx - j
            if k >= 0:
                cum *= (1 + series[k]["ret_cc"])
                cum_excess *= (1 + series[k]["ret_excess"])
        feat[f"price:momentum_{w}d"] = cum - 1.0
        feat[f"price:momentum_excess_{w}d"] = cum_excess - 1.0

    cc_returns_20 = [series[idx - j]["ret_cc"] for j in range(1, VOL_WINDOW_SHORT + 1) if idx - j >= 0]
    cc_returns_60 = [series[idx - j]["ret_cc"] for j in range(1, VOL_WINDOW_LONG + 1) if idx - j >= 0]
    vol_20 = np.std(cc_returns_20, ddof=1) if len(cc_returns_20) > 1 else 0.01
    vol_60 = np.std(cc_returns_60, ddof=1) if len(cc_returns_60) > 1 else 0.01
    feat["price:vol_20d"] = vol_20
    feat["price:vol_ratio"] = vol_20 / max(vol_60, 1e-8)

    feat["price:range_pct_1"] = prev["range_pct"]
    range_5d = [series[idx - j]["range_pct"] for j in range(1, 6) if idx - j >= 0]
    feat["price:range_pct_5d_avg"] = np.mean(range_5d)

    vol_20d_avg = np.mean([series[idx - j]["volume"] for j in range(1, 21) if idx - j >= 0])
    feat["price:vol_z_1"] = prev["zv"]
    vol_5d_avg = np.mean([series[idx - j]["volume"] for j in range(1, 6) if idx - j >= 0])
    feat["price:vol_ratio_5d"] = vol_5d_avg / max(vol_20d_avg, 1) if vol_20d_avg > 0 else 1.0

    feat["price:beta_cc_1"] = prev["beta_cc"]
    feat["price:A_cc_1"] = prev["A_cc"]
    feat["price:A_gap_1"] = prev["A_gap"]
    feat["price:dow"] = series[idx]["dow"]

    return feat, vol_20


def compute_category_features(date, prev_date):
    """Build category features from same-day Phase 2 news.

    For cc_excess prediction: use same-day gap news (overnight, known at open)
    + same-day cc news (intraday, known before close). Both are available
    before the close price is realized.

    sent_dir_change = cc_direction - gap_direction on same day (narrative shift
    from overnight to intraday — did the story change during the day?).
    """
    p2_gap = phase2_data.get((date, "gap"))
    p2_cc = phase2_data.get((date, "cc"))
    # Also get previous day for regime change across days
    p2_prev_cc = phase2_data.get((prev_date, "cc"))

    feat = {}

    # Use the best available same-day news: prefer cc (fuller picture), fallback to gap
    p2_primary = p2_cc if p2_cc is not None else p2_gap

    if p2_primary is None:
        # No Phase 2 news for this day — zero-fill all category features
        for c in CAT_COLUMNS:
            feat[f"cat:{c.replace('cat_', '')}"] = 0.0
        feat["derived:sent_dir_centered"] = 0.0
        feat["derived:sent_dir_change"] = 0.0
        feat["derived:day_narrative_shift"] = 0.0
        feat["derived:strength_signed"] = 0.0
        feat["derived:materiality_urgent"] = 0.0
        feat["derived:distinct_events"] = 0.0
        for gname in CATEGORY_GROUPS:
            feat[f"{gname}"] = 0.0
        feat["derived:has_news"] = 0.0
        feat["derived:has_gap_news"] = 0.0
        feat["derived:has_cc_news"] = 0.0
        # Gap-specific features
        feat["gap:sent_dir_centered"] = 0.0
        feat["gap:strength"] = 0.0
        feat["gap:materiality"] = 0.0
        return feat

    # Raw 29 categories from primary source (cc if available, else gap)
    for c in CAT_COLUMNS:
        feat[f"cat:{c.replace('cat_', '')}"] = p2_primary[c]

    # Derived features from primary
    sent_dir = p2_primary["cat_sentiment_direction"]
    sent_dir_centered = sent_dir - 5.0
    strength = p2_primary["cat_sentiment_strength"]

    feat["derived:sent_dir_centered"] = sent_dir_centered
    feat["derived:strength_signed"] = sent_dir_centered * strength
    feat["derived:materiality_urgent"] = p2_primary["cat_materiality"] * p2_primary["cat_catalyst_immediacy"] / 10.0
    feat["derived:distinct_events"] = float(p2_primary["distinct_events"])
    feat["derived:has_news"] = 1.0
    feat["derived:has_gap_news"] = 1.0 if p2_gap is not None else 0.0
    feat["derived:has_cc_news"] = 1.0 if p2_cc is not None else 0.0

    # Gap-specific features (overnight news — known at open, independent signal)
    if p2_gap is not None:
        gap_dir_c = p2_gap["cat_sentiment_direction"] - 5.0
        feat["gap:sent_dir_centered"] = gap_dir_c
        feat["gap:strength"] = p2_gap["cat_sentiment_strength"]
        feat["gap:materiality"] = p2_gap["cat_materiality"]
    else:
        feat["gap:sent_dir_centered"] = 0.0
        feat["gap:strength"] = 0.0
        feat["gap:materiality"] = 0.0

    # Day narrative shift: did sentiment change from overnight (gap) to intraday (cc)?
    if p2_gap is not None and p2_cc is not None:
        gap_dir_c = p2_gap["cat_sentiment_direction"] - 5.0
        cc_dir_c = p2_cc["cat_sentiment_direction"] - 5.0
        feat["derived:day_narrative_shift"] = cc_dir_c - gap_dir_c
    else:
        feat["derived:day_narrative_shift"] = 0.0

    # Regime change: today's direction vs yesterday's direction
    if p2_prev_cc is not None:
        prev_dir_c = p2_prev_cc["cat_sentiment_direction"] - 5.0
        feat["derived:sent_dir_change"] = sent_dir_centered - prev_dir_c
    else:
        feat["derived:sent_dir_change"] = 0.0

    # Category group means
    for gname, gcols in CATEGORY_GROUPS.items():
        vals = [p2_primary[c] for c in gcols]
        feat[f"{gname}"] = np.mean(vals)

    return feat


def compute_interaction_features(cat_feat, price_feat, vol_20):
    """Cross-feature interactions between categories and price."""
    feat = {}
    sent_dir_c = cat_feat.get("derived:sent_dir_centered", 0.0)
    strength_signed = cat_feat.get("derived:strength_signed", 0.0)

    feat["interact:sent_dir_x_vol"] = sent_dir_c * vol_20
    feat["interact:sent_dir_x_lag1"] = sent_dir_c * price_feat.get("price:ret_cc_1", 0.0)
    feat["interact:sent_dir_x_A_cc"] = sent_dir_c * price_feat.get("price:A_cc_1", 0.0)
    feat["interact:surprise_x_vol"] = cat_feat.get("cat:surprise_unexpectedness", 0.0) * vol_20
    feat["interact:conviction_x_clarity"] = (
        strength_signed * cat_feat.get("cat:directional_clarity", 0.0) / 10.0
    )

    return feat


# --- 3. Build feature matrices ---

print("Building feature matrices...")

series = build_googl_series()

# Same-day news: combine gap+cc Phase 2 data for each day.
# No separate period matrices needed — every sample uses same-day gap+cc news.


def build_samples():
    """Build all samples using same-day news (gap+cc combined)."""
    samples = []
    price_feat_names = None
    cat_feat_names = None
    interact_feat_names = None
    group_feat_names = None

    for idx in range(1, len(series)):
        pf_result = compute_price_features(series, idx)
        if pf_result is None:
            continue
        price_feat, vol_20 = pf_result

        if price_feat_names is None:
            price_feat_names = sorted(price_feat.keys())

        date = series[idx]["date"]
        prev_date = series[idx - 1]["date"]
        cat_feat = compute_category_features(date, prev_date)

        if cat_feat_names is None:
            cat_feat_names = sorted([k for k in cat_feat.keys()
                                     if k.startswith("cat:") or k.startswith("derived:") or k.startswith("gap:")])
            group_feat_names = sorted([k for k in cat_feat.keys()
                                       if k.startswith("grp_")])

        interact_feat = compute_interaction_features(cat_feat, price_feat, vol_20)
        if interact_feat_names is None:
            interact_feat_names = sorted(interact_feat.keys())

        # Targets
        cum3d = None
        if idx + 2 < len(series):
            cum3d = sum(series[idx + j]["ret_excess"] for j in range(3))

        samples.append({
            "date": date,
            "price_feat": price_feat,
            "cat_feat": cat_feat,
            "interact_feat": interact_feat,
            "vol_20": vol_20,
            "target_cc_excess": series[idx]["ret_excess"],
            "target_gap_excess": series[idx]["ret_gap_excess"],
            "target_cum3d_excess": cum3d,
            "sp_ret_cc": series[idx]["sp_ret_cc"],
            "sp_ret_gap": series[idx]["sp_ret_gap"],
            "prev_close": series[idx - 1]["close"],
            "actual_close": series[idx]["close"],
            "actual_open": series[idx]["open"],
            "has_news": cat_feat.get("derived:has_news", 0.0),
        })

    return samples, price_feat_names, cat_feat_names, group_feat_names, interact_feat_names


all_samples, pf_names, cf_names, gf_names, if_names = build_samples()
feat_names = {
    "price": pf_names,
    "cat": cf_names,
    "group": gf_names,
    "interact": if_names,
}
n_news = sum(1 for s in all_samples if s["has_news"] > 0)
print(f"  {len(all_samples)} samples, {n_news} with news ({100*n_news/len(all_samples):.0f}%)")


def build_feature_matrix(samples, feat_names_list):
    """Build numpy array from samples using specified feature name lists."""
    all_names = []
    for names in feat_names_list:
        all_names.extend(names)

    X = np.zeros((len(samples), len(all_names)))
    for i, s in enumerate(samples):
        combined = {}
        combined.update(s["price_feat"])
        combined.update(s["cat_feat"])
        combined.update(s["interact_feat"])
        for j, name in enumerate(all_names):
            X[i, j] = combined.get(name, 0.0)

    return X, all_names


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


MODEL_NAMES = [
    "naive",
    "nudge_all_news",    # fire on ALL news days, magnitude ∝ direction × vol
    "nudge_strength",    # nudge when strength is high
    "nudge_dir_simple",  # nudge proportional to sent_dir_centered × vol, threshold on |dir|
    "nudge_conviction",  # additive conviction: (|dir|+strength+materiality)/30 > threshold
    "nudge_surprise",    # nudge when surprise + directional
    "nudge_regime",      # nudge on sentiment direction shift (regime change)
    "nudge_material",    # nudge when materiality × catalyst are high
    "nudge_combo",       # cascade: regime > conviction > strength (first that fires)
    "price_ridge",
    "cats_ridge",        # keep one ML model as comparison
]


def get_sample_cats(samples, idx):
    """Extract raw category values for nudge models from a sample."""
    cf = samples[idx]["cat_feat"]
    return {
        "has_news": cf.get("derived:has_news", 0.0),
        "sent_dir_c": cf.get("derived:sent_dir_centered", 0.0),
        "strength": cf.get("cat:sentiment_strength", 0.0),
        "strength_signed": cf.get("derived:strength_signed", 0.0),
        "materiality": cf.get("cat:materiality", 0.0),
        "catalyst": cf.get("cat:catalyst_immediacy", 0.0),
        "clarity": cf.get("cat:directional_clarity", 0.0),
        "surprise": cf.get("cat:surprise_unexpectedness", 0.0),
        "sent_dir_change": cf.get("derived:sent_dir_change", 0.0),
        "day_narrative_shift": cf.get("derived:day_narrative_shift", 0.0),
        "gap_dir_c": cf.get("gap:sent_dir_centered", 0.0),
        "gap_strength": cf.get("gap:strength", 0.0),
        "scope": cf.get("cat:scope_breadth", 0.0),
        "info_density": cf.get("cat:information_density", 0.0),
        "vol_20": samples[idx]["vol_20"],
    }


def calibrate_nudge_params(samples, train_indices, y_target):
    """Calibrate nudge parameters from training data.

    Returns dict of calibrated parameters for each nudge model.
    For each approach, grid-search with a MINIMUM FIRE RATE constraint
    (~15% of all days) so models actually activate on news.
    """
    train_cats = [get_sample_cats(samples, i) for i in train_indices]
    y_train = y_target[train_indices]
    n = len(train_cats)

    # Useful training statistics
    news_mask = np.array([c["has_news"] > 0 for c in train_cats])
    abs_returns_news = np.abs(y_train[news_mask]) if news_mask.sum() > 3 else np.abs(y_train)
    median_abs_ret = np.median(abs_returns_news)
    mean_vol = np.mean([c["vol_20"] for c in train_cats])
    naive_mae = np.mean(np.abs(y_train))  # baseline: always predict 0

    MIN_FIRE_RATE = 0.12  # must fire on >= 12% of all days

    params = {}

    def _grid_search(make_preds_fn, param_grid, default_params):
        """Grid search with fire rate constraint. Returns best params."""
        best_mae, best_p = naive_mae, default_params
        for combo in param_grid:
            preds = make_preds_fn(combo)
            n_fired = np.sum(preds != 0)
            fire_rate = n_fired / n if n > 0 else 0
            if fire_rate < MIN_FIRE_RATE:
                continue  # skip configs that fire too rarely
            mae = np.mean(np.abs(y_train - preds))
            if mae < best_mae:
                best_mae = mae
                best_p = combo
        return best_p

    # --- nudge_strength: fire when has_news AND strength > threshold ---
    def _strength_preds(p):
        thr, damp = p["threshold"], p["damping"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0 and c["strength"] >= thr and abs(c["sent_dir_c"]) > 0.3:
                preds[j] = np.sign(c["sent_dir_c"]) * median_abs_ret * damp
        return preds

    grid = [{"threshold": t, "damping": d, "step": median_abs_ret}
            for t in [1, 2, 3, 4, 5] for d in np.arange(0.1, 2.1, 0.1)]
    params["nudge_strength"] = _grid_search(_strength_preds, grid,
        {"threshold": 3, "damping": 0.5, "step": median_abs_ret})

    # --- nudge_dir_simple: fire when |sent_dir_centered| > threshold ---
    # Lower thresholds to fire more; continuous magnitude scaling
    def _dir_preds(p):
        dthr, k = p["dir_thr"], p["k"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0 and abs(c["sent_dir_c"]) >= dthr:
                preds[j] = c["sent_dir_c"] / 5.0 * c["vol_20"] * k
        return preds

    grid = [{"dir_thr": t, "k": k}
            for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5] for k in np.arange(0.2, 6.1, 0.2)]
    params["nudge_dir_simple"] = _grid_search(_dir_preds, grid, {"dir_thr": 1.0, "k": 1.0})

    # --- nudge_conviction: additive conviction score ---
    def _conv_preds(p):
        conv_thr, k = p["conv_thr"], p["k"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0:
                conviction = (abs(c["sent_dir_c"]) + c["strength"] + c["materiality"]) / 3.0
                if conviction >= conv_thr and abs(c["sent_dir_c"]) > 0.3:
                    preds[j] = np.sign(c["sent_dir_c"]) * conviction / 10.0 * c["vol_20"] * k
        return preds

    grid = [{"conv_thr": t, "k": k}
            for t in np.arange(1.0, 5.1, 0.5) for k in np.arange(0.2, 6.1, 0.2)]
    params["nudge_conviction"] = _grid_search(_conv_preds, grid, {"conv_thr": 2.5, "k": 1.0})

    # --- nudge_surprise: fire when surprise > threshold AND directional ---
    def _surprise_preds(p):
        sthr, k = p["surprise_thr"], p["k"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0 and c["surprise"] >= sthr and abs(c["sent_dir_c"]) > 0.3:
                preds[j] = np.sign(c["sent_dir_c"]) * c["surprise"] / 10.0 * c["vol_20"] * k
        return preds

    grid = [{"surprise_thr": t, "k": k}
            for t in [1, 2, 3, 4, 5] for k in np.arange(0.5, 8.1, 0.5)]
    params["nudge_surprise"] = _grid_search(_surprise_preds, grid, {"surprise_thr": 3, "k": 1.0})

    # --- nudge_regime: fire when sent_dir_change is large (narrative shift) ---
    def _regime_preds(p):
        cthr, k = p["change_thr"], p["k"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0 and abs(c["sent_dir_change"]) >= cthr:
                preds[j] = np.sign(c["sent_dir_change"]) * abs(c["sent_dir_change"]) / 10.0 * c["vol_20"] * k
        return preds

    grid = [{"change_thr": t, "k": k}
            for t in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0] for k in np.arange(0.5, 8.1, 0.5)]
    params["nudge_regime"] = _grid_search(_regime_preds, grid, {"change_thr": 1.0, "k": 1.0})

    # --- nudge_material: fire when materiality × catalyst are both high ---
    def _material_preds(p):
        mthr, k = p["mat_thr"], p["k"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0 and c["materiality"] >= mthr and c["catalyst"] >= mthr:
                if abs(c["sent_dir_c"]) > 0.3:
                    urgency = c["materiality"] * c["catalyst"] / 100.0
                    preds[j] = np.sign(c["sent_dir_c"]) * urgency * c["vol_20"] * k
        return preds

    grid = [{"mat_thr": t, "k": k}
            for t in [2, 3, 4, 5, 6] for k in np.arange(0.5, 8.1, 0.5)]
    params["nudge_material"] = _grid_search(_material_preds, grid, {"mat_thr": 3, "k": 1.0})

    # --- nudge_all_news: fire on ALL news days, magnitude proportional to direction ---
    # No threshold — every news day gets a nudge scaled by direction × vol
    def _allnews_preds(p):
        k = p["k"]
        preds = np.zeros(n)
        for j in range(n):
            c = train_cats[j]
            if c["has_news"] > 0 and abs(c["sent_dir_c"]) > 0.0:
                preds[j] = c["sent_dir_c"] / 5.0 * c["vol_20"] * k
        return preds

    grid = [{"k": k} for k in np.arange(0.1, 4.1, 0.1)]
    params["nudge_all_news"] = _grid_search(_allnews_preds, grid, {"k": 0.5})

    return params


def predict_nudge(model_name, cats, params, hist_mean):
    """Generate a single nudge prediction."""
    if cats["has_news"] == 0:
        return hist_mean

    p = params.get(model_name, {})

    if model_name == "nudge_strength":
        thr = p.get("threshold", 99)
        damp = p.get("damping", 0.0)
        step = p.get("step", 0.0)
        if cats["strength"] >= thr and abs(cats["sent_dir_c"]) > 0.3:
            return hist_mean + np.sign(cats["sent_dir_c"]) * step * damp
        return hist_mean

    elif model_name == "nudge_dir_simple":
        dthr = p.get("dir_thr", 99)
        k = p.get("k", 0.0)
        if abs(cats["sent_dir_c"]) >= dthr:
            return hist_mean + cats["sent_dir_c"] / 5.0 * cats["vol_20"] * k
        return hist_mean

    elif model_name == "nudge_conviction":
        conv_thr = p.get("conv_thr", 99)
        k = p.get("k", 0.0)
        conviction = (abs(cats["sent_dir_c"]) + cats["strength"] + cats["materiality"]) / 3.0
        if conviction >= conv_thr and abs(cats["sent_dir_c"]) > 0.3:
            return hist_mean + np.sign(cats["sent_dir_c"]) * conviction / 10.0 * cats["vol_20"] * k
        return hist_mean

    elif model_name == "nudge_surprise":
        sthr = p.get("surprise_thr", 99)
        k = p.get("k", 0.0)
        if cats["surprise"] >= sthr and abs(cats["sent_dir_c"]) > 0.3:
            return hist_mean + np.sign(cats["sent_dir_c"]) * cats["surprise"] / 10.0 * cats["vol_20"] * k
        return hist_mean

    elif model_name == "nudge_regime":
        cthr = p.get("change_thr", 99)
        k = p.get("k", 0.0)
        if abs(cats["sent_dir_change"]) >= cthr:
            return hist_mean + np.sign(cats["sent_dir_change"]) * abs(cats["sent_dir_change"]) / 10.0 * cats["vol_20"] * k
        return hist_mean

    elif model_name == "nudge_material":
        mthr = p.get("mat_thr", 99)
        k = p.get("k", 0.0)
        if cats["materiality"] >= mthr and cats["catalyst"] >= mthr and abs(cats["sent_dir_c"]) > 0.3:
            urgency = cats["materiality"] * cats["catalyst"] / 100.0
            return hist_mean + np.sign(cats["sent_dir_c"]) * urgency * cats["vol_20"] * k
        return hist_mean

    elif model_name == "nudge_all_news":
        k = p.get("k", 0.0)
        if abs(cats["sent_dir_c"]) > 0.0:
            return hist_mean + cats["sent_dir_c"] / 5.0 * cats["vol_20"] * k
        return hist_mean

    return hist_mean


def run_walk_forward(target_name, target_key, embargo_days):
    samples = all_samples
    fn = feat_names

    # Build feature matrices for ML models
    X_price, names_price = build_feature_matrix(samples, [fn["price"]])
    X_all_cats, names_all_cats = build_feature_matrix(
        samples, [fn["price"], fn["cat"], fn["interact"]])

    valid_mask = np.array([s[target_key] is not None for s in samples])
    y_target = np.array([s[target_key] if s[target_key] is not None else 0.0 for s in samples])

    n_samples = len(samples)
    date_indices = np.arange(n_samples)
    test_indices = [i for i in range(n_samples) if i >= TRAIN_MIN and valid_mask[i]]

    predictions = {name: [] for name in MODEL_NAMES}
    actuals_list = []
    sample_indices = []
    expanding_means = []

    # Track which nudge model fires and when
    nudge_fire_counts = defaultdict(int)
    nudge_fire_correct = defaultdict(int)  # direction correct
    nudge_fire_errors = defaultdict(list)  # |pred - actual| on fire days
    nudge_naive_errors_on_fire = defaultdict(list)  # naive |error| on same fire days
    nudge_total_test = 0

    # Feature importance tracking for Ridge
    ridge_coefs = defaultdict(list)
    n_refits = 0

    current_models = {}
    current_nudge_params = {}
    last_refit_idx = -999

    for test_idx in test_indices:
        train_cutoff = test_idx - embargo_days
        train_mask = (date_indices <= train_cutoff) & valid_mask
        train_indices = np.where(train_mask)[0]

        X_train_p = X_price[train_mask]
        y_train = y_target[train_mask]

        if len(X_train_p) < 50 or len(y_train) < 50:
            continue

        hist_mean = np.mean(y_train)
        need_refit = (test_idx - last_refit_idx) >= REFIT_EVERY or not current_models

        if need_refit:
            models = {}
            n_refits += 1

            # --- Calibrate nudge models from training data ---
            current_nudge_params = calibrate_nudge_params(samples, train_indices, y_target)

            # --- Price-only Ridge ---
            scaler_p = StandardScaler()
            X_tr_p = scaler_p.fit_transform(X_train_p)
            ridge_p = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100])
            ridge_p.fit(X_tr_p, y_train)
            models["price_scaler"] = scaler_p
            models["price_ridge"] = ridge_p

            # --- All categories Ridge (one ML comparison) ---
            X_train_ac = X_all_cats[train_mask]
            scaler_ac = StandardScaler()
            X_tr_ac = scaler_ac.fit_transform(X_train_ac)
            ridge_ac = RidgeCV(alphas=[1, 10, 50, 100, 500])
            ridge_ac.fit(X_tr_ac, y_train)
            models["ac_scaler"] = scaler_ac
            models["cats_ridge"] = ridge_ac

            for fi, fname in enumerate(names_all_cats):
                ridge_coefs[fname].append(ridge_ac.coef_[fi])

            current_models = models
            last_refit_idx = test_idx

        # --- Predict ---
        test_cats = get_sample_cats(samples, test_idx)
        actual = y_target[test_idx]
        nudge_total_test += 1

        # Nudge models
        nudge_names = ["nudge_all_news", "nudge_strength", "nudge_dir_simple",
                       "nudge_conviction", "nudge_surprise", "nudge_regime", "nudge_material"]
        nudge_preds = {}
        for nm in nudge_names:
            pred = predict_nudge(nm, test_cats, current_nudge_params, hist_mean)
            nudge_preds[nm] = pred
            if pred != hist_mean:  # model fired
                nudge_fire_counts[nm] += 1
                nudge_fire_errors[nm].append(abs(pred - actual))
                nudge_naive_errors_on_fire[nm].append(abs(hist_mean - actual))
                if np.sign(pred - hist_mean) == np.sign(actual - hist_mean) and actual != hist_mean:
                    nudge_fire_correct[nm] += 1

        # Combo: cascade through models in priority order, use first that fires
        # Regime (narrative shift) > conviction (strong+material+directional) > dir_simple
        priority = ["nudge_regime", "nudge_conviction", "nudge_dir_simple",
                     "nudge_surprise", "nudge_material", "nudge_strength"]
        combo_pred = hist_mean
        for nm in priority:
            if nudge_preds[nm] != hist_mean:
                combo_pred = nudge_preds[nm]
                break

        predictions["naive"].append(hist_mean)
        for nm in nudge_names:
            predictions[nm].append(nudge_preds[nm])
        predictions["nudge_combo"].append(combo_pred)

        # ML models
        X_te_p = X_price[test_idx:test_idx+1]
        X_te_ac = X_all_cats[test_idx:test_idx+1]
        X_te_p_s = current_models["price_scaler"].transform(X_te_p)
        X_te_ac_s = current_models["ac_scaler"].transform(X_te_ac)

        predictions["price_ridge"].append(current_models["price_ridge"].predict(X_te_p_s)[0])
        predictions["cats_ridge"].append(current_models["cats_ridge"].predict(X_te_ac_s)[0])

        actuals_list.append(actual)
        sample_indices.append(test_idx)
        expanding_means.append(hist_mean)

    actuals_arr = np.array(actuals_list)
    expanding_arr = np.array(expanding_means)
    for k in predictions:
        predictions[k] = np.array(predictions[k])

    return {
        "target_name": target_name,
        "predictions": predictions,
        "actuals": actuals_arr,
        "sample_indices": sample_indices,
        "expanding_means": expanding_arr,
        "ridge_coefs": dict(ridge_coefs),
        "n_refits": n_refits,
        "feature_names_all_cats": names_all_cats,
        "nudge_fire_counts": dict(nudge_fire_counts),
        "nudge_fire_correct": dict(nudge_fire_correct),
        "nudge_fire_errors": dict(nudge_fire_errors),
        "nudge_naive_errors_on_fire": dict(nudge_naive_errors_on_fire),
        "nudge_total_test": nudge_total_test,
    }


# --- 5. Analysis ---

def print_nudge_analysis(res):
    """Print how often each nudge model fired and its directional accuracy."""
    counts = res["nudge_fire_counts"]
    correct = res["nudge_fire_correct"]
    total = res["nudge_total_test"]
    if total == 0:
        return

    fire_errs = res.get("nudge_fire_errors", {})
    naive_errs = res.get("nudge_naive_errors_on_fire", {})

    print(f"\n  Nudge model firing analysis ({total} test days):")
    print(f"    {'Model':<22} {'Fired':>6} {'Fire%':>7} {'DirAcc':>8} {'MAE fire':>10} {'Naive fire':>11} {'Improve':>9}")
    print(f"    {'-'*80}")
    for nm in ["nudge_all_news", "nudge_strength", "nudge_dir_simple", "nudge_conviction",
                "nudge_surprise", "nudge_regime", "nudge_material"]:
        n_fire = counts.get(nm, 0)
        n_correct = correct.get(nm, 0)
        fire_pct = 100 * n_fire / total if total > 0 else 0
        dir_acc = 100 * n_correct / n_fire if n_fire > 0 else 0
        if n_fire > 0 and nm in fire_errs:
            mae_fire = np.mean(fire_errs[nm]) * 10000
            mae_naive_fire = np.mean(naive_errs[nm]) * 10000
            improve = (mae_fire / mae_naive_fire - 1) * 100 if mae_naive_fire > 0 else 0
            print(f"    {nm:<22} {n_fire:>6} {fire_pct:>6.1f}% {dir_acc:>7.1f}% {mae_fire:>9.1f}bp {mae_naive_fire:>10.1f}bp {improve:>+8.1f}%")
        else:
            print(f"    {nm:<22} {n_fire:>6} {fire_pct:>6.1f}% {dir_acc:>7.1f}%")


def print_ridge_importance(res):
    """Print Ridge coefficient stability for category features."""
    names = res["feature_names_all_cats"]
    n_refits = res["n_refits"]
    if n_refits == 0:
        return

    print(f"\n  Ridge coefficient stability (top category features, {n_refits} windows):")
    ridge_stable = []
    for fname in names:
        if not (fname.startswith("cat:") or fname.startswith("derived:") or
                fname.startswith("interact:") or fname.startswith("grp_")):
            continue
        coefs = res["ridge_coefs"].get(fname, [])
        if len(coefs) < 3:
            continue
        mean_c = np.mean(coefs)
        std_c = np.std(coefs)
        ratio = abs(mean_c / std_c) if std_c > 1e-10 else 0
        if ratio > 0.5:
            ridge_stable.append((fname, mean_c, std_c, ratio))

    ridge_stable.sort(key=lambda x: -x[3])
    if ridge_stable:
        print(f"    {'Feature':<45} {'Mean coef':>12} {'|M/S|':>8}")
        for fname, mc, sc, ratio in ridge_stable[:10]:
            sign = "+" if mc > 0 else ""
            print(f"    {fname:<45} {sign}{mc:>.6f} {ratio:>8.2f}")
    else:
        print(f"    (no stable category features found)")


# --- 6. Run and evaluate ---

print(f"\nWalk-forward for {TICKER} (min_train={TRAIN_MIN}, refit={REFIT_EVERY}d)")

for tgt_name, tgt_key, tgt_embargo, tgt_horizon in TARGETS:
    n_samples = len(all_samples)

    print(f"\n{'='*110}")
    print(f"  TARGET: {tgt_name} (embargo={tgt_embargo}, horizon={tgt_horizon})")
    print(f"  Samples: {n_samples}, OOS start at idx {TRAIN_MIN}")
    print(f"{'='*110}")

    if n_samples < TRAIN_MIN + 10:
        print(f"  WARNING: Only {n_samples} samples, need at least {TRAIN_MIN + 10}. Skipping.")
        continue

    res = run_walk_forward(tgt_name, tgt_key, tgt_embargo)
    preds = res["predictions"]
    actuals = res["actuals"]
    hm = res["expanding_means"]
    sidxs = res["sample_indices"]
    nw_lags = tgt_horizon - 1
    n_preds = len(actuals)
    samples = all_samples

    if n_preds < 10:
        print(f"  WARNING: Only {n_preds} OOS predictions. Results unreliable.")
        if n_preds == 0:
            continue

    if n_preds < 30:
        print(f"  NOTE: {n_preds} OOS predictions — Clark-West has low power at this sample size.")

    print(f"  {n_preds} OOS predictions ({res['n_refits']} refits)")

    # Price reconstruction
    can_price = tgt_name in ("cc_excess", "gap_excess")
    if can_price:
        prev_closes = np.array([samples[si]["prev_close"] for si in sidxs])
        sp_key = "sp_ret_cc" if tgt_name == "cc_excess" else "sp_ret_gap"
        sp_rets = np.array([samples[si][sp_key] for si in sidxs])
        price_key = "actual_close" if tgt_name == "cc_excess" else "actual_open"
        actual_prices = np.array([samples[si][price_key] for si in sidxs])

    # Table header
    if can_price:
        print(f"\n  {'Model':<25} {'MAE(bps)':>9} {'RMSE(bps)':>10} {'R²_OOS':>8} {'$MAE':>7} {'CW p':>10}")
    else:
        print(f"\n  {'Model':<25} {'MAE(bps)':>9} {'RMSE(bps)':>10} {'R²_OOS':>8} {'CW p':>10}")
    print(f"  {'-'*80}")

    for mname in MODEL_NAMES:
        pred = preds[mname]
        mae = np.mean(np.abs(pred - actuals)) * 10000
        rmse = np.sqrt(np.mean((pred - actuals) ** 2)) * 10000
        r2 = oos_r2(pred, actuals, hm)

        if mname == "naive":
            cw_str = "—"
        else:
            _, p = clark_west_test(pred, preds["naive"], actuals, nw_lags)
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            cw_str = f"{p:.4f}{sig}"

        if can_price:
            pred_prices = prev_closes * (1 + pred + sp_rets)
            price_mae = np.mean(np.abs(pred_prices - actual_prices))
            print(f"  {mname:<25} {mae:>9.1f} {rmse:>10.1f} {r2:>+8.4f} ${price_mae:>6.2f} {cw_str:>10}")
        else:
            print(f"  {mname:<25} {mae:>9.1f} {rmse:>10.1f} {r2:>+8.4f} {cw_str:>10}")

    # Summary
    print(f"\n  Summary — category models vs price-only vs naive:")
    naive_mae = np.mean(np.abs(preds["naive"] - actuals)) * 10000
    price_ridge_mae = np.mean(np.abs(preds["price_ridge"] - actuals)) * 10000

    best_cat_name, best_cat_mae = "", float("inf")
    for mname in MODEL_NAMES:
        if mname in ("naive", "price_ridge"):
            continue
        mae = np.mean(np.abs(preds[mname] - actuals)) * 10000
        if mae < best_cat_mae:
            best_cat_mae = mae
            best_cat_name = mname

    print(f"    Naive:            {naive_mae:.1f} bps")
    print(f"    Price Ridge:      {price_ridge_mae:.1f} bps ({(price_ridge_mae/naive_mae - 1)*100:+.1f}% vs naive)")
    print(f"    Best Category:    {best_cat_mae:.1f} bps ({(best_cat_mae/naive_mae - 1)*100:+.1f}% vs naive) [{best_cat_name}]")

    # News vs no-news breakdown
    has_news = np.array([samples[si]["has_news"] for si in sidxs])
    n_news = has_news.sum()
    if n_news > 5 and (len(has_news) - n_news) > 5:
        print(f"\n  News vs no-news breakdown (best category model: {best_cat_name}):")
        for label, mask in [("With news", has_news > 0), ("No news", has_news == 0)]:
            n_sub = mask.sum()
            mae_naive = np.mean(np.abs(preds["naive"][mask] - actuals[mask])) * 10000
            mae_price = np.mean(np.abs(preds["price_ridge"][mask] - actuals[mask])) * 10000
            mae_best = np.mean(np.abs(preds[best_cat_name][mask] - actuals[mask])) * 10000
            print(f"    {label:<12} (n={int(n_sub):>3}): naive={mae_naive:.1f}, "
                  f"price={mae_price:.1f}, best_cat={mae_best:.1f} bps")

    # Nudge analysis
    print_nudge_analysis(res)
    print_ridge_importance(res)

print("\nDone.")

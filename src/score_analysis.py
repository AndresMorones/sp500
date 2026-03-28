import csv
import math
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "data", "output", "scores_output.csv")

# --- Load data ---
rows = []
with open(INPUT, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        row = {"date": r["date"], "ticker": r["ticker"]}
        for k in ["A_gap", "A_cc", "D_gap", "D_cc", "E_gap", "E_cc",
                   "Ev_gap", "Ev_cc", "Dv_gap", "Dv_cc",
                   "stock_gap", "stock_cc", "sp_gap", "sp_cc", "zv"]:
            row[k] = float(r[k])
        row["headline_gap"] = r.get("headline_gap", "")
        row["headline_cc"] = r.get("headline_cc", "")
        rows.append(row)

METRICS = ["A", "D", "E", "Ev", "Dv"]
PERIODS = ["gap", "cc"]
N = 100

# --- For each metric+period, get top 50 highest and lowest ---

rankings = {}  # (metric, period, "high"/"low") -> list of (row, score)

for m in METRICS:
    for p in PERIODS:
        col = f"{m}_{p}"
        sorted_rows = sorted(rows, key=lambda r: r[col])
        rankings[(m, p, "low")] = [(r, r[col]) for r in sorted_rows[:N]]
        rankings[(m, p, "high")] = [(r, r[col]) for r in sorted_rows[-N:][::-1]]

# --- Compare overlap between metrics ---

def get_keys(ranking_list):
    """Return set of (date, ticker) from a ranking list."""
    return set((r["date"], r["ticker"]) for r, _ in ranking_list)

print("=" * 100)
print("OVERLAP ANALYSIS: How many of the top/bottom 50 rows are shared between metrics?")
print("=" * 100)

for p in PERIODS:
    print(f"\n{'-' * 100}")
    print(f"  {p.upper()} — Top 50 highest scores")
    print(f"{'-' * 100}")
    # Header
    print(f"{'':>8}", end="")
    for m2 in METRICS:
        print(f"{m2:>8}", end="")
    print()
    for m1 in METRICS:
        print(f"{m1:>8}", end="")
        keys1 = get_keys(rankings[(m1, p, "high")])
        for m2 in METRICS:
            keys2 = get_keys(rankings[(m2, p, "high")])
            overlap = len(keys1 & keys2)
            print(f"{overlap:>8}", end="")
        print()

    print(f"\n  {p.upper()} — Bottom 50 lowest scores")
    print(f"{'-' * 100}")
    print(f"{'':>8}", end="")
    for m2 in METRICS:
        print(f"{m2:>8}", end="")
    print()
    for m1 in METRICS:
        print(f"{m1:>8}", end="")
        keys1 = get_keys(rankings[(m1, p, "low")])
        for m2 in METRICS:
            keys2 = get_keys(rankings[(m2, p, "low")])
            overlap = len(keys1 & keys2)
            print(f"{overlap:>8}", end="")
        print()

# --- Show top 10 for each metric with headlines ---

print(f"\n{'=' * 100}")
print("TOP 10 HIGHEST & LOWEST per metric (with headlines)")
print(f"{'=' * 100}")

for m in METRICS:
    for p in PERIODS:
        headline_col = f"headline_{p}"
        col = f"{m}_{p}"

        print(f"\n{'-' * 100}")
        print(f"  {m}_{p} — TOP 10 HIGHEST")
        print(f"{'-' * 100}")
        print(f"{'Rank':>4} {'Date':>12} {'Ticker':>6} {'Score':>10} {'Stock%':>8} {'SP500%':>8}  Headlines")
        for i, (r, score) in enumerate(rankings[(m, p, "high")][:10]):
            stock_pct = r[f"stock_{p}"] * 100
            sp_pct = r[f"sp_{p}"] * 100
            hl = r[headline_col][:80] if r[headline_col] else "(no news)"
            print(f"{i+1:>4} {r['date']:>12} {r['ticker']:>6} {score:>10.1f} {stock_pct:>+8.2f} {sp_pct:>+8.2f}  {hl}")

        print(f"\n  {m}_{p} — BOTTOM 10 LOWEST")
        print(f"{'-' * 100}")
        print(f"{'Rank':>4} {'Date':>12} {'Ticker':>6} {'Score':>10} {'Stock%':>8} {'SP500%':>8}  Headlines")
        for i, (r, score) in enumerate(rankings[(m, p, "low")][:10]):
            stock_pct = r[f"stock_{p}"] * 100
            sp_pct = r[f"sp_{p}"] * 100
            hl = r[headline_col][:80] if r[headline_col] else "(no news)"
            print(f"{i+1:>4} {r['date']:>12} {r['ticker']:>6} {score:>10.1f} {stock_pct:>+8.2f} {sp_pct:>+8.2f}  {hl}")

# --- News presence analysis ---

print(f"\n{'=' * 100}")
print("NEWS PRESENCE: % of top/bottom 50 that have at least one headline")
print(f"{'=' * 100}")

print(f"\n{'':>8} {'High gap':>12} {'Low gap':>12} {'High cc':>12} {'Low cc':>12}")
for m in METRICS:
    vals = []
    for p in PERIODS:
        for direction in ["high", "low"]:
            headline_col = f"headline_{p}"
            count = sum(1 for r, _ in rankings[(m, p, direction)] if r[headline_col])
            vals.append(f"{count}/{N} ({count*100//N}%)")
    print(f"{m:>8} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12} {vals[3]:>12}")

# --- Rank correlation (Spearman-like) between metrics ---

print(f"\n{'=' * 100}")
print("RANK CORRELATION: Do metrics rank days similarly? (on full dataset)")
print(f"{'=' * 100}")

def rank_list(rows, col):
    """Return dict of (date,ticker) -> rank."""
    sorted_keys = sorted(range(len(rows)), key=lambda i: rows[i][col])
    ranks = {}
    for rank, idx in enumerate(sorted_keys):
        r = rows[idx]
        ranks[(r["date"], r["ticker"])] = rank
    return ranks

for p in PERIODS:
    print(f"\n  {p.upper()} rank correlation (Spearman rho):")
    print(f"{'':>8}", end="")
    for m2 in METRICS:
        print(f"{m2:>8}", end="")
    print()

    rank_caches = {}
    for m in METRICS:
        rank_caches[m] = rank_list(rows, f"{m}_{p}")

    n = len(rows)
    for m1 in METRICS:
        print(f"{m1:>8}", end="")
        r1 = rank_caches[m1]
        for m2 in METRICS:
            r2 = rank_caches[m2]
            # Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
            d_sq = sum((r1[k] - r2[k]) ** 2 for k in r1)
            rho = 1 - 6 * d_sq / (n * (n * n - 1))
            print(f"{rho:>8.3f}", end="")
        print()

# =============================================================================
# ANALYSIS 1: Cohen's d — score separation between news and no-news days
# =============================================================================

def cohens_d(group1, group2):
    """Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = sum(group1) / n1, sum(group2) / n2
    v1 = sum((x - m1) ** 2 for x in group1) / (n1 - 1)
    v2 = sum((x - m2) ** 2 for x in group2) / (n2 - 1)
    pooled_std = math.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled_std < 1e-12:
        return float("nan")
    return (m1 - m2) / pooled_std

print(f"\n{'=' * 100}")
print("COHEN'S d: Score separation between news days vs no-news days")
print("  (positive = news days have higher |score|, >0.5 is medium, >0.8 is large)")
print(f"{'=' * 100}")

print(f"\n{'':>8} {'Gap d':>10} {'Gap news':>10} {'Gap none':>10} {'CC d':>10} {'CC news':>10} {'CC none':>10}")
cohens_results = []
for m in METRICS:
    vals = []
    for p in PERIODS:
        col = f"{m}_{p}"
        headline_col = f"headline_{p}"
        news_scores = [abs(r[col]) for r in rows if r[headline_col]]
        no_news_scores = [abs(r[col]) for r in rows if not r[headline_col]]
        d = cohens_d(news_scores, no_news_scores)
        vals.extend([d, len(news_scores), len(no_news_scores)])
    cohens_results.append((m, vals[0], vals[3]))
    print(f"{m:>8} {vals[0]:>10.3f} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10.3f} {vals[4]:>10} {vals[5]:>10}")

print("\n  Ranking by average Cohen's d (gap + cc):")
cohens_results.sort(key=lambda x: -(x[1] + x[2]) / 2)
for i, (m, d_gap, d_cc) in enumerate(cohens_results, 1):
    avg = (d_gap + d_cc) / 2
    print(f"    {i}. {m:<6} avg d={avg:.3f}  (gap={d_gap:.3f}, cc={d_cc:.3f})")

# =============================================================================
# ANALYSIS 2: Precision@K curves — news hit rate as K varies from 10 to 500
# =============================================================================

print(f"\n{'=' * 100}")
print("PRECISION@K: % of top/bottom K events with news (gap + cc combined)")
print(f"{'=' * 100}")

K_VALUES = [10, 20, 50, 100, 150, 200, 300, 500]

for p in PERIODS:
    headline_col = f"headline_{p}"
    print(f"\n  {p.upper()} period:")
    # Header
    header = f"{'K':>6}"
    for m in METRICS:
        header += f" {m:>8}"
    print(header)
    print("  " + "-" * (6 + 9 * len(METRICS)))

    for k in K_VALUES:
        line = f"{k:>6}"
        for m in METRICS:
            col = f"{m}_{p}"
            sorted_rows = sorted(rows, key=lambda r, c=col: r[c])
            bottom_k = sorted_rows[:k]
            top_k = sorted_rows[-k:]
            combined = bottom_k + top_k
            has_news = sum(1 for r in combined if r[headline_col])
            pct = has_news / len(combined) * 100
            line += f" {pct:>7.1f}%"
        print(line)

# =============================================================================
# ANALYSIS 5: Per-ticker consistency — news rate in top/bottom 100 by ticker
# =============================================================================

print(f"\n{'=' * 100}")
print("PER-TICKER CONSISTENCY: News rate in each ticker's top/bottom extremes")
print("  (std = cross-ticker standard deviation — lower = more consistent)")
print(f"{'=' * 100}")

all_tickers = sorted(set(r["ticker"] for r in rows))

for p in PERIODS:
    headline_col = f"headline_{p}"
    print(f"\n  {p.upper()} period — top/bottom events per ticker (K chosen per ticker = max(20, ticker_count//5)):")

    # Header
    header = f"{'Metric':<8}"
    for t in all_tickers:
        header += f" {t:>8}"
    header += f" {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    print(header)
    print("  " + "-" * len(header))

    for m in METRICS:
        col = f"{m}_{p}"
        ticker_rates = []
        line = f"{m:<8}"
        for t in all_tickers:
            ticker_rows = [r for r in rows if r["ticker"] == t]
            k = max(20, len(ticker_rows) // 5)
            sorted_ticker = sorted(ticker_rows, key=lambda r: r[col])
            extremes = sorted_ticker[:k] + sorted_ticker[-k:]
            has_news = sum(1 for r in extremes if r[headline_col])
            rate = has_news / len(extremes) * 100
            ticker_rates.append(rate)
            line += f" {rate:>7.1f}%"

        avg_rate = sum(ticker_rates) / len(ticker_rates)
        std_rate = math.sqrt(sum((r - avg_rate) ** 2 for r in ticker_rates) / len(ticker_rates))
        min_rate = min(ticker_rates)
        max_rate = max(ticker_rates)
        line += f" {avg_rate:>7.1f}% {std_rate:>7.1f}% {min_rate:>7.1f}% {max_rate:>7.1f}%"
        print(line)

    # Also show ticker row counts for context
    print(f"\n  Row counts per ticker:")
    for t in all_tickers:
        cnt = sum(1 for r in rows if r["ticker"] == t)
        print(f"    {t}: {cnt}")

# =============================================================================
# ANALYSIS 6: Forward return (reversal vs continuation) after extreme events
# =============================================================================

HORIZONS = [1, 2, 3, 4, 5]

# Build per-ticker timeline: sorted list of rows with index lookup by (ticker, date)
ticker_timeline = {}  # ticker -> [row, row, ...]
ticker_date_idx = {}  # (ticker, date) -> index into ticker_timeline

for r in rows:
    t = r["ticker"]
    if t not in ticker_timeline:
        ticker_timeline[t] = []
    ticker_timeline[t].append(r)

for t in ticker_timeline:
    ticker_timeline[t].sort(key=lambda r: r["date"])
    for i, r in enumerate(ticker_timeline[t]):
        ticker_date_idx[(t, r["date"])] = i

def forward_returns(row, horizon):
    """Get forward stock_cc and sp_cc returns for days +1 through +horizon.
    Returns list of (stock_cc, sp_cc) tuples, or None if insufficient data."""
    t = row["ticker"]
    idx = ticker_date_idx.get((t, row["date"]))
    if idx is None:
        return None
    timeline = ticker_timeline[t]
    if idx + horizon >= len(timeline):
        return None
    return [(timeline[idx + d]["stock_cc"], timeline[idx + d]["sp_cc"])
            for d in range(1, horizon + 1)]

def compute_forward_stats(event_rows, max_horizon=5):
    """For a list of (row, score) pairs, compute forward return stats.
    Returns dict with cumulative and individual excess returns per horizon."""
    cum_excess = {h: [] for h in HORIZONS}
    ind_excess = {h: [] for h in HORIZONS}
    ind_stock = {h: [] for h in HORIZONS}
    ind_sp = {h: [] for h in HORIZONS}

    for row, score in event_rows:
        fwd = forward_returns(row, max_horizon)
        if fwd is None:
            continue
        # Cumulative returns at each horizon
        cum_stock = 1.0
        cum_sp = 1.0
        for d in range(max_horizon):
            h = d + 1
            cum_stock *= (1 + fwd[d][0])
            cum_sp *= (1 + fwd[d][1])
            cum_excess[h].append((cum_stock - 1) - (cum_sp - 1))
            ind_excess[h].append(fwd[d][0] - fwd[d][1])
            ind_stock[h].append(fwd[d][0])
            ind_sp[h].append(fwd[d][1])

    result = {}
    for h in HORIZONS:
        n = len(cum_excess[h])
        if n == 0:
            result[h] = {"cum_excess": 0, "ind_excess": 0, "ind_stock": 0, "ind_sp": 0, "n": 0}
        else:
            result[h] = {
                "cum_excess": sum(cum_excess[h]) / n,
                "ind_excess": sum(ind_excess[h]) / n,
                "ind_stock": sum(ind_stock[h]) / n,
                "ind_sp": sum(ind_sp[h]) / n,
                "n": n,
            }
    return result

def continuation_rate(event_rows, max_horizon=5):
    """% of events where day+N excess return has same sign as event score."""
    rates = {h: {"cont": 0, "total": 0} for h in HORIZONS}
    for row, score in event_rows:
        fwd = forward_returns(row, max_horizon)
        if fwd is None:
            continue
        for d in range(max_horizon):
            h = d + 1
            excess = fwd[d][0] - fwd[d][1]
            rates[h]["total"] += 1
            if (score > 0 and excess > 0) or (score < 0 and excess < 0):
                rates[h]["cont"] += 1
    return {h: (rates[h]["cont"] / rates[h]["total"] * 100 if rates[h]["total"] > 0 else 0)
            for h in HORIZONS}

# --- Print results for multiple K values ---

FORWARD_K_VALUES = [20, 50, 100]

def build_rankings_for_k(k):
    """Build top-K / bottom-K rankings for all metrics and periods."""
    rk = {}
    for m in METRICS:
        for p in PERIODS:
            col = f"{m}_{p}"
            sr = sorted(rows, key=lambda r: r[col])
            rk[(m, p, "low")] = [(r, r[col]) for r in sr[:k]]
            rk[(m, p, "high")] = [(r, r[col]) for r in sr[-k:][::-1]]
    return rk

for fk in FORWARD_K_VALUES:
    rk = build_rankings_for_k(fk)

    print(f"\n{'#' * 100}")
    print(f"# FORWARD RETURN ANALYSIS — K={fk} (top/bottom {fk} events)")
    print(f"#   Cumulative = compounded day+1..+N. Individual = single day return.")
    print(f"{'#' * 100}")

    for p in PERIODS:
        for direction, label in [("high", f"TOP {fk} (positive scores)"), ("low", f"BOTTOM {fk} (negative scores)")]:
            # --- Cumulative excess returns ---
            print(f"\n{'-' * 100}")
            print(f"  {p.upper()} — {label} — Mean cumulative excess return (stock - SP500) in bps")
            print(f"{'-' * 100}")
            header = f"{'Metric':<8} {'N':>6}"
            for h in HORIZONS:
                header += f" {'Day+' + str(h):>10}"
            print(header)

            for m in METRICS:
                stats = compute_forward_stats(rk[(m, p, direction)])
                line = f"{m:<8} {stats[HORIZONS[0]]['n']:>6}"
                for h in HORIZONS:
                    line += f" {stats[h]['cum_excess'] * 10000:>+10.1f}"
                print(line)

            # --- Individual daily excess returns ---
            print(f"\n  {p.upper()} — {label} — Mean individual daily excess return (bps)")
            header = f"{'Metric':<8} {'N':>6}"
            for h in HORIZONS:
                header += f" {'Day+' + str(h):>10}"
            print(header)

            for m in METRICS:
                stats = compute_forward_stats(rk[(m, p, direction)])
                line = f"{m:<8} {stats[HORIZONS[0]]['n']:>6}"
                for h in HORIZONS:
                    line += f" {stats[h]['ind_excess'] * 10000:>+10.1f}"
                print(line)

            # --- Individual daily stock returns (raw, not excess) ---
            print(f"\n  {p.upper()} — {label} — Mean individual daily STOCK return (bps)")
            header = f"{'Metric':<8} {'N':>6}"
            for h in HORIZONS:
                header += f" {'Day+' + str(h):>10}"
            print(header)

            for m in METRICS:
                stats = compute_forward_stats(rk[(m, p, direction)])
                line = f"{m:<8} {stats[HORIZONS[0]]['n']:>6}"
                for h in HORIZONS:
                    line += f" {stats[h]['ind_stock'] * 10000:>+10.1f}"
                print(line)

    # --- Continuation rate ---

    print(f"\n{'=' * 100}")
    print(f"CONTINUATION RATE (K={fk}): % where day+N excess return has same sign as event score")
    print(f"  >50% = continuation (news), <50% = reversal (noise)")
    print(f"{'=' * 100}")

    cont_summary_k = []

    for p in PERIODS:
        for direction, label in [("high", f"TOP {fk}"), ("low", f"BOTTOM {fk}")]:
            print(f"\n  {p.upper()} — {label}:")
            header = f"{'Metric':<8}"
            for h in HORIZONS:
                header += f" {'Day+' + str(h):>10}"
            print(header)

            for m in METRICS:
                rates = continuation_rate(rk[(m, p, direction)])
                line = f"{m:<8}"
                for h in HORIZONS:
                    line += f" {rates[h]:>9.1f}%"
                print(line)
                cont_summary_k.append((m, p, direction, rates))

    # --- Ranking summary for this K ---

    print(f"\n{'=' * 100}")
    print(f"RANKING SUMMARY (K={fk})")
    print(f"{'=' * 100}")

    print(f"\n  Average continuation rate at Day+1 (across top+bottom, gap+cc):")
    metric_avg_cont = {}
    for m in METRICS:
        rates_day1 = [r[3][1] for r in cont_summary_k if r[0] == m]
        metric_avg_cont[m] = sum(rates_day1) / len(rates_day1)
    for i, (m, avg) in enumerate(sorted(metric_avg_cont.items(), key=lambda x: -x[1]), 1):
        print(f"    {i}. {m:<6} {avg:.1f}%")

    print(f"\n  Average |cumulative excess return| at Day+5 (across all groups, bps):")
    metric_avg_drift = {}
    for m in METRICS:
        drifts = []
        for p in PERIODS:
            for direction in ["high", "low"]:
                stats = compute_forward_stats(rk[(m, p, direction)])
                drifts.append(abs(stats[5]["cum_excess"]) * 10000)
        metric_avg_drift[m] = sum(drifts) / len(drifts)
    for i, (m, avg) in enumerate(sorted(metric_avg_drift.items(), key=lambda x: -x[1]), 1):
        print(f"    {i}. {m:<6} {avg:.1f} bps")

    print(f"\n  Signal persistence (|day+5 ind excess| / |day+1 ind excess|, avg across groups):")
    metric_persistence = {}
    for m in METRICS:
        ratios = []
        for p in PERIODS:
            for direction in ["high", "low"]:
                stats = compute_forward_stats(rk[(m, p, direction)])
                d1 = abs(stats[1]["ind_excess"])
                d5 = abs(stats[5]["ind_excess"])
                if d1 > 1e-10:
                    ratios.append(d5 / d1)
        metric_persistence[m] = sum(ratios) / len(ratios) if ratios else 0
    for i, (m, ratio) in enumerate(sorted(metric_persistence.items(), key=lambda x: -x[1]), 1):
        print(f"    {i}. {m:<6} {ratio:.2f}x")

# --- Cross-K comparison summary ---

print(f"\n{'#' * 100}")
print(f"# CROSS-K COMPARISON: How do rankings change with stricter event selection?")
print(f"{'#' * 100}")

print(f"\n  Avg |cumulative excess return| at Day+5 by K:")
header = f"{'Metric':<8}"
for fk in FORWARD_K_VALUES:
    header += f" {'K=' + str(fk):>10}"
print(header)
for m in METRICS:
    line = f"{m:<8}"
    for fk in FORWARD_K_VALUES:
        rk = build_rankings_for_k(fk)
        drifts = []
        for p in PERIODS:
            for direction in ["high", "low"]:
                stats = compute_forward_stats(rk[(m, p, direction)])
                drifts.append(abs(stats[5]["cum_excess"]) * 10000)
        line += f" {sum(drifts)/len(drifts):>9.1f}"
    print(line)

print(f"\n  Avg continuation rate at Day+1 by K:")
header = f"{'Metric':<8}"
for fk in FORWARD_K_VALUES:
    header += f" {'K=' + str(fk):>10}"
print(header)
for m in METRICS:
    line = f"{m:<8}"
    for fk in FORWARD_K_VALUES:
        rk = build_rankings_for_k(fk)
        rates = []
        for p in PERIODS:
            for direction in ["high", "low"]:
                r = continuation_rate(rk[(m, p, direction)])
                rates.append(r[1])
        line += f" {sum(rates)/len(rates):>9.1f}%"
    print(line)

print(f"\n  Avg continuation rate at Day+3 by K:")
header = f"{'Metric':<8}"
for fk in FORWARD_K_VALUES:
    header += f" {'K=' + str(fk):>10}"
print(header)
for m in METRICS:
    line = f"{m:<8}"
    for fk in FORWARD_K_VALUES:
        rk = build_rankings_for_k(fk)
        rates = []
        for p in PERIODS:
            for direction in ["high", "low"]:
                r = continuation_rate(rk[(m, p, direction)])
                rates.append(r[3])
        line += f" {sum(rates)/len(rates):>9.1f}%"
    print(line)

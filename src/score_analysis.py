import csv
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "data", "output", "scores_output.csv")

# --- Load data ---
rows = []
with open(INPUT, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        row = {"date": r["date"], "ticker": r["ticker"]}
        for k in ["A_gap", "A_intra", "D_gap", "D_intra", "E_gap", "E_intra",
                   "Ev_gap", "Ev_intra", "Dv_gap", "Dv_intra",
                   "stock_gap", "stock_intra", "sp_gap", "sp_intra", "zv"]:
            row[k] = float(r[k])
        row["headline_gap"] = r.get("headline_gap", "")
        row["headline_intra"] = r.get("headline_intra", "")
        rows.append(row)

METRICS = ["A", "D", "E", "Ev", "Dv"]
PERIODS = ["gap", "intra"]
N = 50

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

print(f"\n{'':>8} {'High gap':>12} {'Low gap':>12} {'High intra':>12} {'Low intra':>12}")
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

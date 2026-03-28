import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(BASE_DIR, "data", "output", "scores_output.csv")

# --- Load data ---
rows = []
with open(INPUT, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

METRICS = ["A", "E", "Ev"]
N = 50

# --- Group rows by ticker ---
from collections import defaultdict
ticker_rows = defaultdict(list)
for r in rows:
    ticker_rows[r["ticker"]].append(r)

tickers = sorted(ticker_rows.keys())

def get_top_n(rows, col, n, direction):
    """direction='high' or 'low'"""
    if direction == "high":
        return sorted(rows, key=lambda r: float(r[col]), reverse=True)[:n]
    else:
        return sorted(rows, key=lambda r: float(r[col]))[:n]

def row_key(r):
    return (r["date"], r["ticker"])

for ticker in tickers:
    t_rows = ticker_rows[ticker]
    n = min(N, len(t_rows))

    print(f"\n{'#' * 130}")
    print(f"  {ticker} ({len(t_rows)} trading days, top/bottom {n})")
    print(f"{'#' * 130}")

    for period in ["gap", "cc"]:
        for direction in ["high", "low"]:
            label = f"{'TOP' if direction == 'high' else 'BOTTOM'} {n} {period.upper()}"
            print(f"\n{'=' * 130}")
            print(f"  {ticker} -- {label} -- Rank differences between A, D, Dv")
            print(f"{'=' * 130}")

            # Get rankings for each metric within this ticker
            rankings = {}
            rank_maps = {}
            for m in METRICS:
                col = f"{m}_{period}"
                ranked = get_top_n(t_rows, col, n, direction)
                rankings[m] = ranked
                rank_maps[m] = {}
                for i, r in enumerate(ranked):
                    rank_maps[m][row_key(r)] = i + 1

            # Find all unique events across all 3 metrics' top N
            all_keys = set()
            for m in METRICS:
                for r in rankings[m]:
                    all_keys.add(row_key(r))

            # For each event, get its rank in each metric (None if not in top N)
            event_data = []
            for key in all_keys:
                ranks = {}
                for m in METRICS:
                    ranks[m] = rank_maps[m].get(key, None)
                in_metrics = [m for m in METRICS if ranks[m] is not None]
                out_metrics = [m for m in METRICS if ranks[m] is None]

                if len(in_metrics) == len(METRICS):
                    vals = [ranks[m] for m in METRICS]
                    if max(vals) - min(vals) > 0:
                        event_data.append((key, ranks, "RANK_DIFF"))
                elif len(in_metrics) > 0 and len(out_metrics) > 0:
                    event_data.append((key, ranks, "MISSING"))

            # Sort by most divergent first
            def sort_key(item):
                key, ranks, typ = item
                if typ == "MISSING":
                    return (0, -len([m for m in METRICS if ranks[m] is not None]))
                else:
                    vals = [ranks[m] for m in METRICS]
                    return (1, -(max(vals) - min(vals)))

            event_data.sort(key=sort_key)

            if not event_data:
                print(f"  No differences -- all 3 metrics agree on the same top {n}.")
                continue

            row_lookup = {row_key(r): r for r in t_rows}

            print(f"\n{'':>4} {'Date':>12} {'Stock%':>8} {'SP500%':>8}", end="")
            for m in METRICS:
                print(f" {m+'_rank':>8} {m+'_score':>10}", end="")
            print(f" {'Type':>10}  Headlines")
            print("-" * 130)

            for key, ranks, typ in event_data:
                r = row_lookup[key]
                stock_pct = float(r[f"stock_{period}"]) * 100
                sp_pct = float(r[f"sp_{period}"]) * 100
                hl_col = f"headline_{period}"
                hl = r.get(hl_col, "")[:60] if r.get(hl_col, "") else "(no news)"

                print(f"{'':>4} {r['date']:>12} {stock_pct:>+8.2f} {sp_pct:>+8.2f}", end="")
                for m in METRICS:
                    rank = ranks[m]
                    score = float(r[f"{m}_{period}"])
                    rank_str = f"#{rank}" if rank else "---"
                    print(f" {rank_str:>8} {score:>10.1f}", end="")
                print(f" {typ:>10}  {hl}")

            missing_count = sum(1 for _, _, t in event_data if t == "MISSING")
            diff_count = sum(1 for _, _, t in event_data if t == "RANK_DIFF")
            print(f"\n  Summary: {missing_count} events in some but not all metrics, {diff_count} with rank differences")
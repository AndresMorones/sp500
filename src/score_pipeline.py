import csv
import math
import os
from datetime import datetime

from common import (
    WINDOW, MIN_PERIODS, EPSILON, BASE_DIR, RAW_DIR, OUT_DIR,
    STOCK_HISTORY_FILES, SCORE_FNS,
    sign, mean, std, median, ols, residual_std, parse_investing_vol,
)

# --- 1. Load S&P 500 data (new extended file) ---

sp500 = {}  # date_str -> {open, close}
with open(os.path.join(RAW_DIR, "S&P 500 Historical Data.csv"), encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        dt = datetime.strptime(row["Date"], "%m/%d/%Y")
        date_str = dt.strftime("%Y-%m-%d")
        sp500[date_str] = {
            "open": float(row["Open"].replace(",", "")),
            "close": float(row["Price"].replace(",", "")),
        }

# --- 2. Load price.csv to get authoritative dates per ticker ---

price_dates = {}  # ticker -> set of dates
price_data = {}   # (ticker, date) -> {open, close, volume}
with open(os.path.join(RAW_DIR, "price.csv")) as f:
    for row in csv.DictReader(f):
        ticker = row["ticker"]
        date = row["date"]
        if ticker not in price_dates:
            price_dates[ticker] = set()
        price_dates[ticker].add(date)
        price_data[(ticker, date)] = {
            "open": float(row["open"]),
            "close": float(row["close"]),
            "volume": int(float(row["volume"])),
        }

# --- 3. Load individual stock history files for pre-price.csv dates ---

for ticker, filename in STOCK_HISTORY_FILES.items():
    with open(filename, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            dt = datetime.strptime(row["Date"], "%m/%d/%Y")
            date_str = dt.strftime("%Y-%m-%d")
            key = (ticker, date_str)
            if key in price_data:
                continue  # price.csv is authoritative
            if date_str not in sp500:
                continue  # need SP500 data for this date
            price_data[key] = {
                "open": float(row["Open"].replace(",", "")),
                "close": float(row["Price"].replace(",", "")),  # Price = close
                "volume": parse_investing_vol(row.get("Vol.", "")),
            }

# --- 3b. Load news.csv and map to trading days ---
# Pre-market (00:00-09:29) -> today's gap
# Market hours (09:30-15:59) -> today's close-to-close
# After close (16:00-23:59) -> next trading day's gap

MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR = 16

# Build sorted trading dates per ticker for next-day lookup
trading_dates_per_ticker = {}
for ticker in price_dates:
    trading_dates_per_ticker[ticker] = sorted(price_dates[ticker])

def next_trading_day(ticker, date_str):
    """Find the next trading day after date_str for this ticker."""
    dates = trading_dates_per_ticker.get(ticker, [])
    for d in dates:
        if d > date_str:
            return d
    return None

news_gap = {}   # (ticker, date) -> [(headline, summary), ...]
news_cc = {}    # (ticker, date) -> [(headline, summary), ...] close-to-close (market hours news)

with open(os.path.join(RAW_DIR, "news.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        ticker = row["ticker"]
        if ticker not in price_dates:
            continue
        dt = datetime.strptime(row["datetime"], "%Y-%m-%d %H:%M:%S")
        date_str = dt.strftime("%Y-%m-%d")
        headline = row["headline"].strip()
        summary = row["summary"].strip()
        hour, minute = dt.hour, dt.minute

        if hour < MARKET_OPEN_HOUR or (hour == MARKET_OPEN_HOUR and minute < MARKET_OPEN_MIN):
            # Pre-market -> today's gap
            target_date = date_str
            bucket = news_gap
        elif hour < MARKET_CLOSE_HOUR:
            # Market hours -> today's close-to-close
            target_date = date_str
            bucket = news_cc
        else:
            # After close -> next trading day's gap
            target_date = next_trading_day(ticker, date_str)
            if target_date is None:
                continue
            bucket = news_gap

        key = (ticker, target_date)
        if key not in bucket:
            bucket[key] = []
        bucket[key].append((headline, summary))

# cc return spans the full day (prev close -> close), which includes the gap.
# Any news that drove the gap also affected the cc return — merge gap into cc.
for key, articles in news_gap.items():
    if key not in news_cc:
        news_cc[key] = []
    news_cc[key].extend(articles)

news_gap_count = sum(len(v) for v in news_gap.values())
news_cc_count = sum(len(v) for v in news_cc.values())
print(f"News loaded: {news_gap_count} gap articles, {news_cc_count} cc articles (cc includes gap news)")

# --- 4. Build per-ticker timelines (history + price.csv dates) ---

all_tickers = sorted(price_dates.keys())

stocks = {}  # ticker -> sorted list of day dicts
for ticker in all_tickers:
    target_dates = price_dates[ticker]
    # Collect all dates for this ticker that have both stock and SP500 data
    all_dates_for_ticker = sorted(set(
        d for (t, d) in price_data if t == ticker and d in sp500
    ))
    # Find the earliest target date
    min_target = min(target_dates)
    # We need WINDOW days before min_target for rolling stats
    # Keep all dates from the history that are before min_target, plus all target dates
    days_list = []
    for date in all_dates_for_ticker:
        pd = price_data[(ticker, date)]
        days_list.append({
            "date": date,
            "open": pd["open"],
            "close": pd["close"],
            "volume": pd["volume"],
            "sp_open": sp500[date]["open"],
            "sp_close": sp500[date]["close"],
            "is_target": date in target_dates,
        })
    stocks[ticker] = days_list

# --- 5. Compute returns ---

for ticker, days in stocks.items():
    for i, d in enumerate(days):
        if i == 0:
            d["stock_gap"] = None
            d["stock_cc"] = None
            d["sp_gap"] = None
            d["sp_cc"] = None
        else:
            prev = days[i - 1]
            d["stock_gap"] = (d["open"] - prev["close"]) / prev["close"] if prev["close"] else 0
            d["stock_cc"] = (d["close"] - prev["close"]) / prev["close"] if prev["close"] else 0
            d["sp_gap"] = (d["sp_open"] - prev["sp_close"]) / prev["sp_close"] if prev["sp_close"] else 0
            d["sp_cc"] = (d["sp_close"] - prev["sp_close"]) / prev["sp_close"] if prev["sp_close"] else 0

# --- 6. Rolling stats, z-scores, scores — only output target dates ---

results = []

for ticker, days in stocks.items():
    for i in range(WINDOW, len(days)):
        d = days[i]
        if not d["is_target"]:
            continue
        if d["stock_gap"] is None or d["stock_cc"] is None:
            continue

        window = days[i - WINDOW:i]
        w_gap_stock = [w["stock_gap"] for w in window if w["stock_gap"] is not None]
        w_gap_sp = [w["sp_gap"] for w in window if w["sp_gap"] is not None]
        w_cc_stock = [w["stock_cc"] for w in window if w["stock_cc"] is not None]
        w_cc_sp = [w["sp_cc"] for w in window if w["sp_cc"] is not None]
        w_lnvol = [math.log(w["volume"]) for w in window if w["volume"] > 0]

        if len(w_gap_stock) < MIN_PERIODS or len(w_gap_sp) < MIN_PERIODS:
            continue
        if len(w_cc_stock) < MIN_PERIODS or len(w_cc_sp) < MIN_PERIODS:
            continue

        n_gap = min(len(w_gap_stock), len(w_gap_sp))
        w_gap_stock = w_gap_stock[-n_gap:]
        w_gap_sp = w_gap_sp[-n_gap:]

        n_cc = min(len(w_cc_stock), len(w_cc_sp))
        w_cc_stock = w_cc_stock[-n_cc:]
        w_cc_sp = w_cc_sp[-n_cc:]

        # Beta, alpha, residual std for gap
        alpha_gap, beta_gap = ols(w_gap_sp, w_gap_stock)
        s0_gap = residual_std(w_gap_sp, w_gap_stock, alpha_gap, beta_gap)
        sown_gap = std(w_gap_stock)
        med_gap = median(w_gap_stock)

        # Beta, alpha, residual std for close-to-close
        alpha_cc, beta_cc = ols(w_cc_sp, w_cc_stock)
        s0_cc = residual_std(w_cc_sp, w_cc_stock, alpha_cc, beta_cc)
        sown_cc = std(w_cc_stock)
        med_cc = median(w_cc_stock)

        # Volume z-score (log space)
        if len(w_lnvol) > 1:
            avg_lnvol = mean(w_lnvol)
            std_lnvol = std(w_lnvol)
        else:
            avg_lnvol = 0
            std_lnvol = 1.0
        lnvol_today = math.log(d["volume"]) if d["volume"] > 0 else avg_lnvol
        zv = (lnvol_today - avg_lnvol) / max(std_lnvol, EPSILON)

        # Gap z-scores
        expected_gap = alpha_gap + beta_gap * d["sp_gap"]
        zi_gap = (d["stock_gap"] - expected_gap) / max(s0_gap, EPSILON)
        zo_gap = (d["stock_gap"] - med_gap) / max(sown_gap, EPSILON)

        # Close-to-close z-scores
        expected_cc = alpha_cc + beta_cc * d["sp_cc"]
        zi_cc = (d["stock_cc"] - expected_cc) / max(s0_cc, EPSILON)
        zo_cc = (d["stock_cc"] - med_cc) / max(sown_cc, EPSILON)

        row = {
            "date": d["date"],
            "ticker": ticker,
            "stock_gap": d["stock_gap"],
            "stock_cc": d["stock_cc"],
            "sp_gap": d["sp_gap"],
            "sp_cc": d["sp_cc"],
            "volume": d["volume"],
            "beta_gap": beta_gap,
            "beta_cc": beta_cc,
            "alpha_gap": alpha_gap,
            "alpha_cc": alpha_cc,
            "s0_gap": s0_gap,
            "s0_cc": s0_cc,
            "zv": zv,
        }

        for name, fn in SCORE_FNS:
            row[f"{name}_gap"] = fn(zi_gap, zo_gap, zv)
            row[f"{name}_cc"] = fn(zi_cc, zo_cc, zv)

        # News columns
        nkey = (ticker, d["date"])
        gap_articles = news_gap.get(nkey, [])
        cc_articles = news_cc.get(nkey, [])
        row["headline_gap"] = " | ".join(h for h, s in gap_articles) if gap_articles else ""
        row["summary_gap"] = " | ".join(s for h, s in gap_articles) if gap_articles else ""
        row["headline_cc"] = " | ".join(h for h, s in cc_articles) if cc_articles else ""
        row["summary_cc"] = " | ".join(s for h, s in cc_articles) if cc_articles else ""

        results.append(row)

# --- 7. Output CSV ---

fieldnames = [
    "date", "ticker", "stock_gap", "stock_cc", "sp_gap", "sp_cc",
    "volume", "beta_gap", "beta_cc",
    "alpha_gap", "alpha_cc", "s0_gap", "s0_cc", "zv",
]
for name, _ in SCORE_FNS:
    fieldnames.extend([f"{name}_gap", f"{name}_cc"])
fieldnames.extend(["headline_gap", "summary_gap", "headline_cc", "summary_cc"])

with open(os.path.join(OUT_DIR, "scores_output.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in results:
        out = {}
        for k in fieldnames:
            v = r[k]
            if isinstance(v, float):
                out[k] = f"{v:.6f}"
            else:
                out[k] = v
        w.writerow(out)

print(f"Wrote {len(results)} rows to scores_output.csv")
print(f"Tickers: {sorted(set(r['ticker'] for r in results))}")
print(f"Date range: {results[0]['date']} to {results[-1]['date']}")

# Check for NaN
nan_count = sum(1 for r in results for k, v in r.items() if isinstance(v, float) and math.isnan(v))
print(f"NaN values: {nan_count}")

# Per-ticker counts
for t in sorted(set(r["ticker"] for r in results)):
    rows = [r for r in results if r["ticker"] == t]
    print(f"  {t}: {len(rows)} rows, dates {rows[0]['date']} to {rows[-1]['date']}, "
          f"beta_gap [{min(r['beta_gap'] for r in rows):.2f}, {max(r['beta_gap'] for r in rows):.2f}]")

# --- 8. Print sample for AAPL ---

print(f"\n{'='*120}")
print(f"Sample: AAPL (last 10 scored days)")
print(f"{'='*120}")

aapl = [r for r in results if r["ticker"] == "AAPL"][-10:]

header = f"{'Date':>12} {'Gap%':>7} {'CC%':>7} {'SP Gap%':>7} {'SP CC%':>7} {'Beta_g':>6} {'Beta_cc':>7} {'zv':>5}"
for name, _ in SCORE_FNS:
    header += f" {name+'_g':>8} {name+'_cc':>8}"
print(header)
print("-" * len(header))

for r in aapl:
    line = (f"{r['date']:>12} {r['stock_gap']*100:>7.2f} {r['stock_cc']*100:>7.2f} "
            f"{r['sp_gap']*100:>7.2f} {r['sp_cc']*100:>7.2f} "
            f"{r['beta_gap']:>6.2f} {r['beta_cc']:>7.2f} {r['zv']:>5.1f}")
    for name, _ in SCORE_FNS:
        line += f" {r[name+'_gap']:>8.1f} {r[name+'_cc']:>8.1f}"
    print(line)

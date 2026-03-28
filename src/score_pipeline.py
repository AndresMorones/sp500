import csv
import math
import os
from datetime import datetime

# --- Config ---
WINDOW = 120  # rolling estimation window (MacKinlay 1997, Kolari et al. 2021)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")

# Mapping: ticker -> individual stock history file
STOCK_HISTORY_FILES = {
    "AAPL": os.path.join(RAW_DIR, "Apple Stock Price History.csv"),
    "AMZN": os.path.join(RAW_DIR, "Amazon.com Stock Price History.csv"),
    "GOOGL": os.path.join(RAW_DIR, "Alphabet A Stock Price History.csv"),
    "META": os.path.join(RAW_DIR, "Meta Platforms Stock Price History.csv"),
    "MSFT": os.path.join(RAW_DIR, "Microsoft Stock Price History.csv"),
    "NVDA": os.path.join(RAW_DIR, "NVIDIA Stock Price History.csv"),
    "TSLA": os.path.join(RAW_DIR, "Tesla Stock Price History.csv"),
}

# --- Helpers ---

def sign(x):
    return (1 if x > 0 else -1 if x < 0 else 0)

def mean(xs):
    return sum(xs) / len(xs)

def var(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)

def std(xs):
    return math.sqrt(var(xs))

def cov(xs, ys):
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def ols_beta_alpha(xs, ys):
    """OLS: y = alpha + beta * x. Returns (beta, alpha)."""
    v = var(xs)
    if v < 1e-16:
        return 0.0, mean(ys)
    b = cov(xs, ys) / v
    a = mean(ys) - b * mean(xs)
    return b, a

def residual_std(xs, ys, beta, alpha):
    """Std dev of residuals: y - (alpha + beta*x)."""
    resids = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    return std(resids)

def parse_investing_vol(vol_str):
    """Parse volume strings like '33.50M', '1.23B', '' -> int."""
    vol_str = vol_str.strip()
    if not vol_str or vol_str == "-":
        return 0
    multiplier = 1
    if vol_str.endswith("M"):
        multiplier = 1_000_000
        vol_str = vol_str[:-1]
    elif vol_str.endswith("B"):
        multiplier = 1_000_000_000
        vol_str = vol_str[:-1]
    elif vol_str.endswith("K"):
        multiplier = 1_000
        vol_str = vol_str[:-1]
    return int(float(vol_str.replace(",", "")) * multiplier)

# --- Scoring functions ---

def score_a(zi, zo, zv):
    return sign(zi) * zi * zi

def score_d(zi, zo, zv):
    return sign(zi) * zi * zi * max(1, abs(zo))

def score_e(zi, zo, zv):
    return sign(zo) * zo * zo

def score_ev(zi, zo, zv):
    return sign(zo) * math.sqrt(zo * zo + zv * zv)

def score_dv(zi, zo, zv):
    d = zi * zi * max(1, abs(zo))
    return sign(zi) * math.sqrt(d * d + zv * zv)

SCORE_FNS = [
    ("A", score_a),
    ("D", score_d),
    ("E", score_e),
    ("Ev", score_ev),
    ("Dv", score_dv),
]

# --- 1. Load S&P 500 data (new extended file) ---

sp500 = {}  # date_str -> {open, close}
with open(os.path.join(RAW_DIR, "S&P 500 Historical Data (1).csv"), encoding="utf-8-sig") as f:
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
# Market hours (09:30-15:59) -> today's intraday
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
news_intra = {} # (ticker, date) -> [(headline, summary), ...]

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
            # Market hours -> today's intraday
            target_date = date_str
            bucket = news_intra
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

news_gap_count = sum(len(v) for v in news_gap.values())
news_intra_count = sum(len(v) for v in news_intra.values())
print(f"News loaded: {news_gap_count} gap articles, {news_intra_count} intraday articles")

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
            d["stock_intra"] = (d["close"] - d["open"]) / d["open"] if d["open"] else 0
            d["sp_gap"] = None
            d["sp_intra"] = (d["sp_close"] - d["sp_open"]) / d["sp_open"] if d["sp_open"] else 0
        else:
            prev = days[i - 1]
            d["stock_gap"] = (d["open"] - prev["close"]) / prev["close"] if prev["close"] else 0
            d["stock_intra"] = (d["close"] - d["open"]) / d["open"] if d["open"] else 0
            d["sp_gap"] = (d["sp_open"] - prev["sp_close"]) / prev["sp_close"] if prev["sp_close"] else 0
            d["sp_intra"] = (d["sp_close"] - d["sp_open"]) / d["sp_open"] if d["sp_open"] else 0

# --- 6. Rolling stats, z-scores, scores — only output target dates ---

results = []

for ticker, days in stocks.items():
    for i in range(WINDOW, len(days)):
        d = days[i]
        if not d["is_target"]:
            continue
        if d["stock_gap"] is None:
            continue

        window = days[i - WINDOW:i]
        w_gap_stock = [w["stock_gap"] for w in window if w["stock_gap"] is not None]
        w_gap_sp = [w["sp_gap"] for w in window if w["sp_gap"] is not None]
        w_intra_stock = [w["stock_intra"] for w in window]
        w_intra_sp = [w["sp_intra"] for w in window]
        w_lnvol = [math.log(w["volume"]) for w in window if w["volume"] > 0]

        if len(w_gap_stock) < 30 or len(w_gap_sp) < 30:
            continue

        n_gap = min(len(w_gap_stock), len(w_gap_sp))
        w_gap_stock = w_gap_stock[-n_gap:]
        w_gap_sp = w_gap_sp[-n_gap:]

        # Beta, alpha, residual std for gap
        beta_gap, alpha_gap = ols_beta_alpha(w_gap_sp, w_gap_stock)
        s0_gap = residual_std(w_gap_sp, w_gap_stock, beta_gap, alpha_gap)
        sown_gap = std(w_gap_stock)
        med_gap = median(w_gap_stock)

        # Beta, alpha, residual std for intraday
        beta_intra, alpha_intra = ols_beta_alpha(w_intra_sp, w_intra_stock)
        s0_intra = residual_std(w_intra_sp, w_intra_stock, beta_intra, alpha_intra)
        sown_intra = std(w_intra_stock)
        med_intra = median(w_intra_stock)

        # Volume z-score (log space)
        if len(w_lnvol) > 1:
            avg_lnvol = mean(w_lnvol)
            std_lnvol = std(w_lnvol)
        else:
            avg_lnvol = 0
            std_lnvol = 1.0
        lnvol_today = math.log(d["volume"]) if d["volume"] > 0 else avg_lnvol
        zv = (lnvol_today - avg_lnvol) / max(std_lnvol, 1e-8)

        # Gap z-scores
        expected_gap = alpha_gap + beta_gap * d["sp_gap"]
        zi_gap = (d["stock_gap"] - expected_gap) / max(s0_gap, 1e-8)
        zo_gap = (d["stock_gap"] - med_gap) / max(sown_gap, 1e-8)

        # Intraday z-scores
        expected_intra = alpha_intra + beta_intra * d["sp_intra"]
        zi_intra = (d["stock_intra"] - expected_intra) / max(s0_intra, 1e-8)
        zo_intra = (d["stock_intra"] - med_intra) / max(sown_intra, 1e-8)

        row = {
            "date": d["date"],
            "ticker": ticker,
            "stock_gap": d["stock_gap"],
            "stock_intra": d["stock_intra"],
            "sp_gap": d["sp_gap"],
            "sp_intra": d["sp_intra"],
            "volume": d["volume"],
            "beta_gap": beta_gap,
            "beta_intra": beta_intra,
            "zv": zv,
        }

        for name, fn in SCORE_FNS:
            row[f"{name}_gap"] = fn(zi_gap, zo_gap, zv)
            row[f"{name}_intra"] = fn(zi_intra, zo_intra, zv)

        # News columns
        nkey = (ticker, d["date"])
        gap_articles = news_gap.get(nkey, [])
        intra_articles = news_intra.get(nkey, [])
        row["headline_gap"] = " | ".join(h for h, s in gap_articles) if gap_articles else ""
        row["summary_gap"] = " | ".join(s for h, s in gap_articles) if gap_articles else ""
        row["headline_intra"] = " | ".join(h for h, s in intra_articles) if intra_articles else ""
        row["summary_intra"] = " | ".join(s for h, s in intra_articles) if intra_articles else ""

        results.append(row)

# --- 7. Output CSV ---

fieldnames = [
    "date", "ticker", "stock_gap", "stock_intra", "sp_gap", "sp_intra",
    "volume", "beta_gap", "beta_intra", "zv",
]
for name, _ in SCORE_FNS:
    fieldnames.extend([f"{name}_gap", f"{name}_intra"])
fieldnames.extend(["headline_gap", "summary_gap", "headline_intra", "summary_intra"])

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

header = f"{'Date':>12} {'Gap%':>7} {'Intra%':>7} {'SP Gap%':>7} {'SP Int%':>7} {'Beta_g':>6} {'zv':>5}"
for name, _ in SCORE_FNS:
    header += f" {name+'_g':>8} {name+'_i':>8}"
print(header)
print("-" * len(header))

for r in aapl:
    line = (f"{r['date']:>12} {r['stock_gap']*100:>7.2f} {r['stock_intra']*100:>7.2f} "
            f"{r['sp_gap']*100:>7.2f} {r['sp_intra']*100:>7.2f} "
            f"{r['beta_gap']:>6.2f} {r['zv']:>5.1f}")
    for name, _ in SCORE_FNS:
        line += f" {r[name+'_gap']:>8.1f} {r[name+'_intra']:>8.1f}"
    print(line)
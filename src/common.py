"""Shared utilities for the sp500 scoring pipeline.

Contains: math helpers, OLS regression, scoring functions, data loading,
and project-wide constants.
"""
import csv
import math
import os
from datetime import datetime

# --- Config ---
WINDOW = 120  # rolling estimation window (MacKinlay 1997, Kolari et al. 2021)
MIN_PERIODS = WINDOW  # require full window — history files provide enough pre-target data
EPSILON = 1e-8  # floor for division to avoid div-by-zero

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")

STOCK_HISTORY_FILES = {
    "AAPL": os.path.join(RAW_DIR, "Apple Stock Price History.csv"),
    "AMZN": os.path.join(RAW_DIR, "Amazon.com Stock Price History.csv"),
    "GOOGL": os.path.join(RAW_DIR, "Alphabet A Stock Price History.csv"),
    "META": os.path.join(RAW_DIR, "Meta Platforms Stock Price History.csv"),
    "MSFT": os.path.join(RAW_DIR, "Microsoft Stock Price History.csv"),
    "NVDA": os.path.join(RAW_DIR, "NVIDIA Stock Price History.csv"),
    "TSLA": os.path.join(RAW_DIR, "Tesla Stock Price History.csv"),
}

# --- Math helpers ---

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

# --- OLS regression ---

def ols(xs, ys):
    """OLS: y = alpha + beta * x. Returns (alpha, beta)."""
    v = var(xs)
    if v < 1e-16:
        return mean(ys), 0.0
    b = cov(xs, ys) / v
    a = mean(ys) - b * mean(xs)
    return a, b

def ols_asymmetric(rm_list, rs_list):
    """Piecewise OLS: rs = alpha + bu * max(rm,0) + bd * min(rm,0).
    Returns (alpha, bu, bd). Uses normal equations for 2 regressors + intercept."""
    n = len(rm_list)
    ones = [1.0] * n
    rm_pos = [max(r, 0) for r in rm_list]
    rm_neg = [min(r, 0) for r in rm_list]

    x = [ones, rm_pos, rm_neg]
    y = rs_list

    xtx = [[sum(x[i][k] * x[j][k] for k in range(n)) for j in range(3)] for i in range(3)]
    xty = [sum(x[i][k] * y[k] for k in range(n)) for i in range(3)]

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    d = det3(xtx)
    if abs(d) < 1e-16:
        return mean(rs_list), 0.0, 0.0

    def replace_col(mat, col, vec):
        return [[vec[i] if j == col else mat[i][j] for j in range(3)] for i in range(3)]

    alpha = det3(replace_col(xtx, 0, xty)) / d
    bu = det3(replace_col(xtx, 1, xty)) / d
    bd = det3(replace_col(xtx, 2, xty)) / d
    return alpha, bu, bd

def residual_std(xs, ys, alpha, beta):
    """Std dev of residuals: y - (alpha + beta*x)."""
    resids = [y - (alpha + beta * x) for x, y in zip(xs, ys)]
    return std(resids)

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

# --- Data loading ---

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

def load_sp500():
    """Load S&P 500 data. Returns dict: date_str -> {open, close}."""
    sp500 = {}
    with open(os.path.join(RAW_DIR, "S&P 500 Historical Data.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            dt = datetime.strptime(row["Date"], "%m/%d/%Y")
            date_str = dt.strftime("%Y-%m-%d")
            sp500[date_str] = {
                "open": float(row["Open"].replace(",", "")),
                "close": float(row["Price"].replace(",", "")),
            }
    return sp500

def load_sp500_close():
    """Load S&P 500 close prices only. Returns dict: date_str -> close_price."""
    sp500 = {}
    with open(os.path.join(RAW_DIR, "S&P 500 Historical Data.csv"), encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            dt = datetime.strptime(row["Date"], "%m/%d/%Y")
            date_str = dt.strftime("%Y-%m-%d")
            sp500[date_str] = float(row["Price"].replace(",", ""))
    return sp500

def load_price_csv():
    """Load price.csv. Returns (price_dates, price_data).
    price_dates: ticker -> set of dates
    price_data: (ticker, date) -> {open, close, volume}
    """
    price_dates = {}
    price_data = {}
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
    return price_dates, price_data

def load_price_csv_close():
    """Load price.csv, close prices only. Returns (price_dates, price_data).
    price_dates: ticker -> set of dates
    price_data: (ticker, date) -> close_price (float)
    """
    price_dates = {}
    price_data = {}
    with open(os.path.join(RAW_DIR, "price.csv")) as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"]
            date = row["date"]
            if ticker not in price_dates:
                price_dates[ticker] = set()
            price_dates[ticker].add(date)
            price_data[(ticker, date)] = float(row["close"])
    return price_dates, price_data

def load_stock_history(sp500_dates, price_data, include_ohlcv=True):
    """Load individual stock history files for pre-price.csv dates.
    Mutates price_data in place, adding entries not already present.
    sp500_dates: set or dict of valid SP500 dates.
    include_ohlcv: if True, store {open, close, volume}; if False, store close only.
    """
    for ticker, filename in STOCK_HISTORY_FILES.items():
        with open(filename, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                dt = datetime.strptime(row["Date"], "%m/%d/%Y")
                date_str = dt.strftime("%Y-%m-%d")
                key = (ticker, date_str)
                if key in price_data:
                    continue
                if date_str not in sp500_dates:
                    continue
                if include_ohlcv:
                    price_data[key] = {
                        "open": float(row["Open"].replace(",", "")),
                        "close": float(row["Price"].replace(",", "")),
                        "volume": parse_investing_vol(row.get("Vol.", "")),
                    }
                else:
                    price_data[key] = float(row["Price"].replace(",", ""))

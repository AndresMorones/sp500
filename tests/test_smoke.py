"""Smoke tests encoding the CLAUDE.md pipeline invariants.

Run with: python -m pytest tests/test_smoke.py -v
Or:       python tests/test_smoke.py
"""
import csv
import math
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
OUTPUT = os.path.join(BASE_DIR, "data", "output", "scores_output.csv")

EXPECTED_TICKERS = {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"}
MIN_ROWS = 1500

EXPECTED_COLUMNS = [
    "date", "ticker", "stock_gap", "stock_cc", "sp_gap", "sp_cc",
    "volume", "beta_gap", "beta_cc", "zv",
    "A_gap", "A_cc", "D_gap", "D_cc", "E_gap", "E_cc",
    "Ev_gap", "Ev_cc", "Dv_gap", "Dv_cc",
    "headline_gap", "summary_gap", "headline_cc", "summary_cc",
]

FLOAT_COLUMNS = [
    "stock_gap", "stock_cc", "sp_gap", "sp_cc",
    "beta_gap", "beta_cc", "zv",
    "A_gap", "A_cc", "D_gap", "D_cc", "E_gap", "E_cc",
    "Ev_gap", "Ev_cc", "Dv_gap", "Dv_cc",
]


def run_pipeline():
    """Run score_pipeline.py and return exit code."""
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, "score_pipeline.py")],
        capture_output=True, text=True, cwd=SRC_DIR,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
    return result.returncode


def load_output():
    """Load scores_output.csv and return list of dicts."""
    rows = []
    with open(OUTPUT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def test_pipeline_runs():
    """Pipeline exits with code 0."""
    assert run_pipeline() == 0, "score_pipeline.py failed"


def test_output_exists():
    """Output CSV was created."""
    assert os.path.exists(OUTPUT), f"Output file not found: {OUTPUT}"


def test_expected_columns():
    """Output has all expected columns."""
    rows = load_output()
    assert len(rows) > 0, "Output is empty"
    actual = set(rows[0].keys())
    missing = set(EXPECTED_COLUMNS) - actual
    assert not missing, f"Missing columns: {missing}"


def test_row_count():
    """Output has at least MIN_ROWS rows."""
    rows = load_output()
    assert len(rows) >= MIN_ROWS, f"Only {len(rows)} rows, expected >= {MIN_ROWS}"


def test_all_tickers_present():
    """All 7 tickers are present."""
    rows = load_output()
    tickers = set(r["ticker"] for r in rows)
    missing = EXPECTED_TICKERS - tickers
    assert not missing, f"Missing tickers: {missing}"


def test_no_nan():
    """No NaN values in float columns."""
    rows = load_output()
    nan_count = 0
    for r in rows:
        for col in FLOAT_COLUMNS:
            v = float(r[col])
            if math.isnan(v):
                nan_count += 1
    assert nan_count == 0, f"Found {nan_count} NaN values"


def test_aapl_beta_gap_range():
    """AAPL beta_gap is in a reasonable range [0.5, 3.0]."""
    rows = load_output()
    aapl_betas = [float(r["beta_gap"]) for r in rows if r["ticker"] == "AAPL"]
    assert len(aapl_betas) > 0, "No AAPL rows found"
    min_b, max_b = min(aapl_betas), max(aapl_betas)
    assert min_b >= 0.5, f"AAPL beta_gap min too low: {min_b:.2f}"
    assert max_b <= 3.0, f"AAPL beta_gap max too high: {max_b:.2f}"


# --- Run as script ---

if __name__ == "__main__":
    tests = [
        test_pipeline_runs,
        test_output_exists,
        test_expected_columns,
        test_row_count,
        test_all_tickers_present,
        test_no_nan,
        test_aapl_beta_gap_range,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__doc__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__doc__} — {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {t.__doc__} — {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)

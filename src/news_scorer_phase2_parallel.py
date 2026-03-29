"""Parallel orchestrator for Phase 2 day-level scoring.

Runs Phase 2 scoring for multiple tickers simultaneously using threads.
Each ticker's batched scoring is independent and can execute in parallel.

Usage:
    python src/news_scorer_phase2_parallel.py
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from news_scorer import (
    load_news, load_trading_dates, bucket_gap_cc,
    load_categories_cache, load_existing_output,
    run_phase2, OUT_DIR, OUTPUT_FILE,
)

# --- Config ---
TARGET_TICKERS = ["MSFT", "TSLA", "NVDA"]
MAX_PARALLEL = 3  # number of tickers to process simultaneously


def run_ticker_phase2(ticker, schema, gap_buckets, cc_buckets, already_scored, schema_cache):
    """Run Phase 2 for a single ticker. Thread-safe."""
    print(f"\n[{ticker}] Starting Phase 2 scoring...")
    try:
        count = run_phase2(ticker, schema, gap_buckets, cc_buckets,
                           already_scored, schema_cache)
        print(f"\n[{ticker}] DONE - scored {count} entries")
        return ticker, count
    except Exception as e:
        print(f"\n[{ticker}] FAILED: {e}")
        return ticker, 0


def main():
    print("=" * 60)
    print("PARALLEL PHASE 2 SCORING")
    print(f"Tickers: {TARGET_TICKERS}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print("=" * 60)

    # Load news and trading dates
    print("\nLoading news.csv...")
    news_by_ticker = load_news()
    for t in sorted(news_by_ticker):
        print(f"  {t}: {len(news_by_ticker[t])} articles")

    print("\nLoading trading dates and bucketing articles...")
    trading_dates = load_trading_dates()
    filtered_news = {t: news_by_ticker[t] for t in TARGET_TICKERS if t in news_by_ticker}
    gap_buckets, cc_buckets = bucket_gap_cc(filtered_news, trading_dates)

    gap_count = sum(len(v) for k, v in gap_buckets.items())
    cc_count = sum(len(v) for k, v in cc_buckets.items())
    print(f"  Gap: {gap_count} articles")
    print(f"  CC:  {cc_count} articles")

    # Load Phase 1 schemas
    schema_cache = load_categories_cache()
    print(f"\nCached schemas: {list(schema_cache.keys())}")

    # Check all tickers have schemas
    tickers_to_run = []
    for t in TARGET_TICKERS:
        if t not in schema_cache:
            print(f"  {t}: NO SCHEMA — run Phase 1 first, skipping")
        else:
            tickers_to_run.append(t)

    if not tickers_to_run:
        print("\nNo tickers ready for Phase 2.")
        return

    # Load already-scored entries (shared read, per-ticker sets for writes)
    all_scored = load_existing_output()
    print(f"Already scored: {len(all_scored)} entries")

    # Build per-ticker already_scored sets to avoid thread contention
    # Each ticker only checks/updates its own entries
    per_ticker_scored = {}
    for t in tickers_to_run:
        per_ticker_scored[t] = {(tk, d, p) for tk, d, p in all_scored if tk == t}
        remaining_gap = sum(1 for (tk, d), _ in gap_buckets.items()
                           if tk == t and (t, d, "gap") not in per_ticker_scored[t])
        remaining_cc = sum(1 for (tk, d), _ in cc_buckets.items()
                          if tk == t and (t, d, "cc") not in per_ticker_scored[t])
        total = remaining_gap + remaining_cc
        print(f"  {t}: {total} entries to score ({remaining_gap} gap + {remaining_cc} cc)")

    print(f"\nRunning Phase 2 for: {tickers_to_run}")

    # Run in parallel
    start = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(
                run_ticker_phase2, t, schema_cache[t],
                gap_buckets, cc_buckets, per_ticker_scored[t], schema_cache
            ): t
            for t in tickers_to_run
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, count = future.result()
                results[t] = count
            except Exception as e:
                print(f"\n[{ticker}] Unhandled error: {e}")
                results[ticker] = 0

    elapsed = time.time() - start

    # Summary
    print(f"\n{'='*60}")
    print(f"PARALLEL PHASE 2 COMPLETE")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output file: {OUTPUT_FILE}")
    for t in sorted(results):
        print(f"  {t}: {results[t]} entries scored")
    total = sum(results.values())
    print(f"  Total new: {total}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

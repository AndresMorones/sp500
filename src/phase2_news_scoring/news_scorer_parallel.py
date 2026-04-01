"""Parallel orchestrator for Phase 1 consensus.

Runs Phase 1 consensus for multiple tickers simultaneously using threads.
Each ticker's 3 runs + consensus merge are independent and can execute in parallel.

Usage:
    python src/news_scorer_parallel.py
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from news_scorer import (
    load_news, load_categories_cache, save_categories_cache,
    run_phase1_consensus, CATEGORIES_FILE, PHASE1_RAW_DIR,
    OUT_DIR,
)

# --- Config ---
TARGET_TICKERS = ["GOOGL"]
MAX_PARALLEL = len(TARGET_TICKERS)  # all tickers in parallel


def run_ticker_phase1(ticker, articles):
    """Run Phase 1 consensus for a single ticker. Thread-safe."""
    print(f"\n[{ticker}] Starting Phase 1 consensus...")
    try:
        consensus = run_phase1_consensus(ticker, articles)
        print(f"\n[{ticker}] DONE — {len(consensus['categories'])} total categories")
        return ticker, consensus
    except Exception as e:
        print(f"\n[{ticker}] FAILED: {e}")
        return ticker, None


def main():
    print("=" * 60)
    print("PARALLEL PHASE 1 CONSENSUS")
    print(f"Tickers: {TARGET_TICKERS}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print("=" * 60)

    # Load news
    print("\nLoading news.csv...")
    news_by_ticker = load_news()
    for t in sorted(news_by_ticker):
        print(f"  {t}: {len(news_by_ticker[t])} articles")

    # Load existing cache — skip already-done tickers
    schema_cache = load_categories_cache()
    tickers_to_run = []
    for t in TARGET_TICKERS:
        if t in schema_cache:
            print(f"\n  {t}: already in categories cache, skipping")
        elif t not in news_by_ticker:
            print(f"\n  {t}: not found in news.csv, skipping")
        else:
            tickers_to_run.append(t)

    if not tickers_to_run:
        print("\nAll tickers already cached. Nothing to do.")
        return

    print(f"\nRunning Phase 1 consensus for: {tickers_to_run}")
    os.makedirs(PHASE1_RAW_DIR, exist_ok=True)

    # Run in parallel
    start = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(run_ticker_phase1, t, news_by_ticker[t]): t
            for t in tickers_to_run
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, consensus = future.result()
                if consensus:
                    results[t] = consensus
            except Exception as e:
                print(f"\n[{ticker}] Unhandled error: {e}")

    elapsed = time.time() - start

    # Save all results to categories cache
    for ticker, consensus in results.items():
        schema_cache[ticker] = consensus
    save_categories_cache(schema_cache)

    # Summary
    print(f"\n{'='*60}")
    print(f"PARALLEL PHASE 1 COMPLETE")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Completed: {list(results.keys())}")
    print(f"  Failed: {[t for t in tickers_to_run if t not in results]}")
    print(f"  Categories file: {CATEGORIES_FILE}")

    for ticker, consensus in sorted(results.items()):
        cats = [c['id'] for c in consensus['categories']]
        print(f"\n  {ticker}:")
        print(f"    Categories ({len(cats)}): {cats}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

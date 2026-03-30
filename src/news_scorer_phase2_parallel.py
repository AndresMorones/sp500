"""Parallel Phase 2 scorer — runs all 7 tickers concurrently.

Each ticker writes to its own CSV (news_phase2_{TICKER}.csv) for:
  - No thread-safety issues (no shared file, no locks)
  - Clean resume: re-read per-ticker file on restart, skip already-scored entries
  - No column mismatch: each ticker has its own category/dimension columns

After all tickers finish, merges into the combined news_day_features.csv.

Usage:
    python src/news_scorer_phase2_parallel.py
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from news_scorer import (
    load_news, load_trading_dates, bucket_gap_cc,
    load_categories_cache, build_phase2_prompt, call_claude,
    parse_json_response, validate_phase2_response,
    ALL_TICKERS, OUT_DIR, BATCH_SIZE, SLEEP_BETWEEN,
)

# --- Config ---
MAX_PARALLEL = 7          # all tickers simultaneously
OUTPUT_FILE = os.path.join(OUT_DIR, "news_day_features.csv")


def ticker_output_path(ticker):
    """Per-ticker Phase 2 output file."""
    return os.path.join(OUT_DIR, f"news_phase2_{ticker}.csv")


def load_ticker_scored(ticker):
    """Load already-scored (date, period) pairs for a single ticker."""
    scored = set()
    path = ticker_output_path(ticker)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scored.add((row["date"], row["period"]))
    return scored


def write_ticker_rows(ticker, results, schema):
    """Append scored rows to per-ticker CSV. Creates header if file is new."""
    if not results:
        return

    cat_cols = sorted(f"cat_{c['id']}" for c in schema["categories"])
    dim_cols = sorted(d["id"] for d in schema["dimensions"])
    fieldnames = (["date", "ticker", "period", "direction", "distinct_events"]
                  + cat_cols + dim_cols + ["reasoning"])

    path = ticker_output_path(ticker)
    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = {
                "date": r.get("date", ""),
                "ticker": ticker,
                "period": r.get("period", ""),
                "direction": r.get("direction", ""),
                "distinct_events": r.get("distinct_events", 1),
                "reasoning": r.get("reasoning", ""),
            }
            for col in cat_cols:
                row[col] = r.get(col, 0)
            for dim_id in dim_cols:
                row[dim_id] = r.get(dim_id, "")
            writer.writerow(row)


def run_ticker_phase2(ticker, schema, gap_buckets, cc_buckets):
    """Run Phase 2 for a single ticker. Fully independent — own file, own resume state."""
    already_scored = load_ticker_scored(ticker)

    # Build entry list, skipping already-scored
    all_entries = []
    for (t, date), articles in sorted(gap_buckets.items()):
        if t != ticker:
            continue
        if (date, "gap") not in already_scored:
            all_entries.append((date, "gap", articles))

    for (t, date), articles in sorted(cc_buckets.items()):
        if t != ticker:
            continue
        if (date, "cc") not in already_scored:
            all_entries.append((date, "cc", articles))

    total = len(all_entries)
    already = len(already_scored)
    if total == 0:
        print(f"[{ticker}] All {already} entries already scored. Nothing to do.")
        return ticker, 0, already

    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"[{ticker}] {total} entries to score ({already} already done), "
          f"{total_batches} batches of {BATCH_SIZE}")

    scored = 0
    for batch_idx in range(0, total, BATCH_SIZE):
        batch = all_entries[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        n_articles = sum(len(arts) for _, _, arts in batch)

        print(f"[{ticker}] Batch {batch_num}/{total_batches}: "
              f"{len(batch)} entries, {n_articles} articles")

        try:
            prompt = build_phase2_prompt(ticker, schema, batch)
            response = call_claude(prompt, timeout=300)
            results = parse_json_response(response)

            if not isinstance(results, list):
                results = [results]

            validated = validate_phase2_response(results, schema)

            # Write immediately to per-ticker file
            write_ticker_rows(ticker, validated, schema)
            scored += len(validated)

            print(f"[{ticker}] Batch {batch_num}/{total_batches}: "
                  f"scored {len(validated)}/{len(batch)} "
                  f"(progress: {already + scored}/{already + total})")

        except Exception as e:
            print(f"[{ticker}] ERROR batch {batch_num}: {e}")
            print(f"[{ticker}] Skipping batch — will retry on next run.")

        time.sleep(SLEEP_BETWEEN)

    print(f"[{ticker}] DONE — scored {scored} new entries "
          f"(total: {already + scored})")
    return ticker, scored, already + scored


def merge_ticker_files(schema_cache):
    """Merge all per-ticker CSVs into the combined output file."""
    # Collect superset of all columns
    all_cat_cols = set()
    all_dim_ids = set()
    for ticker, schema in schema_cache.items():
        for c in schema["categories"]:
            all_cat_cols.add(f"cat_{c['id']}")
        for d in schema["dimensions"]:
            all_dim_ids.add(d["id"])
    all_cat_cols = sorted(all_cat_cols)
    all_dim_ids = sorted(all_dim_ids)

    fieldnames = (["date", "ticker", "period", "direction", "distinct_events"]
                  + all_cat_cols + all_dim_ids + ["reasoning"])

    all_rows = []
    for ticker in ALL_TICKERS:
        path = ticker_output_path(ticker)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_rows.append(row)

    # Sort by date, ticker, period
    all_rows.sort(key=lambda r: (r.get("date", ""), r.get("ticker", ""), r.get("period", "")))

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            # Fill missing columns with defaults
            for col in all_cat_cols:
                if col not in row or row[col] == "":
                    row[col] = 0
            for dim_id in all_dim_ids:
                if dim_id not in row or row[dim_id] == "":
                    row[dim_id] = ""
            writer.writerow(row)

    print(f"\nMerged {len(all_rows)} rows into {OUTPUT_FILE}")
    return len(all_rows)


def main():
    print("=" * 60)
    print("PARALLEL PHASE 2 SCORER")
    print(f"Tickers: {ALL_TICKERS}")
    print(f"Max parallel: {MAX_PARALLEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print("=" * 60)

    # 1. Load news + trading dates + buckets
    print("\n1. Loading news and bucketing articles...")
    news_by_ticker = load_news()
    for t in sorted(news_by_ticker):
        print(f"   {t}: {len(news_by_ticker[t])} articles")

    trading_dates = load_trading_dates()
    gap_buckets, cc_buckets = bucket_gap_cc(news_by_ticker, trading_dates)

    gap_days = sum(1 for k in gap_buckets)
    cc_days = sum(1 for k in cc_buckets)
    print(f"   Gap buckets: {gap_days} (ticker,date) pairs")
    print(f"   CC buckets:  {cc_days} (ticker,date) pairs")

    # 2. Load Phase 1 schemas
    print("\n2. Loading Phase 1 schemas...")
    schema_cache = load_categories_cache()
    missing = [t for t in ALL_TICKERS if t not in schema_cache]
    if missing:
        print(f"   ERROR: Missing Phase 1 schemas for: {missing}")
        print(f"   Run Phase 1 first for these tickers.")
        sys.exit(1)
    for t in ALL_TICKERS:
        cats = len(schema_cache[t]["categories"])
        dims = len(schema_cache[t]["dimensions"])
        print(f"   {t}: {cats} categories, {dims} dimensions")

    # 3. Show resume state
    print("\n3. Resume state:")
    for t in ALL_TICKERS:
        scored = load_ticker_scored(t)
        print(f"   {t}: {len(scored)} entries already scored")

    # 4. Run Phase 2 in parallel
    print(f"\n4. Running Phase 2 ({MAX_PARALLEL} tickers in parallel)...")
    start = time.time()
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {
            executor.submit(
                run_ticker_phase2, t, schema_cache[t], gap_buckets, cc_buckets
            ): t
            for t in ALL_TICKERS
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, new_scored, total = future.result()
                results[t] = (new_scored, total)
            except Exception as e:
                print(f"\n[{ticker}] UNHANDLED ERROR: {e}")
                results[ticker] = (0, -1)

    elapsed = time.time() - start

    # 5. Merge per-ticker files into combined output
    print(f"\n5. Merging per-ticker files...")
    total_rows = merge_ticker_files(schema_cache)

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Combined output: {OUTPUT_FILE} ({total_rows} rows)")
    print(f"\n  Per-ticker results:")
    for t in ALL_TICKERS:
        new_scored, total = results.get(t, (0, -1))
        status = f"{new_scored} new, {total} total" if total >= 0 else "FAILED"
        print(f"    {t}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

"""News scoring pipeline using Claude CLI.

Phase 1: Discover 8 business-specific categories + 12-15 impact dimensions per stock.
Phase 2: Score each (ticker, date, period) with combined day-level ratings.

Uses `claude -p` CLI (Max subscription) — no API key needed.
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

# --- Config ---
TARGET_TICKERS = "MSFT,TSLA,NVDA"  # Single ticker to validate, or "ALL" for all 7
PHASE1_ONLY = True        # Set True to run only Phase 1 (category/dimension discovery)
PHASE1_RUNS = 3           # Number of Phase 1 runs per ticker for consensus
BATCH_SIZE = 10           # (ticker, date, period) entries per Phase 2 call
SLEEP_BETWEEN = 2          # seconds between CLI calls
MAX_RETRIES = 3
RETRY_BACKOFF = 2         # seconds, doubles each retry
MODEL = "claude-opus-4-6[1m]"  # Opus 4.6 with 1M context window

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
CATEGORIES_FILE = os.path.join(OUT_DIR, "news_categories.json")
PHASE1_RAW_DIR = os.path.join(OUT_DIR, "news_phase1_raw")
PROGRESS_FILE = os.path.join(OUT_DIR, "news_scorer_progress.json")
OUTPUT_FILE = os.path.join(OUT_DIR, "news_day_features.csv")

ALL_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR = 16

# --- Suggested dimensions for Phase 1 ---
SUGGESTED_DIMENSIONS = [
    ("materiality", "1=noise, 10=direct large impact on revenue/margins/TAM"),
    ("surprise", "1=fully expected/scheduled/baked-in, 10=completely unexpected"),
    ("temporal_horizon", "1=affects next few days only, 10=structural shift for years"),
    ("sentiment_strength", "1=neutral tone, 10=extremely strong positive or negative"),
    ("information_density", "1=pure opinion/speculation, 10=hard data with specific numbers"),
    ("directional_clarity", "1=ambiguous/could go either way, 10=unambiguously good or bad"),
    ("scope", "1=only this company, 5=sector-wide, 10=macro/market-wide"),
    ("competitive_impact", "1=no competitive effect, 10=major competitive shift"),
    ("regulatory_risk", "1=no regulatory angle, 10=major legal/regulatory event"),
    ("management_signal", "1=no leadership angle, 10=major strategic/leadership signal"),
    ("expected_duration", "1=transient story that resolves within a day, 10=persistent story remaining relevant for weeks or longer"),
    ("narrative_shift", "1=reinforces existing narrative, 10=fundamentally changes thesis"),
    ("repeatedness", "1=rehash/follow-up of known story, 10=first report of new info"),
    ("actionability", "1=background context only, 10=investors can act immediately"),
    ("controversy", "1=consensus view, 10=highly divisive among investors"),
    ("financial_result_surprise", "1=no financial results reported or results exactly match expectations, 10=reported figures dramatically exceed or miss prior guidance and analyst expectations as described in the article"),
]


# --- Claude CLI wrapper ---

def call_claude(prompt, timeout=300):
    """Call Claude via CLI. Returns the text response."""
    for attempt in range(MAX_RETRIES):
        try:
            # Pass prompt via stdin ("-") to avoid OS arg length limits
            # shell=True needed on Windows so subprocess finds .cmd files in PATH
            cmd = f'claude -p - --output-format json --model {MODEL}'
            result = subprocess.run(
                cmd, shell=True,
                input=prompt,
                capture_output=True, text=True, timeout=timeout,
                encoding="utf-8",
            )
            if result.returncode != 0:
                raise RuntimeError(f"CLI error (rc={result.returncode}): {result.stderr[:500]}")
            response = json.loads(result.stdout)
            return response["result"]
        except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError, KeyError) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"  Attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"All {MAX_RETRIES} attempts failed") from e


def parse_json_response(text):
    """Extract JSON from Claude's response, handling markdown fences and preamble."""
    text = text.strip()
    if not text:
        raise ValueError("Empty response from Claude")

    # Remove markdown fences if present
    if "```" in text:
        # Find content between first ``` and last ```
        parts = text.split("```")
        # parts[1] is the content inside the first fence pair
        if len(parts) >= 3:
            inner = parts[1]
            # Remove language tag (e.g., "json\n")
            if inner.startswith("json"):
                inner = inner[4:]
            text = inner.strip()

    # Find first { or [ to handle any preamble text
    first_brace = text.find("{")
    first_bracket = text.find("[")
    starts = [i for i in [first_brace, first_bracket] if i >= 0]
    if not starts:
        raise ValueError(f"No JSON found in response: {text[:200]}...")
    start = min(starts)
    text = text[start:]

    # Find matching end — scan from the end for } or ]
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ('}', ']'):
            text = text[:i + 1]
            break

    return json.loads(text)


# --- Data loading ---

def load_news():
    """Load news.csv, group by ticker. Returns dict[ticker] -> list of article dicts."""
    by_ticker = {}
    with open(os.path.join(RAW_DIR, "news.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = row["ticker"]
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append({
                "datetime": row["datetime"],
                "headline": row["headline"].strip(),
                "summary": row["summary"].strip(),
            })
    return by_ticker


def load_trading_dates():
    """Load trading dates from price.csv + stock history files for next-day lookup."""
    from common import load_sp500, load_price_csv, load_stock_history
    sp500 = load_sp500()
    price_dates, price_data = load_price_csv()
    load_stock_history(sp500, price_data)
    # Build sorted trading dates per ticker
    # Collect all dates where we have both stock and SP500 data
    trading_dates = {}
    for (ticker, date) in price_data:
        if date in sp500:
            if ticker not in trading_dates:
                trading_dates[ticker] = set()
            trading_dates[ticker].add(date)
    return {t: sorted(dates) for t, dates in trading_dates.items()}


def bucket_gap_cc(news_by_ticker, trading_dates_per_ticker):
    """Map articles to gap/cc buckets per (ticker, date), replicating score_pipeline.py logic.

    Returns:
        gap_buckets: dict[(ticker, date)] -> list of article dicts
        cc_buckets:  dict[(ticker, date)] -> list of article dicts (includes gap articles)
    """
    gap_buckets = {}
    cc_buckets = {}

    def next_trading_day(ticker, date_str):
        dates = trading_dates_per_ticker.get(ticker, [])
        for d in dates:
            if d > date_str:
                return d
        return None

    for ticker, articles in news_by_ticker.items():
        if ticker not in trading_dates_per_ticker:
            continue
        valid_dates = set(trading_dates_per_ticker[ticker])

        for art in articles:
            dt = datetime.strptime(art["datetime"], "%Y-%m-%d %H:%M:%S")
            date_str = dt.strftime("%Y-%m-%d")
            hour, minute = dt.hour, dt.minute

            if hour < MARKET_OPEN_HOUR or (hour == MARKET_OPEN_HOUR and minute < MARKET_OPEN_MIN):
                target_date = date_str
                bucket = gap_buckets
            elif hour < MARKET_CLOSE_HOUR:
                target_date = date_str
                bucket = cc_buckets
            else:
                target_date = next_trading_day(ticker, date_str)
                if target_date is None:
                    continue
                bucket = gap_buckets

            # Only include if target_date is a valid trading day
            if target_date not in valid_dates:
                continue

            key = (ticker, target_date)
            if key not in bucket:
                bucket[key] = []
            bucket[key].append(art)

    # CC includes gap articles (cc return spans gap + intraday)
    for key, articles in gap_buckets.items():
        if key not in cc_buckets:
            cc_buckets[key] = []
        cc_buckets[key].extend(articles)

    return gap_buckets, cc_buckets


# --- Cache management ---

def load_categories_cache():
    """Load cached Phase 1 results."""
    if os.path.exists(CATEGORIES_FILE):
        with open(CATEGORIES_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_categories_cache(categories):
    """Save Phase 1 results."""
    with open(CATEGORIES_FILE, "w", encoding="utf-8") as f:
        json.dump(categories, f, indent=2)


def load_progress():
    """Load Phase 2 progress tracker."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    """Save Phase 2 progress."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def load_existing_output():
    """Load already-scored rows from output CSV to avoid duplicates."""
    scored = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scored.add((row["ticker"], row["date"], row["period"]))
    return scored


# --- Phase 1: Category & Dimension Discovery ---

def build_phase1_prompt(ticker, articles):
    """Build the Phase 1 discovery prompt for a ticker."""
    dim_table = "\n".join(f"| `{dim_id}` | {scale} |" for dim_id, scale in SUGGESTED_DIMENSIONS)

    article_lines = []
    for i, art in enumerate(articles, 1):
        hl = art["headline"]
        sm = art["summary"]
        article_lines.append(f"[{i}] {art['datetime'][:10]} | {hl} | {sm}")

    articles_text = "\n".join(article_lines)

    return f"""You are a senior equity research analyst. You will receive ALL {len(articles)} news \
articles about {ticker} from November 2023 to October 2024. Your task is to \
analyze the news landscape for this stock and produce a structured scoring framework.

IMPORTANT RULES:
- You have ZERO access to stock prices, returns, or trading data.
- Base your analysis ONLY on the content and nature of the news events.
- Do NOT speculate about what the stock price did in response.
- Think about what types of events TEND to matter for this specific company, \
based on its business model, competitive position, and investor base.

TASK 1 — DISCOVER 9 NEWS CATEGORIES

Identify exactly 9 categories from the articles you just read. These must \
include a MIX of:

- **6 company-specific categories**: Themes specific to {ticker}'s business. \
Do NOT create hyper-specific categories that only match a handful of \
articles — merge related themes into broader buckets. But do NOT merge \
genuinely distinct themes just because one has fewer articles. A category \
covering 10 important articles is better than losing that signal by folding \
it into a broader bucket.

  GOOD: "gaming_entertainment_strategy" (covers console strategy, acquisitions, \
game releases, subscription models)
  BAD: "activision_integration" (too narrow, only ~5 articles about one event)

- **1 required: "earnings_financial_results"**: Quarterly earnings reports, \
revenue and EPS figures, profit margins, guidance revisions, delivery or \
production numbers, and any article reporting specific financial results \
vs expectations. This category MUST exist even if only a few articles \
cover earnings — these are among the most important events for any stock.

- **1 required: "market_sector_sentiment"**: Broad market roundups, sector \
trends, macro news that mentions {ticker}. Examples: "Tech Up on Renewed \
Chip Optimism", "Magnificent Seven earnings preview", "Nasdaq correction." \
These articles have low company-specific materiality but reflect the \
environment the stock trades in.

- **1 required: "analyst_consensus_signals"**: Analyst upgrades/downgrades, \
price target changes, buy/sell ratings, institutional positioning. Examples: \
"Microsoft: UBS remains Buy", "Goldman raises NVDA target to $150."

Each category needs: a snake_case id, a short label, and a one-sentence \
description. Every article in the corpus should fit into at least one category.

TASK 2 — SELECT 12-15 IMPACT DIMENSIONS

Below is a pool of suggested dimensions. You must CRITICALLY EVALUATE each one \
against the actual articles you just read:
- KEEP dimensions where you found clear variation across articles (some articles \
  would score 2, others 8-9 on this dimension).
- DROP dimensions that would be uniformly low or uniformly high across most \
  articles in this corpus (uninformative for {ticker}).
- ADD custom dimensions if you spotted recurring patterns in the articles \
  that none of the suggested dimensions capture. There is no limit on custom \
  additions — add as many as the corpus warrants.

Select all dimensions that show meaningful variation — there is no upper limit.

SCALE DESCRIPTION RULES:
- Define ONLY the 1 endpoint and the 10 endpoint using general EVENT TYPES \
(not specific articles or events from this corpus).
- Do NOT pre-assign scores to specific articles, companies, or events. \
Do NOT write things like "Altman firing = 10" or "earnings beat scores 7-8."
- The middle range (2-9) is smooth interpolation — do not anchor it.
- Keep descriptions general enough to apply consistently across all articles. \
The scorer in Phase 2 will judge each article relative to others in the corpus.

IMPORTANT on the "surprise" dimension: This measures how UNEXPECTED the \
OUTCOME or CONTENT of the news was — NOT whether the event itself was \
scheduled. A scheduled event with an unexpected outcome (e.g., earnings that \
massively beat or miss) is HIGH surprise. A scheduled event that meets \
expectations is LOW surprise. Rate the surprise of what was REVEALED, not \
whether the event was on the calendar.

Suggested dimension pool (all scored 1-10):

| Dimension | Default scale |
|-----------|---------------|
{dim_table}

TASK 3 — JUSTIFY YOUR DIMENSION CHOICES

For every dimension you keep, drop, or add, provide a brief justification based \
on what you observed in the articles. Only keep ones where you saw actual \
variation. In your rationale, describe the TYPE of variation you observed (e.g., \
"ranges from routine operational updates to major strategic pivots"), but do NOT \
assign specific scores to specific events.

RESPOND WITH ONLY THIS JSON (no markdown fences, no commentary):
{{
  "ticker": "{ticker}",
  "categories": [
    {{"id": "snake_case_id", "label": "Short Label", "description": "One sentence"}},
    ... (exactly 9, including earnings_financial_results, market_sector_sentiment, and analyst_consensus_signals)
  ],
  "dimensions": [
    {{"id": "dimension_id", "scale": "1=low end description, 10=high end description"}},
    ... (12-15 total)
  ],
  "dimension_rationale": {{
    "kept": {{"dimension_id": "Type of variation observed in the corpus", ...}},
    "dropped": {{"dimension_id": "Why this dimension lacks variation for {ticker}", ...}},
    "added": {{"dimension_id": "Pattern observed that needs this dimension"}}
  }}
}}

ARTICLES FOR {ticker}:

{articles_text}"""


def run_phase1(ticker, articles):
    """Run Phase 1 for a single ticker. Returns the parsed schema dict."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Discovering categories & dimensions for {ticker}")
    print(f"  Sending {len(articles)} articles...")
    print(f"{'='*60}")

    prompt = build_phase1_prompt(ticker, articles)
    response = call_claude(prompt, timeout=600)  # large input, allow more time
    print(f"  Response length: {len(response)} chars")
    try:
        schema = parse_json_response(response)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ERROR parsing JSON: {e}")
        print(f"  First 500 chars of response:\n{response[:500]}")
        raise

    # Validate
    assert len(schema["categories"]) == 9, f"Expected 9 categories, got {len(schema['categories'])}"
    n_dims = len(schema["dimensions"])
    if n_dims < 10:
        print(f"  WARNING: Only {n_dims} dimensions — unusually few")

    print(f"\n  Categories discovered:")
    for cat in schema["categories"]:
        print(f"    - {cat['id']}: {cat['label']}")
    print(f"\n  Dimensions selected ({n_dims}):")
    for dim in schema["dimensions"]:
        print(f"    - {dim['id']}: {dim['scale']}")
    if "dimension_rationale" in schema:
        dr = schema["dimension_rationale"]
        if dr.get("dropped"):
            print(f"\n  Dropped dimensions:")
            for did, reason in dr["dropped"].items():
                print(f"    - {did}: {reason}")
        if dr.get("added"):
            print(f"\n  Added custom dimensions:")
            for did, reason in dr["added"].items():
                print(f"    - {did}: {reason}")

    return schema


def build_consensus_prompt(ticker, runs):
    """Build prompt to merge multiple Phase 1 runs into consensus."""
    runs_text = ""
    for i, run in enumerate(runs, 1):
        runs_text += f"\n--- RUN {i} ---\n{json.dumps(run, indent=2)}\n"

    return f"""You are merging three independent analyses of {ticker}'s news corpus into \
one consensus framework. Below are three separate category/dimension schemas \
produced by different analysts reviewing the same {ticker} articles.

Your job is to produce ONE consensus schema by applying these rules:

CATEGORIES (exactly 9):
- Keep categories that appear in 2+ runs, even if the snake_case id differs \
slightly (match by THEME, not exact name).
- The required categories "earnings_financial_results", "market_sector_sentiment", \
and "analyst_consensus_signals" must always be included.
- If more than 9 survive consensus, keep the 9 most frequently appearing.
- If fewer than 9 survive, include the best unique categories from any run.
- Use the clearest label and description from whichever run expressed it best.

DIMENSIONS (no upper limit):
- Keep dimensions from the suggested pool that appear in 2+ runs.
- Keep CUSTOM dimensions if the same underlying concept appears in 2+ runs \
(even with different names).
- Drop custom dimensions that appeared in only 1 run — they are not robust.
- Keep all that survived consensus — do not artificially limit the count.

SCALES:
- Use the clearest, most general endpoint descriptions (1= and 10= only).
- No hardcoded references to specific events, companies, or scores.

RATIONALE:
- Merge the strongest justifications from all runs.
- For dropped dimensions, note if they were dropped by all 3 or just 2.

{runs_text}

RESPOND WITH ONLY THE CONSENSUS JSON (same format as each run, no markdown fences):
{{
  "ticker": "{ticker}",
  "categories": [...],
  "dimensions": [...],
  "dimension_rationale": {{
    "kept": {{...}},
    "dropped": {{...}},
    "added": {{...}}
  }}
}}"""


def run_phase1_consensus(ticker, articles):
    """Run Phase 1 multiple times and merge into consensus."""
    os.makedirs(PHASE1_RAW_DIR, exist_ok=True)

    # Check for existing raw runs
    existing_runs = []
    for i in range(1, PHASE1_RUNS + 1):
        raw_file = os.path.join(PHASE1_RAW_DIR, f"{ticker}_run{i}.json")
        if os.path.exists(raw_file):
            with open(raw_file, encoding="utf-8") as f:
                existing_runs.append(json.load(f))
            print(f"  Run {i}: loaded from cache")
        else:
            existing_runs.append(None)

    # Run missing runs
    for i in range(PHASE1_RUNS):
        if existing_runs[i] is not None:
            continue
        print(f"\n  --- Run {i+1}/{PHASE1_RUNS} ---")
        schema = run_phase1(ticker, articles)
        existing_runs[i] = schema
        # Save raw run
        raw_file = os.path.join(PHASE1_RAW_DIR, f"{ticker}_run{i+1}.json")
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2)
        time.sleep(SLEEP_BETWEEN)

    # Build consensus via Claude (retry on JSON parse failure)
    print(f"\n  --- Consensus merge for {ticker} ---")
    prompt = build_consensus_prompt(ticker, existing_runs)
    consensus = None
    for attempt in range(MAX_RETRIES):
        response = call_claude(prompt, timeout=300)
        print(f"  Consensus response length: {len(response)} chars")
        try:
            consensus = parse_json_response(response)
            break
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ERROR parsing consensus JSON (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            print(f"  First 500 chars:\n{response[:500]}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying consensus merge...")
                time.sleep(RETRY_BACKOFF * (2 ** attempt))
            else:
                raise

    # Validate consensus
    n_cats = len(consensus.get("categories", []))
    n_dims = len(consensus.get("dimensions", []))
    print(f"\n  Consensus result:")
    print(f"    Categories: {n_cats}")
    for cat in consensus.get("categories", []):
        print(f"      - {cat['id']}: {cat['label']}")
    print(f"    Dimensions: {n_dims}")
    for dim in consensus.get("dimensions", []):
        print(f"      - {dim['id']}")

    # Show what was consistent across runs
    all_dim_ids = [set(d["id"] for d in run["dimensions"]) for run in existing_runs]
    for dim in consensus.get("dimensions", []):
        count = sum(1 for s in all_dim_ids if dim["id"] in s)
        marker = "ALL" if count == PHASE1_RUNS else f"{count}/{PHASE1_RUNS}"
        print(f"        [{marker}]")

    # Save consensus
    consensus_file = os.path.join(PHASE1_RAW_DIR, f"{ticker}_consensus.json")
    with open(consensus_file, "w", encoding="utf-8") as f:
        json.dump(consensus, f, indent=2)

    return consensus


# --- Phase 2: Day-Level Scoring ---

def build_phase2_prompt(ticker, schema, entries):
    """Build Phase 2 scoring prompt for a batch of (date, period, articles) entries.

    entries: list of (date_str, period, articles_list)
    """
    cats_text = "\n".join(
        f"  - {c['id']}: {c['label']} — {c['description']}"
        for c in schema["categories"]
    )
    dims_text = "\n".join(
        f"  - {d['id']}: {d['scale']}"
        for d in schema["dimensions"]
    )
    dim_ids = [d["id"] for d in schema["dimensions"]]

    entries_text = []
    for date_str, period, articles in entries:
        lines = [f"--- ENTRY: {date_str} | {ticker} | {period} ---"]
        for i, art in enumerate(articles, 1):
            t = art["datetime"][11:16]  # HH:MM
            hl = art["headline"]
            sm = art["summary"]
            lines.append(f"  [{i}] {t} | {hl} | {sm}")
        entries_text.append("\n".join(lines))

    all_entries = "\n\n".join(entries_text)

    cat_ids = [c["id"] for c in schema["categories"]]
    dim_json_example = ", ".join(f'"{d}": 5' for d in dim_ids)
    cat_json_example = ", ".join(f'"cat_{c}": 0' for c in cat_ids)

    return f"""You are a senior equity research analyst for {ticker}. For each entry below, \
you are given a set of news articles from a specific date and time window.

Read all articles for each entry. Many articles may cover the SAME event — \
identify duplicates and treat them as one. Then produce a SINGLE combined \
score reflecting the most important news in that set.

Rate each dimension based on the NATURE and CONTENT of the news as it relates \
to {ticker}'s business fundamentals and competitive position.

REMINDER on "surprise": Rate how unexpected the OUTCOME/CONTENT was, not \
whether the event was scheduled. Scheduled earnings with a 40% beat = high \
surprise. Scheduled earnings meeting consensus = low surprise.

CATEGORY RELEVANCE SCORING:
For each entry, score how relevant each category is to that day's news \
(0-10, where 0 = not relevant at all, 10 = entirely about this category). \
A day can score high on MULTIPLE categories if multiple distinct events occurred.

CATEGORIES FOR {ticker}:
{cats_text}

DIMENSIONS (each 1-10):
{dims_text}

ALSO PROVIDE:
- direction: "positive", "negative", or "mixed"
- distinct_events: how many genuinely different news events are in this set
- reasoning: ONE sentence explaining your most important scoring decisions

{all_entries}

RESPOND WITH ONLY THIS JSON ARRAY (no markdown fences, no commentary):
[
  {{
    "date": "YYYY-MM-DD",
    "period": "gap or cc",
    "direction": "positive/negative/mixed",
    "distinct_events": 1,
    {cat_json_example},
    {dim_json_example},
    "reasoning": "One sentence"
  }},
  ...
]"""


def validate_phase2_response(results, schema, entries):
    """Validate Phase 2 response. Returns list of valid results."""
    cat_ids = [c["id"] for c in schema["categories"]]
    dim_ids = {d["id"] for d in schema["dimensions"]}
    valid_directions = {"positive", "negative", "mixed"}
    validated = []

    for r in results:
        # Check required fields
        if "date" not in r or "period" not in r:
            print(f"    WARNING: Missing date/period in response entry, skipping")
            continue
        if r.get("direction") not in valid_directions:
            r["direction"] = "mixed"  # default
        # Clamp category relevance scores to 0-10
        for cat_id in cat_ids:
            key = f"cat_{cat_id}"
            val = r.get(key)
            if val is None or not isinstance(val, (int, float)):
                r[key] = 0  # default to not relevant
            else:
                r[key] = max(0, min(10, int(val)))
        # Clamp dimension scores to 1-10
        for dim_id in dim_ids:
            val = r.get(dim_id)
            if val is None or not isinstance(val, (int, float)):
                r[dim_id] = 5  # default to middle
            else:
                r[dim_id] = max(1, min(10, int(val)))
        validated.append(r)

    return validated


def run_phase2(ticker, schema, gap_entries, cc_entries, already_scored, schema_cache):
    """Run Phase 2 for a single ticker. Writes results incrementally after each batch."""
    # Build list of all (date, period, articles) to score
    all_entries = []
    for (t, date), articles in sorted(gap_entries.items()):
        if t != ticker:
            continue
        if (ticker, date, "gap") not in already_scored:
            all_entries.append((date, "gap", articles))

    for (t, date), articles in sorted(cc_entries.items()):
        if t != ticker:
            continue
        if (ticker, date, "cc") not in already_scored:
            all_entries.append((date, "cc", articles))

    if not all_entries:
        print(f"  All entries for {ticker} already scored, skipping.")
        return 0

    print(f"\n{'='*60}")
    print(f"Phase 2: Scoring {len(all_entries)} (date, period) entries for {ticker}")
    print(f"  Batches of {BATCH_SIZE}, ~{len(all_entries) // BATCH_SIZE + 1} calls")
    print(f"{'='*60}")

    total_scored = 0
    for batch_idx in range(0, len(all_entries), BATCH_SIZE):
        batch = all_entries[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(all_entries) + BATCH_SIZE - 1) // BATCH_SIZE
        n_articles = sum(len(arts) for _, _, arts in batch)

        print(f"\n  Batch {batch_num}/{total_batches}: "
              f"{len(batch)} entries, {n_articles} articles total")

        try:
            prompt = build_phase2_prompt(ticker, schema, batch)
            response = call_claude(prompt, timeout=300)
            results = parse_json_response(response)

            if not isinstance(results, list):
                print(f"    WARNING: Expected list, got {type(results)}. Wrapping.")
                results = [results]

            validated = validate_phase2_response(results, schema, batch)

            # Tag with ticker and write immediately
            for r in validated:
                r["_ticker"] = ticker
            write_output(validated, schema_cache)

            # Update already_scored so we don't re-score on restart
            for r in validated:
                already_scored.add((ticker, r["date"], r["period"]))

            total_scored += len(validated)
            print(f"    Scored {len(validated)}/{len(batch)} entries (total: {total_scored})")

        except Exception as e:
            print(f"    ERROR on batch {batch_num}: {e}")
            print(f"    Skipping batch, will retry on next run.")

        time.sleep(SLEEP_BETWEEN)

    return total_scored


# --- Output writing ---

_write_lock = threading.Lock()


def write_output(all_results, schema_cache):
    """Write/append scored results to CSV. Thread-safe via _write_lock."""
    if not all_results:
        return

    # Collect all category and dimension columns across all tickers
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

    with _write_lock:
        file_exists = os.path.exists(OUTPUT_FILE)

        with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            for r in all_results:
                row = {
                    "date": r.get("date", ""),
                    "ticker": r.get("_ticker", ""),
                    "period": r.get("period", ""),
                    "direction": r.get("direction", ""),
                    "distinct_events": r.get("distinct_events", 1),
                    "reasoning": r.get("reasoning", ""),
                }
                for col in all_cat_cols:
                    row[col] = r.get(col, 0)
                for dim_id in all_dim_ids:
                    row[dim_id] = r.get(dim_id, "")
                writer.writerow(row)

    print(f"\n  Wrote {len(all_results)} rows to {OUTPUT_FILE}")


# --- Main ---

def get_target_tickers():
    if TARGET_TICKERS.upper() == "ALL":
        return ALL_TICKERS
    return [t.strip().upper() for t in TARGET_TICKERS.split(",")]


def main():
    print("=" * 60)
    print("NEWS SCORER PIPELINE")
    print("=" * 60)

    # 1. Load news
    print("\n1. Loading news.csv...")
    news_by_ticker = load_news()
    for t in sorted(news_by_ticker):
        print(f"   {t}: {len(news_by_ticker[t])} articles")

    # 2. Filter to target tickers
    tickers = get_target_tickers()
    print(f"\n2. Target tickers: {tickers}")
    for t in tickers:
        if t not in news_by_ticker:
            print(f"   WARNING: {t} not found in news.csv")

    # 3. Load trading dates and bucket articles
    print("\n3. Loading trading dates and bucketing articles...")
    trading_dates = load_trading_dates()
    filtered_news = {t: news_by_ticker[t] for t in tickers if t in news_by_ticker}
    gap_buckets, cc_buckets = bucket_gap_cc(filtered_news, trading_dates)

    gap_count = sum(len(v) for k, v in gap_buckets.items() if k[0] in tickers)
    cc_count = sum(len(v) for k, v in cc_buckets.items() if k[0] in tickers)
    gap_days = sum(1 for k in gap_buckets if k[0] in tickers)
    cc_days = sum(1 for k in cc_buckets if k[0] in tickers)
    print(f"   Gap: {gap_count} articles across {gap_days} (ticker,date) pairs")
    print(f"   CC:  {cc_count} articles across {cc_days} (ticker,date) pairs")

    # 4. Load caches
    print("\n4. Loading caches...")
    schema_cache = load_categories_cache()
    already_scored = load_existing_output()
    print(f"   Cached schemas: {list(schema_cache.keys())}")
    print(f"   Already scored: {len(already_scored)} entries")

    # 5. Phase 1: Category & dimension discovery (with consensus)
    print(f"\n5. Phase 1: Category & Dimension Discovery ({PHASE1_RUNS}x consensus)")
    for ticker in tickers:
        if ticker in schema_cache:
            print(f"   {ticker}: already cached in categories file, skipping")
            continue
        if ticker not in news_by_ticker:
            continue

        consensus = run_phase1_consensus(ticker, news_by_ticker[ticker])
        schema_cache[ticker] = consensus
        save_categories_cache(schema_cache)
        time.sleep(SLEEP_BETWEEN)

    if PHASE1_ONLY:
        print(f"\n{'='*60}")
        print(f"PHASE 1 COMPLETE (PHASE1_ONLY=True, skipping Phase 2)")
        print(f"  Categories file: {CATEGORIES_FILE}")
        print(f"  Review categories, then set PHASE1_ONLY=False to run Phase 2")
        print(f"{'='*60}")
        return

    # 6. Phase 2: Day-level scoring (writes incrementally after each batch)
    print("\n6. Phase 2: Day-Level Scoring")
    new_scored = 0
    for ticker in tickers:
        if ticker not in schema_cache:
            print(f"   {ticker}: no schema found, skipping Phase 2")
            continue

        count = run_phase2(ticker, schema_cache[ticker],
                           gap_buckets, cc_buckets, already_scored, schema_cache)
        new_scored += count

    # 7. Stats
    total_scored = len(already_scored)
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  New entries scored this run: {new_scored}")
    print(f"  Total entries in output:     {total_scored}")
    print(f"  Output file: {OUTPUT_FILE}")
    print(f"  Categories file: {CATEGORIES_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Add src/ to path for common.py imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()

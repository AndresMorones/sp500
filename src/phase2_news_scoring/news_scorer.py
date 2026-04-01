"""News scoring pipeline using Claude CLI.

Phase 1: Discover 8 company-specific custom categories per stock (consensus of 3 runs).
         20 universal categories are always injected by code — LLM cannot affect them.
Phase 2: Score each (ticker, date, period) across all 28-30 categories (0-10 each).

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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
CATEGORIES_FILE = os.path.join(OUT_DIR, "news_categories.json")
PHASE1_RAW_DIR = os.path.join(OUT_DIR, "news_phase1_raw")
PROGRESS_FILE = os.path.join(OUT_DIR, "news_scorer_progress.json")
OUTPUT_FILE = os.path.join(OUT_DIR, "news_day_features.csv")

ALL_TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

MARKET_OPEN_HOUR, MARKET_OPEN_MIN = 9, 30
MARKET_CLOSE_HOUR = 16

# ---------------------------------------------------------------------------
# 20 hardcoded categories (always present, code-injected — LLM cannot remove)
# ---------------------------------------------------------------------------
REQUIRED_CATEGORIES = [
    {
        "id": "materiality",
        "label": "Materiality",
        "description": (
            "Earnings Beat/Miss, Product Recall, Major Contract Win/Loss, "
            "Bankruptcy Filing, Regulatory Penalty, M&A Completion. "
            "0 = noise with no impact, 10 = direct large impact on revenue/margins/TAM."
        ),
    },
    {
        "id": "surprise_unexpectedness",
        "label": "Surprise & Unexpectedness",
        "description": (
            "Sudden CEO Departure, Unannounced Acquisition, Regulatory Shock, "
            "Unexpected Dividend Cut, Flash Pre-announcement. "
            "0 = fully expected outcome, 10 = completely unexpected outcome."
        ),
    },
    {
        "id": "sentiment_strength",
        "label": "Sentiment Strength",
        "description": (
            "Earnings Blowout, Catastrophic Loss, Transformative Partnership, "
            "Fraud Revelation, Analyst Consensus Reversal. "
            "0 = neutral tone, 10 = extremely strong positive or negative signal."
        ),
    },
    {
        "id": "directional_clarity",
        "label": "Directional Clarity",
        "description": (
            "Clear Guidance Raise/Cut, FDA Approval/Rejection, Definitive Merger "
            "Agreement, Regulatory Ruling, Contract Win/Loss. "
            "0 = ambiguous/could go either way, 10 = unambiguously good or bad."
        ),
    },
    {
        "id": "information_density",
        "label": "Information Density",
        "description": (
            "Full Earnings Release, Detailed Guidance, SEC Filing with Data, "
            "Analyst Research Report, Verified Technical Data. "
            "0 = pure opinion/speculation, 10 = hard data with specific numbers."
        ),
    },
    {
        "id": "impact_timeline_duration",
        "label": "Impact Timeline & Duration",
        "description": (
            "Score how long the price impact is likely to persist: "
            "0-3 = fades within days (sentiment-only, speculative, one-time); "
            "4-6 = resolves within weeks to months; "
            "7-10 = permanently alters business trajectory (structural, confirmed, recurring)."
        ),
    },
    {
        "id": "narrative_shift",
        "label": "Narrative Shift",
        "description": (
            "CEO Scandal reversing trust, Earnings Miss breaking growth story, "
            "Turnaround Success, Business Model Pivot. "
            "0 = reinforces existing narrative, 10 = fundamentally changes investment thesis."
        ),
    },
    {
        "id": "scope_breadth",
        "label": "Scope & Breadth",
        "description": (
            "Full Company Guidance, CEO Departure, Capital Structure Change, "
            "Single Product Issue, Regional Sales Miss. "
            "0 = only this company/product, 5 = sector-wide, 10 = macro/market-wide."
        ),
    },
    {
        "id": "competitive_impact",
        "label": "Competitive Impact",
        "description": (
            "Competitor Bankruptcy, Patent Win/Loss, Market Share Data Release, "
            "Competitor Product Launch, Pricing War. "
            "0 = no competitive effect, 10 = major competitive shift."
        ),
    },
    {
        "id": "regulatory_risk",
        "label": "Regulatory Risk",
        "description": (
            "Antitrust Investigation, FDA Rejection, SEC Investigation, Tax Ruling, "
            "License Revocation, Compliance Violation. "
            "0 = no regulatory angle, 10 = major legal/regulatory event."
        ),
    },
    {
        "id": "financial_result_surprise",
        "label": "Financial Result Surprise",
        "description": (
            "EPS Beat/Miss vs Consensus, Revenue vs Guidance, Margin Surprise, "
            "Cash Flow Miss, Financial Restatement. "
            "0 = no financial results or results exactly match expectations, "
            "10 = figures dramatically exceed or miss prior guidance and analyst expectations."
        ),
    },
    {
        "id": "earnings_financial_results",
        "label": "Earnings & Financial Results",
        "description": (
            "Earnings Release, Revenue Guidance, Profit Warning, Annual Report, "
            "Quarterly Filing, Financial Restatement. "
            "Score how central this news is to the company's reported financial performance."
        ),
    },
    {
        "id": "market_sector_sentiment",
        "label": "Market & Sector Sentiment",
        "description": (
            "Sector Rotation, Industry-wide Regulation, Commodity Price Move, "
            "Macro Data Release, Fed Decision, Peer Earnings. "
            "Score how much this news reflects broad market or sector-level dynamics."
        ),
    },
    {
        "id": "analyst_consensus_signals",
        "label": "Analyst Consensus Signals",
        "description": (
            "Rating Upgrade/Downgrade, Price Target Revision, Analyst Initiation, "
            "Consensus Estimate Change, Analyst Day. "
            "Score how much this news reflects analyst community views on the stock."
        ),
    },
    {
        "id": "catalyst_immediacy",
        "label": "Catalyst Immediacy",
        "description": (
            "Trading Halt, Breaking Announcement, Flash Pre-announcement, "
            "Scheduled Earnings, Long-term Roadmap Reveal. "
            "0 = background/long-term context only, 10 = immediate market-moving catalyst."
        ),
    },
    {
        "id": "systemic_contagion_risk",
        "label": "Systemic & Contagion Risk",
        "description": (
            "Banking Sector Stress, Supply Chain Collapse, Industry-wide Regulatory Sweep, "
            "Peer Earnings Warning, Macro Shock. "
            "Score how much this news signals risk spreading beyond the single company."
        ),
    },
    {
        "id": "management_credibility",
        "label": "Management & Credibility",
        "description": (
            "CEO/CFO Appointment or Resignation, Insider Trading, Executive Misconduct, "
            "Board Restructuring, Governance Change. "
            "Score how much this news affects trust in and stability of leadership."
        ),
    },
    {
        "id": "capital_allocation_signal",
        "label": "Capital Allocation Signal",
        "description": (
            "Share Buyback Program, Dividend Initiation/Cut, Major CapEx Announcement, "
            "Debt Issuance, Equity Offering. "
            "Score how much this news reveals management's capital deployment priorities."
        ),
    },
    {
        "id": "valuation_growth_outlook",
        "label": "Valuation & Growth Outlook",
        "description": (
            "Long-term Growth Rate Revision, Margin Expansion/Compression, TAM Update, "
            "Business Model Change, Multiple Re-rating. "
            "Score how much this news changes the long-term growth or valuation narrative."
        ),
    },
    {
        "id": "sentiment_direction",
        "label": "Sentiment Direction",
        "description": (
            "Score the net directional tone of the news: "
            "0 = entirely negative/bearish, "
            "5 = neutral or balanced with no clear lean, "
            "10 = entirely positive/bullish. "
            "Score the article's net tone — a mixed article with more negative weight scores 3, not 5."
        ),
    },
]

REQUIRED_IDS = {c["id"] for c in REQUIRED_CATEGORIES}


def inject_required_categories(schema):
    """Prepend the 20 hardcoded categories to schema['categories'], deduplicating any LLM collision."""
    custom = [c for c in schema.get("categories", []) if c["id"] not in REQUIRED_IDS]
    schema = dict(schema)  # shallow copy to avoid mutating caller's dict
    schema["categories"] = REQUIRED_CATEGORIES + custom
    return schema


# --- Claude CLI wrapper ---

def call_claude(prompt, timeout=300):
    """Call Claude via CLI. Returns the text response."""
    for attempt in range(MAX_RETRIES):
        try:
            # Pass prompt via stdin ("-") to avoid OS arg length limits
            # shell=True needed on Windows so subprocess finds .cmd files in PATH
            claude_bin = os.environ.get("CLAUDE_BIN", "claude")
            cmd = f'{claude_bin} -p - --output-format json --model {MODEL}'
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


# --- Phase 1: Custom Category Discovery ---

def build_phase1_prompt(ticker, articles):
    """Build the Phase 1 discovery prompt for a ticker."""
    required_table = "\n".join(
        f"  - {c['id']}: {c['description']}"
        for c in REQUIRED_CATEGORIES
    )
    required_ids_list = ", ".join(c["id"] for c in REQUIRED_CATEGORIES)

    article_lines = []
    for i, art in enumerate(articles, 1):
        hl = art["headline"]
        sm = art["summary"]
        article_lines.append(f"[{i}] {art['datetime'][:10]} | {hl} | {sm}")

    articles_text = "\n".join(article_lines)

    return f"""You are a senior equity research analyst. You will receive ALL {len(articles)} news \
articles about {ticker} from November 2023 to October 2024. Your task is to \
identify 8 company-specific news categories that are UNIQUE to {ticker}'s business model.

IMPORTANT RULES:
- You have ZERO access to stock prices, returns, or trading data.
- Base your analysis ONLY on the content and nature of the news events.
- Do NOT speculate about what the stock price did in response.
- Think about what types of events are specific to {ticker}'s business model, \
competitive position, and investor base.

HARDCODED CATEGORIES — DO NOT CREATE CATEGORIES WITH THESE IDs
The following 20 categories are ALWAYS included separately and will be scored in Phase 2. \
Your job is to find what is UNIQUE to {ticker} that these 20 universal signals do NOT cover:

{required_table}

TASK — DISCOVER 8 COMPANY-SPECIFIC CATEGORIES

Identify exactly 8 categories from the articles that capture {ticker}-specific \
news themes NOT already covered by the 20 hardcoded categories above.

Rules:
- Every category must be specific to {ticker}'s business — not a rewording of \
any of the 20 hardcoded ones above.
- Do NOT create categories with these IDs (already hardcoded): {required_ids_list}
- Do NOT create hyper-specific categories covering only 2–3 articles. \
Merge related themes into broader buckets. But do NOT merge genuinely distinct \
themes just because one has fewer articles. A category covering 10 important \
articles is better than losing that signal by folding it into a broader bucket.

  GOOD: "gaming_entertainment_strategy" (covers console strategy, acquisitions, \
game releases, subscription models)
  BAD: "activision_integration" (too narrow, only ~5 articles about one event)

Each category needs: a snake_case id, a short label, and a one-sentence description. \
Every article in the corpus should fit into at least one of the 20 hardcoded + 8 custom categories.

RESPOND WITH ONLY THIS JSON (no markdown fences, no commentary):
{{
  "ticker": "{ticker}",
  "categories": [
    {{"id": "snake_case_id", "label": "Short Label", "description": "One sentence"}},
    ... (exactly 8, none matching any of the 20 hardcoded IDs above)
  ]
}}

ARTICLES FOR {ticker}:

{articles_text}"""


def run_phase1(ticker, articles):
    """Run Phase 1 for a single ticker. Returns the parsed schema dict with all 28 categories."""
    print(f"\n{'='*60}")
    print(f"Phase 1: Discovering custom categories for {ticker}")
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

    # Inject the 20 hardcoded categories
    schema = inject_required_categories(schema)

    # Validate custom count
    custom_count = sum(1 for c in schema["categories"] if c["id"] not in REQUIRED_IDS)
    total_count = len(schema["categories"])
    if custom_count < 6 or custom_count > 15:
        print(f"  WARNING: Unusual custom category count: {custom_count} "
              f"(expected 6–15). Total with hardcoded: {total_count}")
    else:
        print(f"  Categories: {custom_count} custom + 20 hardcoded = {total_count} total")

    print(f"\n  Custom categories discovered:")
    for cat in schema["categories"]:
        if cat["id"] not in REQUIRED_IDS:
            print(f"    - {cat['id']}: {cat['label']}")

    return schema


def build_consensus_prompt(ticker, runs):
    """Build prompt to merge multiple Phase 1 runs into consensus (custom categories only)."""
    required_ids_list = ", ".join(c["id"] for c in REQUIRED_CATEGORIES)

    # Filter each run to only its custom categories (exclude hardcoded ones)
    custom_runs = []
    for run in runs:
        custom_cats = [c for c in run.get("categories", []) if c["id"] not in REQUIRED_IDS]
        custom_runs.append({"ticker": run.get("ticker", ticker), "categories": custom_cats})

    runs_text = ""
    for i, run in enumerate(custom_runs, 1):
        runs_text += f"\n--- RUN {i} ---\n{json.dumps(run, indent=2)}\n"

    return f"""You are merging three independent analyses of {ticker}'s news corpus. \
Each run identified company-specific news categories unique to {ticker}. \
Your job is to produce ONE consensus list of custom categories.

CATEGORIES — merge all unique custom themes:
- Keep categories that appear in 2+ runs, even if the snake_case id differs \
slightly (match by THEME, not exact name).
- Drop a category only if it appeared in exactly 1 run and covers a theme \
already well-addressed by another surviving category.
- If in doubt whether two categories are the same theme, keep both.
- Target is 8 custom categories, but keep more if they appear consistently \
across runs — there is NO upper limit.
- Use the clearest label and description from whichever run expressed it best.
- Do NOT include categories with these IDs — they are hardcoded separately: \
{required_ids_list}

{runs_text}

RESPOND WITH ONLY THE CONSENSUS JSON (no markdown fences, no commentary):
{{
  "ticker": "{ticker}",
  "categories": [
    {{"id": "snake_case_id", "label": "Short Label", "description": "One sentence"}},
    ...
  ]
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

    # Inject the 20 hardcoded categories into the consensus result
    consensus = inject_required_categories(consensus)

    # Validate & report
    custom_count = sum(1 for c in consensus.get("categories", []) if c["id"] not in REQUIRED_IDS)
    total_count = len(consensus.get("categories", []))
    print(f"\n  Consensus result: {custom_count} custom + 20 hardcoded = {total_count} total")
    print(f"  Custom categories:")
    for cat in consensus.get("categories", []):
        if cat["id"] not in REQUIRED_IDS:
            print(f"    - {cat['id']}: {cat['label']}")

    # Show cross-run consistency for custom categories
    all_custom_ids = [
        set(c["id"] for c in run.get("categories", []) if c["id"] not in REQUIRED_IDS)
        for run in existing_runs
    ]
    for cat in consensus.get("categories", []):
        if cat["id"] not in REQUIRED_IDS:
            count = sum(1 for s in all_custom_ids if cat["id"] in s)
            marker = "ALL" if count == PHASE1_RUNS else f"{count}/{PHASE1_RUNS}"
            print(f"        [{marker}] {cat['id']}")

    # Save consensus (full schema with 20 hardcoded + custom)
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
    cat_json_example = ", ".join(f'"cat_{c}": 0' for c in cat_ids)

    return f"""You are a senior equity research analyst for {ticker}. For each entry below, \
you are given a set of news articles from a specific date and time window.

Read all articles for each entry. Many articles may cover the SAME event — \
identify duplicates and treat them as one. Then produce a SINGLE combined \
score reflecting the most important news in that set.

SCORING RULES:
- ALL scores are 0–10 integers. Use the FULL range — do not cluster around the middle.
- Score based ONLY on the NATURE and CONTENT of the articles provided.
- Do not speculate about stock price reactions or trading outcomes.
- Infer everything from what the articles say — do not inject outside knowledge.

CATEGORY SCORES (each 0–10):
Score how much each category applies to this day's news. \
0 = not relevant at all, 10 = the news is entirely about this category. \
A day can score high on MULTIPLE categories if multiple distinct events occurred.

CATEGORIES FOR {ticker}:
{cats_text}

ALSO PROVIDE:
- distinct_events: integer count of genuinely different news events in this entry
- reasoning: ONE sentence explaining your most important scoring decisions

{all_entries}

RESPOND WITH ONLY THIS JSON ARRAY (no markdown fences, no commentary):
[
  {{
    "date": "YYYY-MM-DD",
    "period": "gap or cc",
    "distinct_events": 1,
    {cat_json_example},
    "reasoning": "One sentence"
  }},
  ...
]"""


def validate_phase2_response(results, schema):
    """Validate Phase 2 response. Returns list of valid results."""
    cat_ids = [c["id"] for c in schema["categories"]]
    validated = []

    for r in results:
        # Check required fields
        if "date" not in r or "period" not in r:
            print(f"    WARNING: Missing date/period in response entry, skipping")
            continue
        # Clamp all category scores to 0-10
        for cat_id in cat_ids:
            key = f"cat_{cat_id}"
            val = r.get(key)
            if val is None or not isinstance(val, (int, float)):
                r[key] = 0  # default to not relevant
            else:
                r[key] = max(0, min(10, int(val)))
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

            validated = validate_phase2_response(results, schema)

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

    # Collect all category columns across all tickers (no separate dim columns)
    all_cat_cols = set()
    for schema in schema_cache.values():
        for c in schema["categories"]:
            all_cat_cols.add(f"cat_{c['id']}")
    all_cat_cols = sorted(all_cat_cols)

    fieldnames = ["date", "ticker", "period", "distinct_events"] + all_cat_cols + ["reasoning"]

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
                    "distinct_events": r.get("distinct_events", 1),
                    "reasoning": r.get("reasoning", ""),
                }
                for col in all_cat_cols:
                    row[col] = r.get(col, 0)
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

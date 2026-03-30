"""
Parallel sentiment analysis — runs tickers in parallel (multiprocessing)
and 3 models in parallel within each ticker (threading).

With 16GB RAM / 12GB free:
  - 3 concurrent ticker processes × ~1.8GB each = ~5.4GB
  - Within each process: 3 threads for FinBERT/RoBERTa/DeBERTa simultaneously

Expected ~3x speedup over sequential.

Usage:
    cd LLM-Sentiment-Stock-Prediction-DeBERTa-TimesNet
    PYTHONPATH=. python sentiment_analysis/run_parallel.py
    PYTHONPATH=. python sentiment_analysis/run_parallel.py --workers 5  # all 5 tickers at once
"""
import sys
import os
import re
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from stock_prediction.config import NEWS_CSV, INTERIM_DATA_DIR, TICKERS
from loguru import logger


def load_models_and_run_single(model_name, texts):
    """Load one model and run inference on all texts. Thread-safe."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from models.bert_model import BertModel

    model_map = {
        'finbert': 'ProsusAI/finbert',
        'roberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
        'deberta': 'nickmuchi/deberta-v3-base-finetuned-finance-text-classification',
    }

    hf_name = model_map[model_name]
    model = AutoModelForSequenceClassification.from_pretrained(hf_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    bert = BertModel(model, tokenizer)

    predictions = [bert.generate(text) for text in texts]
    return model_name, predictions


def map_sentiment(model_name, pred):
    label = pred['label']
    if model_name in ('finbert', 'roberta'):
        if label.lower() == 'positive': return 2
        elif label.lower() == 'negative': return 0
        else: return 1
    else:
        if label.upper() in ('POSITIVE', 'POS', 'BULLISH'): return 2
        elif label.upper() in ('NEGATIVE', 'NEG', 'BEARISH'): return 0
        else: return 1


def process_ticker(ticker_and_data):
    """Process a single ticker — loads models, runs all 3 in parallel threads."""
    ticker, ticker_news_csv = ticker_and_data

    # Re-import inside subprocess
    import pandas as pd
    from scipy.stats import mode as scipy_mode

    df = pd.read_csv(ticker_news_csv) if isinstance(ticker_news_csv, str) else ticker_news_csv.copy()

    # Prepare text
    df['text'] = df['headline'].fillna('') + ". " + df['summary'].fillna('')
    df['text'] = df['text'].str.strip()
    df['date'] = pd.to_datetime(df['datetime']).dt.date

    # Deduplicate
    def canon(t):
        t = str(t).lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^\w\s]", "", t)
        return t.strip()
    df['_canon'] = df['text'].map(canon)
    df = df.drop_duplicates(subset=['_canon'], keep='first').drop(columns=['_canon'])

    texts = df['text'].tolist()
    n = len(texts)
    t0 = time.time()
    logger.info(f"  {ticker}: starting {n} articles through 3 models in parallel...")

    # Run all 3 models in parallel threads
    model_results = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(load_models_and_run_single, name, texts): name
            for name in ['finbert', 'roberta', 'deberta']
        }
        for future in futures:
            model_name, predictions = future.result()
            model_results[model_name] = predictions
            elapsed = time.time() - t0
            logger.info(f"    {ticker}/{model_name}: done ({elapsed:.0f}s)")

    # Apply results to dataframe
    for model_name, predictions in model_results.items():
        df[f'{model_name}_sentiment'] = [map_sentiment(model_name, p) for p in predictions]
        df[f'{model_name}_confidence'] = [max(p['probabilities'].values()) for p in predictions]
        for key in predictions[0]['probabilities'].keys():
            df[f'{model_name}_label_{key}'] = [p['probabilities'][key] for p in predictions]

    # Majority vote
    sentiment_cols = df[['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']].values
    df['transformer_majority_vote'] = scipy_mode(sentiment_cols, axis=1, keepdims=False).mode

    # Save
    out_path = str(Path(sys.path[0]).parent / "llm_baseline_model_modified" / "LLM-Sentiment-Stock-Prediction-DeBERTa-TimesNet" / "data" / "interim" / f"{ticker}_sentiment_results.csv")
    # Use INTERIM_DATA_DIR from config
    from stock_prediction.config import INTERIM_DATA_DIR as idir
    idir.mkdir(parents=True, exist_ok=True)
    out_path = idir / f"{ticker}_sentiment_results.csv"
    df.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    logger.success(f"  {ticker}: DONE — {n} articles in {elapsed:.0f}s ({elapsed/n:.2f}s/article) -> {out_path}")
    return ticker, n, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel ticker processes")
    args = parser.parse_args()

    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which tickers need processing
    remaining = []
    for t in TICKERS:
        fpath = INTERIM_DATA_DIR / f"{t}_sentiment_results.csv"
        if fpath.exists() and fpath.stat().st_size > 1000:
            logger.info(f"  {t}: already done ({fpath.stat().st_size:,} bytes), skipping")
        else:
            remaining.append(t)

    if not remaining:
        logger.success("All tickers already processed!")
        return

    logger.info(f"Tickers to process: {remaining} ({len(remaining)} tickers)")
    logger.info(f"Using {args.workers} parallel worker processes")

    # Load news data
    all_news = pd.read_csv(str(NEWS_CSV))
    logger.info(f"Total articles: {len(all_news)}")

    # Prepare per-ticker data
    ticker_data = []
    for t in remaining:
        tn = all_news[all_news['ticker'] == t].copy()
        logger.info(f"  {t}: {len(tn)} articles")
        ticker_data.append((t, tn))

    total_start = time.time()

    # Process tickers in parallel
    n_workers = min(args.workers, len(remaining))
    logger.info(f"\nLaunching {n_workers} parallel processes...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_ticker, ticker_data))

    total = time.time() - total_start
    logger.success(f"\n{'='*60}")
    logger.success(f"ALL DONE in {total/60:.1f} minutes")
    logger.success(f"{'='*60}")
    for ticker, n, elapsed in results:
        logger.info(f"  {ticker}: {n} articles in {elapsed:.0f}s")


if __name__ == "__main__":
    main()

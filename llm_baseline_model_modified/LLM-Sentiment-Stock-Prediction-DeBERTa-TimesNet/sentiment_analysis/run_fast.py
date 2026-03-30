"""
Fast sentiment analysis — loads models once, processes all tickers sequentially,
skips visualizations for speed.

Usage:
    cd LLM-Sentiment-Stock-Prediction-DeBERTa-TimesNet
    PYTHONPATH=. python sentiment_analysis/run_fast.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore")

from stock_prediction.config import NEWS_CSV, INTERIM_DATA_DIR, TICKERS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.bert_model import BertModel
from loguru import logger


def load_all_models():
    """Load all 3 transformer models once."""
    logger.info("Loading FinBERT...")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert = BertModel(finbert_model, finbert_tokenizer)

    logger.info("Loading RoBERTa...")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    roberta = BertModel(roberta_model, roberta_tokenizer)

    logger.info("Loading DeBERTa...")
    deberta_model = AutoModelForSequenceClassification.from_pretrained("nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
    deberta_tokenizer = AutoTokenizer.from_pretrained("nickmuchi/deberta-v3-base-finetuned-finance-text-classification")
    deberta = BertModel(deberta_model, deberta_tokenizer)

    return {'finbert': finbert, 'roberta': roberta, 'deberta': deberta}


def map_sentiment(model_name, pred):
    """Map model prediction to 0/1/2."""
    label = pred['label']
    if model_name in ('finbert', 'roberta'):
        if label.lower() == 'positive': return 2
        elif label.lower() == 'negative': return 0
        else: return 1
    else:  # deberta
        if label.upper() in ('POSITIVE', 'POS', 'BULLISH'): return 2
        elif label.upper() in ('NEGATIVE', 'NEG', 'BEARISH'): return 0
        else: return 1


def analyze_ticker(ticker, df, models):
    """Run sentiment analysis on a single ticker's articles."""
    import re
    # Prepare text
    df = df.copy()
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

    for model_name, model in models.items():
        t0 = time.time()
        logger.info(f"  {ticker}: {model_name} on {n} articles...")
        predictions = [model.generate(text) for text in texts]
        df[f'{model_name}_sentiment'] = [map_sentiment(model_name, p) for p in predictions]
        df[f'{model_name}_confidence'] = [max(p['probabilities'].values()) for p in predictions]
        for key in predictions[0]['probabilities'].keys():
            df[f'{model_name}_label_{key}'] = [p['probabilities'][key] for p in predictions]
        elapsed = time.time() - t0
        logger.info(f"    done in {elapsed:.0f}s ({elapsed/n:.2f}s/article)")

    # Majority vote
    from scipy.stats import mode as scipy_mode
    sentiment_cols = df[['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']].values
    df['transformer_majority_vote'] = scipy_mode(sentiment_cols, axis=1, keepdims=False).mode

    return df


def main():
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check which tickers are already done
    remaining = []
    for t in TICKERS:
        fpath = INTERIM_DATA_DIR / f"{t}_sentiment_results.csv"
        if fpath.exists() and fpath.stat().st_size > 1000:
            logger.info(f"  {t}: already done ({fpath.stat().st_size} bytes), skipping")
        else:
            remaining.append(t)

    if not remaining:
        logger.success("All tickers already processed!")
        return

    logger.info(f"Tickers to process: {remaining}")

    # Load news data
    logger.info(f"Loading news from {NEWS_CSV}")
    all_news = pd.read_csv(str(NEWS_CSV))
    logger.info(f"Total articles: {len(all_news)}")

    # Load models ONCE
    logger.info("Loading transformer models...")
    t0 = time.time()
    models = load_all_models()
    logger.info(f"Models loaded in {time.time()-t0:.0f}s")

    total_start = time.time()

    for ticker in remaining:
        ticker_news = all_news[all_news['ticker'] == ticker].copy()
        logger.info(f"\n{'='*50}")
        logger.info(f"{ticker}: {len(ticker_news)} articles")

        if len(ticker_news) == 0:
            logger.warning(f"  No news for {ticker}, skipping")
            continue

        t0 = time.time()
        results_df = analyze_ticker(ticker, ticker_news, models)

        # Save
        out_path = INTERIM_DATA_DIR / f"{ticker}_sentiment_results.csv"
        results_df.to_csv(out_path, index=False)
        elapsed = time.time() - t0
        logger.success(f"  {ticker} done: {len(results_df)} rows in {elapsed:.0f}s -> {out_path}")

    total = time.time() - total_start
    logger.success(f"\nAll done in {total/60:.1f} minutes")


if __name__ == "__main__":
    main()

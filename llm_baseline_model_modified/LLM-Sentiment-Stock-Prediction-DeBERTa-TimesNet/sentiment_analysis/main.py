"""
Sentiment analysis pipeline adapted for sp500 project.

Runs FinBERT, RoBERTa, and DeBERTa on news articles from the consolidated
news.csv file. Classical models (SVM, LR, RF) are not used because no
human-labeled benchmark dataset (SEntFiN-v1.1) is available.

Usage:
    python sentiment_analysis/main.py
"""
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_prediction.config import (
    PROJ_ROOT, NEWS_CSV, INTERIM_DATA_DIR, TICKERS, REPORTS_DIR, FIGURES_DIR,
)
from comparison.stock import StockSentimentComparison
import warnings

warnings.filterwarnings("ignore")


def main():
    # Ensure output directories exist
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "output").mkdir(parents=True, exist_ok=True)
    (FIGURES_DIR / "sentiment_analysis").mkdir(parents=True, exist_ok=True)

    # Load the consolidated news file once
    logger.info(f"Loading news data from {NEWS_CSV}")
    all_news = pd.read_csv(NEWS_CSV)
    logger.info(f"Total news articles: {len(all_news)}")

    for ticker in TICKERS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {ticker}")
        logger.info(f"{'='*60}")

        # Filter news for this ticker
        ticker_news = all_news[all_news['ticker'] == ticker].copy()
        logger.info(f"  {ticker} articles: {len(ticker_news)}")

        if len(ticker_news) == 0:
            logger.warning(f"  No news found for {ticker}, skipping")
            continue

        # Run transformer-based sentiment analysis
        analyzer = StockSentimentComparison(
            path=None,
            ticker=ticker,
            df=ticker_news,
        )
        results_df, agreement_matrix = analyzer.run_complete_analysis()

        # Save results
        results_file = INTERIM_DATA_DIR / f"{ticker}_sentiment_results.csv"
        agree_file = INTERIM_DATA_DIR / f"{ticker}_model_agreement_matrix.csv"

        results_df.to_csv(results_file, index=False)
        agreement_matrix.to_csv(agree_file)

        logger.success(f"  Results saved: {results_file}")
        logger.info(f"  DataFrame shape: {results_df.shape}")
        logger.info(f"  Columns: {list(results_df.columns)}")

    logger.success("\nAll tickers processed!")


if __name__ == "__main__":
    main()

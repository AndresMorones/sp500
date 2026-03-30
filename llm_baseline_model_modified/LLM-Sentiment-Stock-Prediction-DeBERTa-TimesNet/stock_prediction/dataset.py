"""
Data loading and preprocessing pipeline adapted for sp500 project.

Loads price data from the consolidated price.csv, loads sentiment results
from the interim directory, aggregates sentiment by date, and merges with
price data to produce per-ticker preprocessed datasets.

Usage:
    python stock_prediction/dataset.py                   # Process all tickers
    python stock_prediction/dataset.py --stock_symbol AAPL  # Single ticker
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
import sys

# Add the project root to the path to ensure imports work
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from datetime import datetime, date

from stock_prediction.config import (
    PROCESSED_DATA_DIR, INTERIM_DATA_DIR, PRICE_CSV, TICKERS,
)

app = typer.Typer()


def load_price_data(stock_symbol: str) -> pd.DataFrame:
    """Load price data for a given stock symbol from the consolidated price.csv."""
    if not PRICE_CSV.exists():
        raise FileNotFoundError(f"Price data file not found: {PRICE_CSV}")

    logger.info(f"Loading price data for {stock_symbol} from {PRICE_CSV}")
    df = pd.read_csv(PRICE_CSV)

    # Filter for the requested ticker
    df = df[df['ticker'] == stock_symbol].copy()

    if len(df) == 0:
        raise ValueError(f"No price data found for {stock_symbol}")

    # Rename lowercase columns to Title Case for downstream compatibility
    df.rename(columns={
        'date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    # Drop the ticker column (no longer needed after filtering)
    df.drop(columns=['ticker'], inplace=True, errors='ignore')

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    logger.info(f"  {stock_symbol} price data: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


def load_sentiment_data(stock_symbol: str) -> pd.DataFrame:
    """Load sentiment analysis results for a given stock symbol."""
    file_path = INTERIM_DATA_DIR / f"{stock_symbol}_sentiment_results.csv"

    if not file_path.exists():
        logger.warning(f"Sentiment data not found: {file_path}")
        return pd.DataFrame()

    logger.info(f"Loading sentiment data from {file_path}")
    return pd.read_csv(file_path)


def extract_date_column(news_df: pd.DataFrame) -> pd.Series:
    """Extract and standardize date column from news dataframe."""
    date_columns = ['date', 'datetime', 'Date', 'DateTime']

    for col in date_columns:
        if col in news_df.columns:
            return pd.to_datetime(news_df[col]).dt.date

    # Fallback: try to find any datetime-like column
    for col in news_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                return pd.to_datetime(news_df[col]).dt.date
            except Exception:
                continue

    raise ValueError("Could not find a valid date column in the news dataframe")


def identify_model_columns(news_df: pd.DataFrame) -> Dict[str, str]:
    """Identify sentiment model columns in the dataframe."""
    model_names = ["deberta", "finbert", "roberta"]
    model_cols = {}

    for name in model_names:
        matches = [col for col in news_df.columns if name.lower() in col.lower()]
        if matches:
            model_cols[name] = matches[0]

    return model_cols


def aggregate_sentiment_by_date(news_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment scores by date for transformer models only."""
    news_df = news_df.copy()
    news_df["date_only"] = extract_date_column(news_df)

    # Base: total news count per date
    result_df = (
        news_df
        .groupby("date_only")
        .size()
        .reset_index(name="total_news_count")
    )

    def majority_vote(series):
        return series.mode().iloc[0] if not series.mode().empty else None

    # Build aggregation dict dynamically based on available columns
    aggregation = {}

    # Sentiment columns for transformer models only
    for model in ["finbert", "roberta", "deberta"]:
        sent_col = f"{model}_sentiment"
        if sent_col in news_df.columns:
            aggregation[sent_col] = [
                majority_vote, 'min', 'max',
                lambda x, m=model: (x == 2).sum(),  # positive count
                lambda x, m=model: (x == 0).sum(),  # negative count
                lambda x, m=model: (x == 1).sum(),  # neutral count
            ]

    # Confidence/label probability columns
    for col in news_df.columns:
        if col.endswith('_confidence') or '_label_' in col:
            if any(m in col for m in ["finbert", "roberta", "deberta"]):
                aggregation[col] = 'sum'

    if not aggregation:
        logger.warning("No sentiment columns found for aggregation")
        return result_df

    # Perform aggregation
    grouped = news_df.groupby('date_only').agg(aggregation)

    # Flatten column names
    flat_cols = []
    for col in grouped.columns:
        if isinstance(col, tuple):
            flat_cols.append('_'.join(str(c) for c in col).strip('_'))
        else:
            flat_cols.append(col)
    grouped.columns = flat_cols

    # Rename to clean names
    rename_map = {}
    for model in ["finbert", "roberta", "deberta"]:
        rename_map[f"{model}_sentiment_majority_vote"] = f"{model}_majority_vote"
        rename_map[f"{model}_sentiment_min"] = f"{model}_min"
        rename_map[f"{model}_sentiment_max"] = f"{model}_max"
        rename_map[f"{model}_sentiment_<lambda_0>"] = f"{model}_count_positive"
        rename_map[f"{model}_sentiment_<lambda_1>"] = f"{model}_count_negative"
        rename_map[f"{model}_sentiment_<lambda_2>"] = f"{model}_count_neutral"

    # Handle DeBERTa label columns (model outputs bearish/bullish/neutral)
    # Normalize to consistent positive/negative/neutral naming
    for old_suffix, new_suffix in [
        ("bearish", "negative"), ("bullish", "positive"),
    ]:
        old_key = f"deberta_label_{old_suffix}_sum"
        new_key = f"deberta_label_{new_suffix}_sum"
        if old_key in grouped.columns:
            rename_map[old_key] = new_key

    # Also handle _sum suffix for label probability columns
    for col in grouped.columns:
        if col.endswith("_sum") and col not in rename_map:
            # Already has _sum suffix from aggregation, keep as is
            pass

    grouped.rename(columns=rename_map, inplace=True)
    grouped = grouped.reset_index()

    result_df = result_df.merge(grouped, on="date_only", how="left")

    if len(result_df.columns) == 1:
        raise ValueError("No valid model columns found for aggregation")

    return result_df


def merge_price_and_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge price data with aggregated sentiment data in horizontal format."""
    # Ensure we have a date column in price data
    date_col = None
    for col in ['Date', 'date']:
        if col in price_df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError("No date column found in price data")

    # Convert price date to date only for merging
    price_df = price_df.copy()
    price_df['date_only'] = pd.to_datetime(price_df[date_col]).dt.date

    # Merge on date - one row per date with all model metrics as columns
    merged_df = price_df.merge(sentiment_df, on='date_only', how='left')

    # Fill NaN sentiment values with 0 (no-news days)
    sentiment_cols = [c for c in merged_df.columns if any(
        m in c for m in ['finbert', 'roberta', 'deberta', 'total_news_count']
    )]
    merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)

    # Sort by date
    merged_df = merged_df.sort_values('date_only')

    return merged_df


@app.command()
def main(
    stock_symbol: str = typer.Option("all", help="Stock symbol to process, or 'all' for all tickers"),
    output_path: Optional[Path] = typer.Option(None, help="Custom output path"),
    verbose: bool = typer.Option(False, help="Enable verbose logging")
):
    """
    Process stock price data and sentiment analysis results to create preprocessed datasets.

    Combines price data with sentiment analysis results from transformer models,
    aggregating sentiment scores by date and merging with price information.
    """
    if verbose:
        logger.add(sys.stderr, level="DEBUG")

    tickers = TICKERS if stock_symbol == "all" else [stock_symbol]

    for ticker in tickers:
        logger.info(f"\nProcessing dataset for {ticker}...")

        try:
            # Load data
            price_df = load_price_data(ticker)
            news_df = load_sentiment_data(ticker)

            if len(news_df) > 0:
                # Identify model columns
                model_cols = identify_model_columns(news_df)
                logger.info(f"Found model columns: {list(model_cols.keys())}")

                # Aggregate sentiment by date
                sentiment_agg_df = aggregate_sentiment_by_date(news_df)
                logger.info(f"Aggregated sentiment data shape: {sentiment_agg_df.shape}")
            else:
                logger.warning(f"No sentiment data for {ticker}, creating price-only dataset")
                sentiment_agg_df = pd.DataFrame(columns=['date_only'])

            # Merge with price data
            merged_df = merge_price_and_sentiment(price_df, sentiment_agg_df)
            logger.info(f"Merged dataset shape: {merged_df.shape}")

            # Determine output path
            if output_path and len(tickers) == 1:
                final_output_path = output_path
            else:
                final_output_path = PROCESSED_DATA_DIR / f"{ticker}_preprocessed_dataset.csv"

            # Ensure output directory exists
            final_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the processed dataset
            merged_df.to_csv(final_output_path, index=False)
            logger.success(f"Processed dataset saved to: {final_output_path}")

            # Print summary
            logger.info(f"  Total rows: {len(merged_df)}")
            logger.info(f"  Date range: {merged_df['date_only'].min()} to {merged_df['date_only'].max()}")
            logger.info(f"  Total columns: {len(merged_df.columns)}")

            sentiment_cols = [col for col in merged_df.columns if any(
                model in col for model in ['finbert', 'roberta', 'deberta']
            )]
            logger.info(f"  Sentiment columns: {len(sentiment_cols)}")

        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            raise


if __name__ == "__main__":
    app()

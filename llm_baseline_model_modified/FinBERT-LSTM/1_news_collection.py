"""
Step 1: Load news data from raw CSV and prepare for sentiment analysis.

Original: Fetched 10 articles/day from NYT API in wide format.
Adapted: Loads from data/raw/news.csv (long format, ticker-specific),
         concatenates headline + summary for FinBERT input.
"""
import pandas as pd
from config import NEWS_RAW, NEWS_DATA_CSV, TICKERS


def load_and_prepare_news():
    df = pd.read_csv(NEWS_RAW)

    # Extract date from datetime column (YYYY-MM-DD HH:MM:SS -> YYYY-MM-DD)
    df["date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")

    # Filter to our tickers only
    df = df[df["ticker"].isin(TICKERS)].copy()

    # Concatenate headline + summary for richer FinBERT input
    # Original used NYT abstracts; this gives equivalent context
    df["summary"] = df["summary"].fillna("")
    df["text"] = df["headline"] + ". " + df["summary"]

    # Keep only what downstream scripts need
    df = df[["date", "ticker", "text"]].reset_index(drop=True)

    df.to_csv(NEWS_DATA_CSV, index=False)
    print(f"News data saved: {len(df)} articles across {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Articles per ticker:\n{df['ticker'].value_counts().to_string()}")


if __name__ == "__main__":
    load_and_prepare_news()

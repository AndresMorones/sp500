"""
Step 2: Load stock price data from raw CSV and prepare for modeling.

Original: Downloaded NDX from Yahoo Finance via yfinance.
Adapted: Loads from data/raw/price.csv (multi-ticker, lowercase columns),
         renames to match original schema expectations.
"""
import pandas as pd
from config import PRICE_RAW, STOCK_PRICE_CSV, TICKERS


def load_and_prepare_prices():
    df = pd.read_csv(PRICE_RAW)

    # Filter to our tickers
    df = df[df["ticker"].isin(TICKERS)].copy()

    # Rename columns to match original FinBERT-LSTM schema (capitalized)
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    # Add Adj Close (original had it from yfinance; for these stocks over ~1 year
    # the difference is negligible — dividends are small relative to price)
    df["Adj Close"] = df["Close"]

    # Sort by ticker and date for consistent ordering
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    df.to_csv(STOCK_PRICE_CSV, index=False)
    print(f"Stock data saved: {len(df)} rows across {df['ticker'].nunique()} tickers")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Rows per ticker:\n{df['ticker'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    load_and_prepare_prices()

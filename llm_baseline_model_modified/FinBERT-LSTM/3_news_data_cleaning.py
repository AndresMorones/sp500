"""
Step 3: Align news data with stock trading days.

Original: Filtered wide-format news to dates present in stock_price.csv.
Adapted: Per-ticker alignment — only keep news articles whose date matches
         a trading day for that specific ticker (drops weekend/holiday news).
"""
import pandas as pd
from config import NEWS_DATA_CSV, STOCK_PRICE_CSV


def clean_news_data():
    news_df = pd.read_csv(NEWS_DATA_CSV)
    stock_df = pd.read_csv(STOCK_PRICE_CSV)

    # Build set of valid (ticker, date) pairs from stock data
    stock_df["Date"] = stock_df["Date"].astype(str).str[:10]
    valid_pairs = set(zip(stock_df["ticker"], stock_df["Date"]))

    # Filter news to only trading days for the matching ticker
    mask = news_df.apply(lambda r: (r["ticker"], r["date"]) in valid_pairs, axis=1)
    news_clean = news_df[mask].reset_index(drop=True)

    dropped = len(news_df) - len(news_clean)
    news_clean.to_csv(NEWS_DATA_CSV, index=False)
    print(f"News cleaned: {len(news_clean)} articles kept, {dropped} dropped (non-trading days)")
    print(f"Articles per ticker:\n{news_clean['ticker'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    clean_news_data()

"""
Feature engineering: generate prediction target columns from preprocessed datasets.

For each ticker, reads the preprocessed dataset and adds:
    - Float_Price:  next day's closing price
    - Binary_Price: 1 if next close > today's close, else 0
    - Factor_Price: next_close / today_close
    - Delta_Price:  next_close - today_close

Usage:
    python stock_prediction/features.py              # Process all tickers
    python stock_prediction/features.py --ticker AAPL  # Single ticker
"""
from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd

from stock_prediction.config import PROCESSED_DATA_DIR, TICKERS

app = typer.Typer()


@app.command()
def main(
    ticker: str = typer.Option("all", help="Ticker to process, or 'all' for all tickers"),
):
    tickers = TICKERS if ticker == "all" else [ticker]

    for t in tickers:
        input_path = PROCESSED_DATA_DIR / f"{t}_preprocessed_dataset.csv"
        output_path = PROCESSED_DATA_DIR / f"{t}_preprocessed_dataset_with_features.csv"

        if not input_path.exists():
            logger.warning(f"Input not found for {t}: {input_path}, skipping")
            continue

        logger.info(f"Generating features for {t}...")
        df = pd.read_csv(input_path)

        # Generate output features based on next day's closing price
        df["Float_Price"] = df["Close"].shift(-1)
        df["Binary_Price"] = (df["Float_Price"] > df["Close"]).astype(int)
        df["Factor_Price"] = df["Float_Price"] / df["Close"]
        df["Delta_Price"] = df["Float_Price"] - df["Close"]

        df.to_csv(output_path, index=False)
        logger.success(f"  {t}: {len(df)} rows -> {output_path}")

    logger.success("Features generation complete.")


if __name__ == "__main__":
    app()

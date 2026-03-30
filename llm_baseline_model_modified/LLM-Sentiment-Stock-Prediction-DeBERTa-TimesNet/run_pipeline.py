#!/usr/bin/env python3
"""
End-to-end pipeline runner for the adapted DeBERTa-TimesNet framework.

Runs:
  1. Sentiment analysis (FinBERT, RoBERTa, DeBERTa on all ticker news)
  2. Dataset processing (merge price + sentiment, aggregate by date)
  3. Feature engineering (Float_Price, Binary_Price, Factor_Price, Delta_Price)
  4. LSTM training + prediction (single seed, all tickers, deberta features)
  5. Results analysis and summary report

Usage:
    PYTHONPATH=. python run_pipeline.py                 # Full pipeline
    PYTHONPATH=. python run_pipeline.py --skip-sentiment  # Skip step 1 if already done
    PYTHONPATH=. python run_pipeline.py --skip-training   # Skip step 4
"""
import sys
import os
import argparse
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "sentiment_analysis"))

from stock_prediction.config import (
    PROJ_ROOT, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR,
    TICKERS, MODELS_DIR, REPORTS_DIR, FIGURES_DIR,
)
from loguru import logger


def step1_sentiment_analysis():
    """Run transformer sentiment analysis on all news articles."""
    logger.info("=" * 70)
    logger.info("STEP 1: Sentiment Analysis")
    logger.info("=" * 70)
    from sentiment_analysis.main import main as run_sentiment
    run_sentiment()


def step2_dataset_processing():
    """Merge price data with aggregated sentiment scores."""
    logger.info("=" * 70)
    logger.info("STEP 2: Dataset Processing")
    logger.info("=" * 70)

    from stock_prediction.dataset import load_price_data, load_sentiment_data
    from stock_prediction.dataset import aggregate_sentiment_by_date, merge_price_and_sentiment
    import pandas as pd

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in TICKERS:
        logger.info(f"Processing {ticker}...")
        price_df = load_price_data(ticker)
        news_df = load_sentiment_data(ticker)

        if len(news_df) > 0:
            sentiment_agg_df = aggregate_sentiment_by_date(news_df)
            logger.info(f"  Aggregated sentiment: {sentiment_agg_df.shape}")
        else:
            logger.warning(f"  No sentiment data for {ticker}")
            sentiment_agg_df = pd.DataFrame(columns=['date_only'])

        merged_df = merge_price_and_sentiment(price_df, sentiment_agg_df)
        output_path = PROCESSED_DATA_DIR / f"{ticker}_preprocessed_dataset.csv"
        merged_df.to_csv(output_path, index=False)
        logger.success(f"  {ticker}: {len(merged_df)} rows -> {output_path}")


def step3_feature_engineering():
    """Generate prediction target columns."""
    logger.info("=" * 70)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 70)

    import pandas as pd

    for ticker in TICKERS:
        input_path = PROCESSED_DATA_DIR / f"{ticker}_preprocessed_dataset.csv"
        output_path = PROCESSED_DATA_DIR / f"{ticker}_preprocessed_dataset_with_features.csv"

        if not input_path.exists():
            logger.warning(f"  {ticker}: input not found, skipping")
            continue

        df = pd.read_csv(input_path)
        df["Float_Price"] = df["Close"].shift(-1)
        df["Binary_Price"] = (df["Float_Price"] > df["Close"]).astype(int)
        df["Factor_Price"] = df["Float_Price"] / df["Close"]
        df["Delta_Price"] = df["Float_Price"] - df["Close"]

        df.to_csv(output_path, index=False)
        logger.success(f"  {ticker}: {len(df)} rows -> {output_path}")


def step4_train_and_predict():
    """Train LSTM on each ticker with deberta features and generate predictions."""
    logger.info("=" * 70)
    logger.info("STEP 4: LSTM Training & Prediction")
    logger.info("=" * 70)

    import subprocess

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "output" / "LSTM").mkdir(parents=True, exist_ok=True)

    seed = 42
    model = "deberta"
    feature_columns = "Close Volume total_news_count deberta_majority_vote deberta_count_positive deberta_count_negative deberta_count_neutral deberta_label_positive_sum deberta_label_negative_sum deberta_label_neutral_sum"

    for target_column in ["Binary_Price", "Float_Price", "Delta_Price", "Factor_Price"]:
        for ticker in TICKERS:
            data_path = f"data/processed/{ticker}_preprocessed_dataset_with_features.csv"
            if not (PROJ_ROOT / data_path).exists():
                logger.warning(f"  {ticker}: data not found, skipping")
                continue

            model_path = f"models/lstm_{target_column}_{model}_model_{seed}.pth"
            scaler_path = f"models/{ticker}_lstm_{target_column}_{model}_scaler.pkl"

            logger.info(f"  Training LSTM: {ticker} / {target_column} / {model} / seed={seed}")
            cmd = [
                sys.executable, "stock_prediction/modeling/train.py",
                "--data_path", data_path,
                "--feature_columns", *feature_columns.split(),
                "--target_column", target_column,
                "--model_path", model_path,
                "--scaler_path", scaler_path,
                "--seq_length", "30",
                "--hidden_size", "64",
                "--num_layers", "2",
                "--dropout", "0.5",
                "--test_ratio", "0.2",
                "--lr", "0.001",
                "--epochs", "20",
                "--force_retrain",
                "--seed", str(seed),
            ]
            result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"  Training failed: {result.stderr[-500:]}")
                continue

            logger.info(f"  Predicting: {ticker} / {target_column}")
            cmd = [
                sys.executable, "stock_prediction/modeling/predict.py",
                "--model_path", model_path,
                "--scaler_path", scaler_path,
                "--data_path", data_path,
                "--feature_columns", *feature_columns.split(),
                "--target_column", target_column,
                "--ticker", ticker,
                "--seq_length", "30",
                "--hidden_size", "64",
                "--num_layers", "2",
                "--dropout", "0.5",
                "--test_ratio", "0.2",
                "--news_model", model,
                "--seed", str(seed),
            ]
            result = subprocess.run(cmd, cwd=str(PROJ_ROOT), capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"  Prediction failed: {result.stderr[-500:]}")


def step5_analyze_results():
    """Generate summary analysis of all results."""
    logger.info("=" * 70)
    logger.info("STEP 5: Results Analysis")
    logger.info("=" * 70)

    import pandas as pd
    import numpy as np
    import json

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DEBERTA-TIMESNET BASELINE: PIPELINE RESULTS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # --- Data summary ---
    report_lines.append("1. DATA SUMMARY")
    report_lines.append("-" * 50)

    for ticker in TICKERS:
        fpath = PROCESSED_DATA_DIR / f"{ticker}_preprocessed_dataset_with_features.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            news_days = (df['total_news_count'] > 0).sum() if 'total_news_count' in df.columns else 0
            report_lines.append(
                f"  {ticker}: {len(df)} rows, {news_days} news days "
                f"({news_days/len(df)*100:.1f}%), "
                f"Close range [{df['Close'].min():.2f}, {df['Close'].max():.2f}]"
            )
        else:
            report_lines.append(f"  {ticker}: NOT FOUND")
    report_lines.append("")

    # --- Sentiment summary ---
    report_lines.append("2. SENTIMENT ANALYSIS SUMMARY")
    report_lines.append("-" * 50)

    for ticker in TICKERS:
        spath = INTERIM_DATA_DIR / f"{ticker}_sentiment_results.csv"
        if spath.exists():
            sdf = pd.read_csv(spath)
            if 'finbert_sentiment' in sdf.columns:
                fb = sdf['finbert_sentiment'].value_counts().to_dict()
                report_lines.append(f"  {ticker}: {len(sdf)} articles, FinBERT distribution: {fb}")
        else:
            report_lines.append(f"  {ticker}: no sentiment data")
    report_lines.append("")

    # --- Model results ---
    report_lines.append("3. LSTM MODEL RESULTS")
    report_lines.append("-" * 50)

    results_dir = REPORTS_DIR / "output" / "LSTM"
    if results_dir.exists():
        txt_files = list(results_dir.glob("**/*_pred_vs_true.txt"))
        report_lines.append(f"  Found {len(txt_files)} result files")

        for txt_file in sorted(txt_files):
            try:
                with open(txt_file, 'r') as f:
                    content = f.read()
                for line in content.split('\n'):
                    if 'Metrics:' in line:
                        report_lines.append(f"  {txt_file.name}: {line.strip()}")
                        break
            except Exception:
                pass
    else:
        report_lines.append("  No LSTM results found (training not yet run)")
    report_lines.append("")

    # --- Feature correlation ---
    report_lines.append("4. FEATURE-TARGET CORRELATION ANALYSIS")
    report_lines.append("-" * 50)

    for ticker in TICKERS[:3]:  # Just first 3 for brevity
        fpath = PROCESSED_DATA_DIR / f"{ticker}_preprocessed_dataset_with_features.csv"
        if fpath.exists():
            df = pd.read_csv(fpath)
            targets = ['Float_Price', 'Binary_Price', 'Delta_Price']
            sentiment_cols = [c for c in df.columns if any(m in c for m in ['finbert', 'roberta', 'deberta', 'total_news'])]
            if sentiment_cols and 'Delta_Price' in df.columns:
                corrs = df[sentiment_cols].corrwith(df['Delta_Price']).abs().sort_values(ascending=False)
                report_lines.append(f"  {ticker} top correlations with Delta_Price:")
                for col, val in corrs.head(5).items():
                    report_lines.append(f"    {col}: {val:.4f}")
    report_lines.append("")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = OUTPUT_DIR / "pipeline_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.success(f"Report saved to {report_path}")
    print(report_text)


def main():
    parser = argparse.ArgumentParser(description="Run DeBERTa-TimesNet pipeline")
    parser.add_argument("--skip-sentiment", action="store_true", help="Skip sentiment analysis step")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training step")
    args = parser.parse_args()

    start = time.time()

    if not args.skip_sentiment:
        step1_sentiment_analysis()
    else:
        logger.info("Skipping sentiment analysis (--skip-sentiment)")

    step2_dataset_processing()
    step3_feature_engineering()

    if not args.skip_training:
        step4_train_and_predict()
    else:
        logger.info("Skipping training (--skip-training)")

    step5_analyze_results()

    elapsed = time.time() - start
    logger.success(f"\nPipeline complete in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()

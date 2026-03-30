"""
Run all LSTM training+prediction for a single seed.
Usage: python run_seed.py --seed 16
"""
import subprocess, sys, argparse, time
from pathlib import Path

PROJ = Path(__file__).resolve().parent
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
MODELS = {
    "deberta": "Close Volume total_news_count deberta_majority_vote deberta_count_positive deberta_count_negative deberta_count_neutral deberta_label_positive_sum deberta_label_negative_sum deberta_label_neutral_sum",
    "finbert": "Close Volume total_news_count finbert_majority_vote finbert_count_positive finbert_count_negative finbert_count_neutral finbert_label_positive_sum finbert_label_negative_sum finbert_label_neutral_sum",
    "roberta": "Close Volume total_news_count roberta_majority_vote roberta_count_positive roberta_count_negative roberta_count_neutral roberta_label_positive_sum roberta_label_negative_sum roberta_label_neutral_sum",
}
TARGETS = ["Binary_Price", "Float_Price", "Factor_Price", "Delta_Price"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed

    t0 = time.time()
    total = len(MODELS) * len(TARGETS) * len(TICKERS)
    done = 0

    for model, features in MODELS.items():
        for target in TARGETS:
            for ticker in TICKERS:
                done += 1
                print(f"[seed={seed}] {done}/{total} {ticker}/{model}/{target}", flush=True)

                # Train
                subprocess.run([
                    sys.executable, "stock_prediction/modeling/train.py",
                    "--model_path", f"models/lstm_price_{target}_{model}_model_{seed}.pth",
                    "--data_path", f"data/processed/{ticker}_preprocessed_dataset_with_features.csv",
                    "--feature_columns", *features.split(),
                    "--target_column", target,
                    "--scaler_path", f"models/{ticker}_lstm_price_{target}_{model}_scaler.pkl",
                    "--seq_length", "30", "--hidden_size", "64", "--num_layers", "2",
                    "--dropout", "0.5", "--test_ratio", "0.2", "--lr", "0.001",
                    "--epochs", "20", "--force_retrain", "--seed", str(seed),
                ], cwd=str(PROJ), capture_output=True)

                # Predict
                subprocess.run([
                    sys.executable, "stock_prediction/modeling/predict.py",
                    "--model_path", f"models/lstm_price_{target}_{model}_model_{seed}.pth",
                    "--scaler_path", f"models/{ticker}_lstm_price_{target}_{model}_scaler.pkl",
                    "--data_path", f"data/processed/{ticker}_preprocessed_dataset_with_features.csv",
                    "--feature_columns", *features.split(),
                    "--target_column", target, "--ticker", ticker,
                    "--seq_length", "30", "--hidden_size", "64", "--num_layers", "2",
                    "--dropout", "0.5", "--test_ratio", "0.2",
                    "--news_model", model, "--seed", str(seed),
                ], cwd=str(PROJ), capture_output=True)

    elapsed = time.time() - t0
    print(f"[seed={seed}] ALL DONE in {elapsed/60:.1f} minutes", flush=True)

if __name__ == "__main__":
    main()

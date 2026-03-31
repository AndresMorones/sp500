"""Phase 3 shared configuration — model hyperparameters, paths, and constants."""

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "output")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

PRICE_CSV = os.path.join(RAW_DIR, "price.csv")
SP500_CSV = os.path.join(RAW_DIR, "S&P 500 Historical Data.csv")
SCORES_CSV = os.path.join(OUT_DIR, "scores_output.csv")

# --- Tickers ---
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
METRIC_A_TICKERS = ["GOOGL"]  # only GOOGL has news_phase2 data

# --- Data split ---
LOOKBACK = 10                          # days of history per sample
SPLIT_RATIOS = (0.72, 0.08, 0.20)     # train / val / test

# --- LSTM standardized parameters ---
LSTM_HIDDEN = [32, 16]                 # 2 layers
LSTM_DROPOUT = 0.3
LSTM_LR = 0.001
LSTM_LOSS = "mse"
LSTM_BATCH = 32
LSTM_EPOCHS = 50
LSTM_PATIENCE = 10                     # early stopping on val_loss
LSTM_OPTIMIZER = "adam"
SEEDS = [16, 32, 42, 64, 128]

# --- Ridge ---
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# --- LightGBM (conservative for small dataset) ---
LGBM_PARAMS = {
    "objective": "huber",
    "huber_delta": 1.35,
    "max_depth": 3,
    "num_leaves": 8,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbosity": -1,
}
LGBM_ROUNDS = 150
LGBM_EARLY_STOP = 20

# --- Targets ---
# gap: predict next-day open; cc: predict next-day close
TARGETS = ["gap", "cc"]

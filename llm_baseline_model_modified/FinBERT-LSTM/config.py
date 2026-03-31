"""
Shared configuration for FinBERT-LSTM baseline pipeline.
Adapted from arXiv:2211.07392 to work with our multi-ticker dataset.

Sentiment model selection:
  Set SENTIMENT_MODEL to switch between models.
  Each model gets its own output folder for clean comparison.
  Models are either "classifier" (pipeline-based) or "generative" (prompt-based).
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# --- Sentiment model selection ---
# Override at runtime via env var (used by run_all_models.py):
#   FINBERT_SENTIMENT_MODEL=ProsusAI/finbert python 4_news_sentiment_analysis.py
SENTIMENT_MODEL = os.getenv("FINBERT_SENTIMENT_MODEL", "ProsusAI/finbert")

# Model registry: type determines inference path, tag determines output folder
_MODEL_REGISTRY = {
    "ProsusAI/finbert": {
        "type": "classifier",
        "tag": "finbert_lstm_results",
    },
    "mrm8488/deberta-v3-ft-financial-news-sentiment-analysis": {
        "type": "classifier",
        "tag": "deberta_v3_lstm_results",
    },
    "google/gemma-3-1b-it": {
        "type": "generative",
        "tag": "gemma_3_1b_lstm_results",
    },
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "type": "generative",
        "tag": "qwen25_lstm_results",
    },
    "oopere/Llama-FinSent-S": {
        "type": "generative",
        "tag": "llama_finsent_lstm_results",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "type": "generative",
        "tag": "llama32_1b_lstm_results",
    },
}

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output",
                          _MODEL_REGISTRY[SENTIMENT_MODEL]["tag"])

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]

# Hyperparameters (from original paper)
SEQUENCE_LENGTH = 10
SPLIT_RATIO = 0.85
EPOCHS = 100
MLP_LEARNING_RATE = 0.01
LSTM_LEARNING_RATE = 0.02

# File paths — raw inputs
NEWS_RAW = os.path.join(RAW_DIR, "news.csv")
PRICE_RAW = os.path.join(RAW_DIR, "price.csv")

# File paths — intermediate outputs (stored locally)
NEWS_DATA_CSV = os.path.join(OUTPUT_DIR, "news_data.csv")
STOCK_PRICE_CSV = os.path.join(OUTPUT_DIR, "stock_price.csv")
SENTIMENT_CSV = os.path.join(OUTPUT_DIR, "sentiment.csv")

# File paths — model results
MLP_RESULTS_CSV = os.path.join(OUTPUT_DIR, "mlp_results.csv")
LSTM_RESULTS_CSV = os.path.join(OUTPUT_DIR, "lstm_results.csv")
BERT_LSTM_RESULTS_CSV = os.path.join(OUTPUT_DIR, "bert_lstm_results.csv")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

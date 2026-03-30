from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# ---------------------------------------------------------------------------
# Path configuration adapted for sp500 project data layout
# ---------------------------------------------------------------------------

# PROJ_ROOT = this project's directory (LLM-Sentiment-Stock-Prediction-DeBERTa-TimesNet/)
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# PROJECT_ROOT = top-level sp500/sp500/ directory containing data/raw/
PROJECT_ROOT = PROJ_ROOT.parents[1]

# --- Raw data (consolidated single-file CSVs) ---
NEWS_CSV = PROJECT_ROOT / "data" / "raw" / "news.csv"
PRICE_CSV = PROJECT_ROOT / "data" / "raw" / "price.csv"

# --- Internal working directories (inside this project) ---
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Output directory for final results ---
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "deberta_timesnet_results"

# --- Tickers and sentiment models ---
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
SENTIMENT_MODELS = ["finbert", "roberta", "deberta"]

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

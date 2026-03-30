import pandas as pd
import re
from loguru import logger
from sklearn.metrics import accuracy_score
from models.base import BaseSentimentComparison


class StockSentimentComparison(BaseSentimentComparison):
    """Sentiment comparison for stock news articles.

    Adapted to work with the sp500 project's consolidated news.csv format:
        columns: datetime, ticker, headline, summary
    No ground-truth sentiment labels are available, so classical-model
    training and original_sentiment comparisons are removed.
    """

    def __init__(self, path, ticker, df=None):
        """
        Args:
            path: unused (kept for interface compat); pass None.
            ticker: stock ticker symbol.
            df: pre-filtered DataFrame with columns [datetime, ticker, headline, summary].
        """
        self._source_df = df
        super().__init__(path, ticker, has_ground_truth=False)

    @staticmethod
    def _drop_text_duplicates(
        df: pd.DataFrame,
        text_col: str,
        *,
        normalise: bool = True,
        keep: str = "first",
        inplace: bool = False
    ) -> pd.DataFrame:
        work = df if inplace else df.copy()
        if normalise:
            def _canon(text: str) -> str:
                text = str(text).lower()
                text = re.sub(r"\s+", " ", text)
                text = re.sub(r"[^\w\s]", "", text)
                return text.strip()

            canon_col = f"__canon_{text_col}"
            work[canon_col] = work[text_col].map(_canon, na_action="ignore")
            deduped = work.drop_duplicates(subset=[canon_col], keep=keep)
            deduped = deduped.drop(columns=[canon_col])
        else:
            deduped = work.drop_duplicates(subset=[text_col], keep=keep)

        return deduped

    def load_fin_data(self, path):
        """Load and prepare news data from a pre-filtered DataFrame."""
        if self._source_df is not None:
            df = self._source_df.copy()
        elif path is not None:
            df = pd.read_csv(path)
        else:
            raise ValueError("No data source provided")

        # Build concatenated text column: headline + ". " + summary
        df['text'] = df['headline'].fillna('') + ". " + df['summary'].fillna('')
        df['text'] = df['text'].str.strip()

        # Parse date
        df['date'] = pd.to_datetime(df['datetime']).dt.date

        # Deduplicate by text content
        df = self._drop_text_duplicates(df, 'text')

        return df

    def create_sample_data(self):
        temp = self.load_fin_data(self.path)
        sample_data = {
            'text': temp['text'],
            'date': temp['date'],
        }
        return pd.DataFrame(sample_data)

    def sentiment_to_numeric(self, series) -> pd.Series:
        """Map sentiment labels to numeric values.
        Kept for interface compatibility but not used in this adapter
        since transformer models produce 0/1/2 directly via map_sentiment_scores.
        """
        mapping = {
            "Bearish": 0,
            "Somewhat-Bearish": 0,
            "Neutral": 1,
            "Somewhat-Bullish": 2,
            "Bullish": 2
        }
        if pd.api.types.is_categorical_dtype(series):
            return series.astype(str).map(mapping).astype("Int8")
        else:
            return series.map(mapping).astype("Int8")

    def create_detailed_comparison_table(self, df):
        """Log inter-model agreement statistics (no ground truth available)."""
        logger.info("\n" + "=" * 80)
        logger.success("INTER-MODEL AGREEMENT REPORT")
        logger.info("=" * 80)
        models = ['finbert_sentiment', 'roberta_sentiment', 'deberta_sentiment']
        model_names = ['FinBERT', 'RoBERTa', 'DeBERTa']

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                agreement = accuracy_score(df[models[i]], df[models[j]])
                logger.success(f"{model_names[i]} vs {model_names[j]} agreement: {agreement:.3f}")

        # Majority vote across the 3 transformer models
        from scipy.stats import mode as scipy_mode
        sentiment_cols = df[models].values
        majority = scipy_mode(sentiment_cols, axis=1, keepdims=False).mode
        df['transformer_majority_vote'] = majority

        for model, name in zip(models, model_names):
            agreement_with_majority = accuracy_score(df['transformer_majority_vote'], df[model])
            logger.info(f"{name} agreement with majority vote: {agreement_with_majority:.3f}")

        # Distribution summary
        logger.info("\nSentiment distribution (majority vote):")
        dist = pd.Series(majority).value_counts().sort_index()
        label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        for val, count in dist.items():
            logger.info(f"  {label_map.get(val, val)}: {count} ({count/len(majority)*100:.1f}%)")

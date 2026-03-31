# Sentiment Model Comparison Card

Models used in the FinBERT-LSTM baseline pipeline for S&P 500 price prediction.

## Sentiment-Specific Models

| | FinBERT | DeBERTa-v3 | Llama-FinSent-S |
|---|---|---|---|
| **HuggingFace ID** | `ProsusAI/finbert` | `mrm8488/deberta-v3-ft-financial-news-sentiment-analysis` | `oopere/Llama-FinSent-S` |
| **Type** | Classifier | Classifier | Generative (LoRA) |
| **Params** | 110M | 142M | 914M (pruned from 1.23B) |
| **Base model** | BERT-base-uncased | DeBERTa-v3-small | Llama 3.2-1B (40% pruned) |
| **Input modalities** | Text | Text | Text |
| **Output modalities** | 3-class sentiment (pos/neg/neutral) | 3-class sentiment (pos/neg/neutral) | 7-class sentiment (strong neg → strong pos) |
| **Context length** | 512 tokens | 512 tokens | 128k tokens (truncated to 512) |
| **Training data** | Reuters TRC2 financial corpus + Financial PhraseBank (4,840 sentences) | Financial PhraseBank (4,840 sentences) | FinGPT sentiment (Twitter Financial News ~9,540 + FiQA ~961 samples) |
| **Pre-training token count** | ~3.3B (BERT) | ~160GB text (DeBERTa-v3) | Up to 9T tokens (Llama 3.2 base) |
| **Knowledge cutoff** | ~2018 (BERT pre-training); fine-tuning data from pre-2014 | ~2021 (DeBERTa-v3 pre-training); fine-tuning data from pre-2014 | December 2023 (Llama 3.2 base); fine-tuning data pre-2023 |
| **Release date** | August 2019 | January 2024 | February 2025 |
| **Paper** | arXiv:1908.10063 | — | — |
| **Pipeline MAPE (%)** | **2.14** | **2.18** | **2.21** |

## General-Purpose LLMs (zero-shot sentiment via 7-class prompting)

| | Qwen2.5-1.5B | Gemma-3-1B | Llama-3.2-1B |
|---|---|---|---|
| **HuggingFace ID** | `Qwen/Qwen2.5-1.5B-Instruct` | `google/gemma-3-1b-it` | `meta-llama/Llama-3.2-1B-Instruct` |
| **Type** | Generative (instruction-tuned) | Generative (instruction-tuned) | Generative (instruction-tuned) |
| **Params** | 1.54B (1.31B non-embedding) | ~1B | 1.23B |
| **Base model** | Qwen2.5-1.5B | Gemma-3-1b-pt | Llama-3.2-1B |
| **Input modalities** | Multilingual text | Multilingual text | Multilingual text |
| **Output modalities** | Multilingual text and code | Multilingual text and code | Multilingual text and code |
| **Context length** | 128k tokens | 32k tokens | 128k tokens |
| **GQA** | Yes | Yes | Yes |
| **Training data** | Publicly available online data | Publicly available online data | A new mix of publicly available online data |
| **Pre-training token count** | 18T tokens | 2T tokens | Up to 9T tokens |
| **Knowledge cutoff** | ~Early 2024 (not officially documented) | August 2024 | December 2023 |
| **Release date** | September 2024 | March 2025 | September 2024 |
| **Paper** | arXiv:2412.15115 | — | — |
| **Pipeline MAPE (%)** | **2.37** | **2.72** | *pending (gated — license not yet accepted)* |

## Baselines (no sentiment)

| | LSTM | MLP |
|---|---|---|
| **Type** | Price-only (10-day window) | Price-only (10-day window) |
| **Architecture** | 3-layer LSTM (70→30→10) | Single hidden layer |
| **Pipeline MAPE (%)** | 2.32 | 3.66 |

## Results — MAPE (%) by Ticker

| Ticker | MLP | LSTM | LSTM+FinBERT | LSTM+DeBERTa-v3 | LSTM+Llama-FinSent | LSTM+Qwen2.5-1.5B | LSTM+Gemma-3-1B |
|--------|-----|------|-------------|----------------|-------------------|------------------|----------------|
| AAPL | 2.01 | 1.71 | 1.42 | **0.99** | 1.77 | 2.06 | 1.00 |
| AMZN | 1.74 | **1.05** | 1.18 | 1.87 | 1.52 | 1.60 | 1.54 |
| GOOGL | 1.12 | 0.94 | **0.74** | 0.78 | 0.90 | 1.77 | 0.98 |
| META | 10.94 | 3.68 | 4.80 | **3.47** | 4.18 | 4.27 | 6.15 |
| MSFT | **0.60** | 0.61 | 0.92 | 0.85 | 0.74 | 0.63 | 0.71 |
| NVDA | 6.22 | 3.34 | **2.84** | 4.44 | 2.98 | 3.68 | 4.89 |
| TSLA | 3.00 | 4.89 | 3.07 | 2.84 | 3.41 | **2.60** | 3.80 |
| **MEAN** | **3.66** | **2.32** | **2.14** | **2.18** | **2.21** | **2.37** | **2.72** |

## Results — MAE (USD) by Ticker

| Ticker | MLP | LSTM | LSTM+FinBERT | LSTM+DeBERTa-v3 | LSTM+Llama-FinSent | LSTM+Qwen2.5-1.5B | LSTM+Gemma-3-1B |
|--------|-----|------|-------------|----------------|-------------------|------------------|----------------|
| AAPL | 4.64 | 3.94 | 3.28 | **2.27** | 4.10 | 4.76 | 2.31 |
| AMZN | 3.28 | **1.95** | 2.22 | 3.48 | 2.85 | 3.00 | 2.90 |
| GOOGL | 1.85 | 1.55 | **1.21** | 1.29 | 1.46 | 2.90 | 1.60 |
| META | 63.21 | 21.28 | 27.79 | **20.08** | 24.18 | 24.69 | 35.56 |
| MSFT | **2.55** | 2.57 | 3.87 | 3.61 | 3.11 | 2.67 | 3.00 |
| NVDA | 8.23 | 4.47 | **3.70** | 5.99 | 3.92 | 4.57 | 6.58 |
| TSLA | 7.35 | 11.39 | 7.62 | 7.00 | 8.20 | **6.38** | 9.39 |
| **MEAN** | **13.02** | **6.73** | **7.10** | **6.25** | **6.83** | **7.00** | **8.76** |

Bold = best model per ticker.

## Notes

- **Classifier models (FinBERT, DeBERTa-v3)** have no knowledge cutoff in the LLM sense. They are encoder models that classify sentiment patterns in input text. Their fine-tuning data (Financial PhraseBank, pre-2014) teaches sentiment language patterns, not world knowledge.
- **Llama-FinSent-S** is not a general-purpose LLM. It is a pruned and LoRA fine-tuned version of Llama 3.2-1B, trained specifically for financial sentiment classification.
- **General-purpose LLMs** perform zero-shot sentiment classification using a 7-class prompt template mapped to [-1, 1].
- All models use the same downstream architecture: 3-layer LSTM with 10-day price history + current-day sentiment score.
- MAPE is mean absolute percentage error averaged across 7 tickers (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA).

"""
Step 4: Compute sentiment scores for news articles.

Supports two inference modes (set via config.SENTIMENT_MODEL):
  - "classifier": FinBERT/DeBERTa via pipeline("sentiment-analysis"), batch inference
  - "generative": LLMs (Gemma, Qwen, Llama-FinSent) via prompt-based generate()

All generative models use 7 granular sentiment labels mapped to [-1, 1].
Output contract: sentiment.csv with columns [date, ticker, finbert_score].
Column name "finbert_score" is legacy — kept for downstream compatibility with step 7.
"""
import os
import json
import time
import pandas as pd
from config import NEWS_DATA_CSV, SENTIMENT_CSV, SENTIMENT_MODEL, _MODEL_REGISTRY


# ---------------------------------------------------------------------------
# 7-class sentiment scale (used by all generative models)
# ---------------------------------------------------------------------------
_7CLASS_MAP = {
    "strong positive": 1.0,
    "moderately positive": 0.66,
    "mildly positive": 0.33,
    "neutral": 0.0,
    "mildly negative": -0.33,
    "moderately negative": -0.66,
    "strong negative": -1.0,
}

_PROMPT_TEMPLATE = (
    "What is the sentiment of this news? Please choose an answer from "
    "{{strong negative/moderately negative/mildly negative/"
    "neutral/mildly positive/moderately positive/strong positive}}\n"
    "Input: {text}\n"
    "Answer:"
)

# Models that need chat template wrapping (instruction-tuned, not fine-tuned on raw prompts)
_CHAT_TEMPLATE_MODELS = {
    "google/gemma-3-1b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
}


# ---------------------------------------------------------------------------
# Classifier path (FinBERT, DeBERTa — unchanged from original)
# ---------------------------------------------------------------------------
def load_classifier_pipeline():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer,
                   truncation=True, max_length=512)
    return nlp


def classifier_score_to_value(result):
    """Map classifier output to [-1, 1]. Handles label casing differences across models."""
    label = result["label"].lower()
    score = result["score"]
    if label == "positive":
        return score
    elif label == "neutral":
        return 0.0
    else:
        return -score


def run_classifier_inference(texts):
    print(f"Loading classifier model: {SENTIMENT_MODEL}")
    nlp = load_classifier_pipeline()
    print(f"Running classifier inference on {len(texts)} articles (batch_size=32)...")
    results = nlp(texts, batch_size=32)
    return [classifier_score_to_value(r) for r in results]


# ---------------------------------------------------------------------------
# Generative path (Gemma, Qwen, Llama-FinSent)
# ---------------------------------------------------------------------------
def _get_torch_device():
    """Auto-detect best available device: MPS (Apple Silicon GPU) > CUDA > CPU."""
    import torch
    if torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon GPU)")
        return "mps", torch.float16
    elif torch.cuda.is_available():
        print("Device: CUDA GPU")
        return "cuda", torch.float16
    else:
        print("Device: CPU (no GPU found)")
        return "cpu", torch.float32


def load_generative_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device, dtype = _get_torch_device()
    print(f"Loading generative model: {SENTIMENT_MODEL} [{dtype}]")
    tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_MODEL, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        SENTIMENT_MODEL, dtype=dtype,
        device_map=device, trust_remote_code=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_prompt(text, tokenizer):
    """Build prompt, wrapping in chat template for instruction-tuned models."""
    raw_prompt = _PROMPT_TEMPLATE.format(text=text)

    if SENTIMENT_MODEL in _CHAT_TEMPLATE_MODELS:
        messages = [{"role": "user", "content": raw_prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return raw_prompt


def parse_generative_output(generated_text):
    """Extract sentiment label from generated text, map to [-1, 1] via 7-class scale."""
    text = generated_text.lower().strip()
    # Match longest labels first ("strong positive" before "positive")
    for label in sorted(_7CLASS_MAP.keys(), key=len, reverse=True):
        if label in text:
            return _7CLASS_MAP[label], label
    return 0.0, None  # unparseable → neutral


def run_generative_inference(texts):
    """Score articles one-by-one with checkpointing for crash recovery."""
    import torch

    model, tokenizer = load_generative_model()

    checkpoint_dir = os.path.dirname(SENTIMENT_CSV)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "_sentiment_checkpoint.json")

    # Load checkpoint if exists
    scored = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            scored = json.load(f)
        print(f"Resumed from checkpoint: {len(scored)}/{len(texts)} already scored")

    scores = [None] * len(texts)
    total = len(texts)
    start_time = time.time()
    new_count = 0
    parse_failures = 0

    for i, text in enumerate(texts):
        # Use checkpoint if available
        if str(i) in scored:
            scores[i] = scored[str(i)]
            continue

        prompt = build_prompt(text, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        score, matched_label = parse_generative_output(generated_text)
        if matched_label is None:
            parse_failures += 1

        scores[i] = score
        scored[str(i)] = score
        new_count += 1

        # Checkpoint every 50 new articles
        if new_count % 50 == 0:
            with open(checkpoint_path, 'w') as f:
                json.dump(scored, f)

        # Progress every 100 articles
        if new_count % 100 == 0 or new_count == 1:
            elapsed = time.time() - start_time
            rate = new_count / elapsed
            remaining = total - len(scored)
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(f"  [{len(scored)}/{total}] {rate:.1f} art/sec, "
                  f"ETA: {eta_min:.0f} min, parse failures: {parse_failures}")

    # Final checkpoint save
    with open(checkpoint_path, 'w') as f:
        json.dump(scored, f)

    print(f"Inference complete. Parse failures: {parse_failures}/{total} "
          f"({parse_failures/total*100:.1f}%)")

    # Clean up checkpoint on success
    if parse_failures / total < 0.1:
        os.remove(checkpoint_path)
        print("Checkpoint cleaned up (run complete)")

    return scores


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def compute_sentiment():
    news_df = pd.read_csv(NEWS_DATA_CSV)
    model_info = _MODEL_REGISTRY[SENTIMENT_MODEL]
    model_type = model_info["type"]

    print(f"Model: {SENTIMENT_MODEL}")
    print(f"Type: {model_type}")

    texts = news_df["text"].tolist()
    print(f"Articles: {len(texts)}")

    if model_type == "classifier":
        scores = run_classifier_inference(texts)
    elif model_type == "generative":
        scores = run_generative_inference(texts)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    news_df["finbert_score"] = scores

    # Aggregate: mean sentiment per (ticker, date)
    sentiment = (
        news_df.groupby(["date", "ticker"])["finbert_score"]
        .mean()
        .reset_index()
    )

    os.makedirs(os.path.dirname(SENTIMENT_CSV), exist_ok=True)
    sentiment.to_csv(SENTIMENT_CSV, index=False)
    print(f"Sentiment saved: {len(sentiment)} (ticker, date) pairs")
    print(f"Score stats: mean={sentiment['finbert_score'].mean():.4f}, "
          f"std={sentiment['finbert_score'].std():.4f}")
    print(f"Pairs per ticker:\n{sentiment['ticker'].value_counts().sort_index().to_string()}")


if __name__ == "__main__":
    compute_sentiment()

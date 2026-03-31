"""
Cross-model comparison: load results from all 5 sentiment model runs and
produce a unified comparison table + bar charts.

Key design decisions:
  - MLP and LSTM are price-only baselines — identical across all sentiment model runs.
    We load them once (from finbert reference run) and show them as shared baselines.
  - The "FinBERT-LSTM" entry in each model_comparison.csv is the LSTM+Sentiment hybrid.
    We rename it to "LSTM+{SentimentModel}" for clarity in the cross-model table.

Input:  data/output/{tag}/plots/model_comparison.csv  (produced by analysis.py per run)
Output: data/output/llm_model_comparison_summary.csv
        data/output/llm_model_comparison_mape.png
        data/output/llm_model_comparison_mae.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_BASE = PROJECT_ROOT / "data" / "output"

SENTIMENT_MODELS = [
    {"tag": "finbert_lstm_results",       "label": "FinBERT"},
    {"tag": "deberta_v3_lstm_results",    "label": "DeBERTa-v3"},
    {"tag": "llama_finsent_lstm_results", "label": "Llama-FinSent"},
    {"tag": "qwen25_lstm_results",        "label": "Qwen2.5-1.5B"},
    {"tag": "gemma_3_1b_lstm_results",    "label": "Gemma-3-1B"},
    {"tag": "llama32_1b_lstm_results",    "label": "Llama-3.2-1B"},
]

# Reference run used for shared price-only baselines (MLP, LSTM)
REFERENCE_TAG = "finbert_lstm_results"

TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_all_results():
    """
    Returns two DataFrames:
      baselines  — MLP and LSTM rows (price-only, loaded once from reference run)
      hybrid     — LSTM+Sentiment rows per sentiment model (renamed from "FinBERT-LSTM")
    """
    # Load shared baselines from reference run
    ref_csv = OUTPUT_BASE / REFERENCE_TAG / "plots" / "model_comparison.csv"
    if not ref_csv.exists():
        print(f"[ERROR] Reference run not found: {ref_csv}")
        return None, None

    ref_df = pd.read_csv(ref_csv)
    baselines = ref_df[ref_df["Model"].isin(["MLP", "LSTM"])].copy()

    # Load hybrid (LSTM+Sentiment) rows from each completed sentiment model run
    hybrid_rows = []
    missing = []

    for sm in SENTIMENT_MODELS:
        csv_path = OUTPUT_BASE / sm["tag"] / "plots" / "model_comparison.csv"
        if not csv_path.exists():
            missing.append(sm["label"])
            continue
        df = pd.read_csv(csv_path)
        # "FinBERT-LSTM" is the hybrid architecture regardless of which LLM was used
        hybrid = df[df["Model"] == "FinBERT-LSTM"].copy()
        hybrid["Model"] = f"LSTM+{sm['label']}"
        hybrid_rows.append(hybrid)

    if missing:
        print(f"[WARNING] Missing results for: {', '.join(missing)} — skipped.")

    if not hybrid_rows:
        print("No completed hybrid runs found. Run run_all_models.py first.")
        return baselines, None

    hybrid = pd.concat(hybrid_rows, ignore_index=True)
    return baselines, hybrid


# ---------------------------------------------------------------------------
# Print tables
# ---------------------------------------------------------------------------
def print_pivot(df, metric, title):
    print(f"\n{'='*70}")
    print(f"{title}")
    print("="*70)
    pivot = df.pivot_table(index="Ticker", columns="Model", values=metric, aggfunc="mean")
    # Ensure TICKERS are rows
    pivot = pivot.reindex(TICKERS)
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nMean across tickers:")
    for col in pivot.columns:
        print(f"  {col:30s}: {pivot[col].mean():.3f}")


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def plot_bar(df, metric, ylabel, title, out_path):
    models = list(df["Model"].unique())
    n_tickers = len(TICKERS)
    n_models = len(models)
    x = np.arange(n_tickers)
    width = 0.8 / n_models

    # Separate colors: grey tones for baselines, distinct colors for hybrid models
    baseline_names = {"MLP", "LSTM"}
    baseline_colors = {"MLP": "#888888", "LSTM": "#bbbbbb"}
    hybrid_colors = cm.tab10(np.linspace(0, 0.9, sum(m not in baseline_names for m in models)))

    fig, ax = plt.subplots(figsize=(15, 6))
    h_idx = 0
    for i, model_label in enumerate(models):
        vals = []
        for ticker in TICKERS:
            row = df[(df["Model"] == model_label) & (df["Ticker"] == ticker)]
            vals.append(row[metric].values[0] if len(row) else np.nan)

        color = baseline_colors.get(model_label, hybrid_colors[h_idx])
        if model_label not in baseline_names:
            h_idx += 1

        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model_label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=11)
    ax.set_xlabel("Ticker", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading results...")
    baselines, hybrid = load_all_results()

    if baselines is None:
        return

    # --- Baselines summary ---
    print_pivot(baselines, "MAPE_%", "MAPE (%) — Price-Only Baselines (shared across all runs)")
    print_pivot(baselines, "MAE",    "MAE (USD) — Price-Only Baselines (shared across all runs)")

    if hybrid is not None and len(hybrid) > 0:
        n_hybrid = hybrid["Model"].nunique()
        print(f"\n{n_hybrid} LSTM+Sentiment model(s) completed.")

        print_pivot(hybrid, "MAPE_%", "MAPE (%) — LSTM+Sentiment Hybrid (by Sentiment Model)")
        print_pivot(hybrid, "MAE",    "MAE (USD) — LSTM+Sentiment Hybrid (by Sentiment Model)")

        # Combined table: baselines + all hybrid models
        combined = pd.concat([baselines, hybrid], ignore_index=True)

        # Reorder columns: baselines first
        baseline_order = ["MLP", "LSTM"]
        hybrid_order = sorted([m for m in combined["Model"].unique() if m not in baseline_order])
        col_order = baseline_order + hybrid_order

        print(f"\n{'='*70}")
        print("Full comparison — MAPE (%) — all models")
        print("="*70)
        pivot = combined.pivot_table(index="Ticker", columns="Model", values="MAPE_%", aggfunc="mean")
        pivot = pivot.reindex(TICKERS)[col_order]
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

        print("\nMean across tickers:")
        for col in col_order:
            if col in pivot.columns:
                print(f"  {col:30s}: {pivot[col].mean():.3f}")

        # Save unified CSV
        out_csv = OUTPUT_BASE / "llm_model_comparison_summary.csv"
        combined.to_csv(out_csv, index=False)
        print(f"\nFull summary saved: {out_csv}")

        # Charts
        if n_hybrid >= 1:
            plot_bar(
                combined, "MAPE_%", "MAPE (%)",
                "MAPE (%) by Ticker — MLP / LSTM baselines vs LSTM+Sentiment models",
                OUTPUT_BASE / "llm_model_comparison_mape.png",
            )
            plot_bar(
                combined, "MAE", "MAE (USD)",
                "MAE (USD) by Ticker — MLP / LSTM baselines vs LSTM+Sentiment models",
                OUTPUT_BASE / "llm_model_comparison_mae.png",
            )
    else:
        print("\nNo hybrid runs complete yet — showing baselines only.")
        out_csv = OUTPUT_BASE / "llm_model_comparison_summary.csv"
        baselines.to_csv(out_csv, index=False)
        print(f"Baselines saved: {out_csv}")


if __name__ == "__main__":
    main()

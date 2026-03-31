"""
Step 8: Compare and visualize results from all three models.

Original: Hardcoded prediction lists for NDX.
Adapted:  Dynamically loads results CSVs, generates per-ticker plots
          and a summary comparison table.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from config import (MLP_RESULTS_CSV, LSTM_RESULTS_CSV, BERT_LSTM_RESULTS_CSV,
                    PLOTS_DIR, TICKERS)


def load_results():
    mlp = pd.read_csv(MLP_RESULTS_CSV)
    lstm = pd.read_csv(LSTM_RESULTS_CSV)
    bert = pd.read_csv(BERT_LSTM_RESULTS_CSV)
    return mlp, lstm, bert


def compute_metrics(df):
    return {
        "MAE": mean_absolute_error(df["actual"], df["predicted"]),
        "MAPE": mean_absolute_percentage_error(df["actual"], df["predicted"]),
    }


def plot_ticker(ticker, mlp, lstm, bert, save_dir):
    mlp_t = mlp[mlp["ticker"] == ticker].sort_values("date")
    lstm_t = lstm[lstm["ticker"] == ticker].sort_values("date")
    bert_t = bert[bert["ticker"] == ticker].sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(mlp_t["date"].values, mlp_t["actual"].values, linewidth=2.6,
            color="black", label="Actual price")
    ax.plot(mlp_t["date"].values, mlp_t["predicted"].values,
            color="green", label="MLP model")
    ax.plot(lstm_t["date"].values, lstm_t["predicted"].values,
            color="orange", label="LSTM model")
    ax.plot(bert_t["date"].values, bert_t["predicted"].values,
            color="red", label="FinBERT-LSTM model")

    # Show fewer x-tick labels to avoid overlap
    n_ticks = min(10, len(mlp_t))
    step = max(1, len(mlp_t) // n_ticks)
    ax.set_xticks(range(0, len(mlp_t), step))
    ax.set_xticklabels(mlp_t["date"].values[::step], rotation=45, fontsize=8)

    ax.set_xlabel("Date", fontsize=10, labelpad=10)
    ax.set_ylabel("Closing price (USD)", fontsize=10, labelpad=20)
    ax.set_title(f"{ticker} — Closing Price Predictions", fontsize=16, pad=15)
    ax.legend(loc="upper left")
    plt.tight_layout()

    path = os.path.join(save_dir, f"{ticker}_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    mlp, lstm, bert = load_results()
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Per-ticker metrics
    rows = []
    for ticker in TICKERS:
        for name, df in [("MLP", mlp), ("LSTM", lstm), ("FinBERT-LSTM", bert)]:
            t = df[df["ticker"] == ticker]
            if len(t) == 0:
                continue
            m = compute_metrics(t)
            rows.append({"Ticker": ticker, "Model": name,
                         "MAE": m["MAE"], "MAPE_%": m["MAPE"] * 100})

    summary = pd.DataFrame(rows)

    print("=" * 70)
    print("Model Comparison — Per-Ticker Results")
    print("=" * 70)

    # Pivot for readable display
    for metric in ["MAE", "MAPE_%"]:
        pivot = summary.pivot(index="Ticker", columns="Model", values=metric)
        pivot = pivot[["MLP", "LSTM", "FinBERT-LSTM"]]
        print(f"\n{metric}:")
        print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    # Aggregate
    print("\n" + "=" * 70)
    print("Aggregate (mean across tickers)")
    print("=" * 70)
    agg = summary.groupby("Model")[["MAE", "MAPE_%"]].mean()
    agg = agg.loc[["MLP", "LSTM", "FinBERT-LSTM"]]
    print(agg.to_string(float_format=lambda x: f"{x:.2f}"))

    # Generate plots
    print(f"\nGenerating per-ticker plots in {PLOTS_DIR}/")
    for ticker in TICKERS:
        path = plot_ticker(ticker, mlp, lstm, bert, PLOTS_DIR)
        print(f"  {path}")

    # Save summary CSV
    summary_path = os.path.join(PLOTS_DIR, "model_comparison.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()

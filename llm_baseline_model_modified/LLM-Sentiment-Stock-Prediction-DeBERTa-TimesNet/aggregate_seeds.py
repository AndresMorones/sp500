"""Aggregate LSTM results across multiple seeds and produce a summary report."""
import re
from pathlib import Path
from collections import defaultdict
import statistics

PROJ = Path(__file__).resolve().parent
RESULTS_DIR = PROJ / "reports" / "output" / "LSTM_price"
OUTPUT_DIR = PROJ.parents[1] / "data" / "output" / "deberta_timesnet_results"

SEEDS = [16, 32, 42, 64]
TICKERS = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
MODELS = ["deberta", "finbert", "roberta"]
TARGETS = ["Binary_Price", "Float_Price", "Delta_Price", "Factor_Price"]


def parse_result_file(fpath):
    """Extract metrics from a prediction result file."""
    import ast
    text = fpath.read_text(errors='ignore')
    metrics = {}
    m = re.search(r"Metrics:\s*(\{.+\})", text, re.DOTALL)
    if m:
        try:
            # Remove np.float64() wrappers
            raw = re.sub(r'np\.float64\(([^)]+)\)', r'\1', m.group(1))
            d = ast.literal_eval(raw)
            # Normalize keys to lowercase
            d = {k.lower(): v for k, v in d.items()}
            if 'accuracy' in d: metrics['accuracy'] = float(d['accuracy'])
            if 'auc' in d: metrics['auc'] = float(d['auc'])
            if 'mae' in d: metrics['mae'] = float(d['mae'])
            if 'rmse' in d: metrics['rmse'] = float(d['rmse'])
            if 'corr' in d: metrics['corr'] = float(d['corr'])
            if 'correlation' in d: metrics['corr'] = float(d['correlation'])
        except:
            pass
    return metrics


def main():
    # Collect all results: key = (ticker, model, target) -> list of metric dicts
    all_results = defaultdict(list)

    for seed in SEEDS:
        seed_dir = RESULTS_DIR / str(seed)
        if not seed_dir.exists():
            print(f"WARNING: seed {seed} dir not found")
            continue
        for f in seed_dir.glob("*_pred_vs_true.txt"):
            name = f.stem.replace("_pred_vs_true", "")
            # Parse: TICKER_model_target
            for ticker in TICKERS:
                if name.startswith(ticker + "_"):
                    rest = name[len(ticker)+1:]
                    for model in MODELS:
                        if rest.startswith(model + "_"):
                            target = rest[len(model)+1:]
                            metrics = parse_result_file(f)
                            metrics['seed'] = seed
                            all_results[(ticker, model, target)].append(metrics)
                            break
                    break

    # Build report
    lines = []
    lines.append("=" * 80)
    lines.append("LSTM MULTI-SEED RESULTS (seeds: 16, 32, 42, 64)")
    lines.append("=" * 80)
    lines.append("")

    # Binary Price summary
    lines.append("BINARY PRICE (Up/Down) - Mean Accuracy across seeds")
    lines.append("-" * 60)
    binary_accs = []
    for ticker in TICKERS:
        for model in MODELS:
            vals = [r['accuracy'] for r in all_results.get((ticker, model, 'Binary_Price'), []) if 'accuracy' in r]
            if vals:
                mean_acc = statistics.mean(vals)
                std_acc = statistics.stdev(vals) if len(vals) > 1 else 0
                binary_accs.append(mean_acc)
                lines.append(f"  {ticker:5s} {model:8s}: {mean_acc:.1%} +/- {std_acc:.1%}  (n={len(vals)} seeds)")
    if binary_accs:
        lines.append(f"  OVERALL MEAN: {statistics.mean(binary_accs):.1%}")
    lines.append("")

    # Float Price summary
    lines.append("FLOAT PRICE (Next-Day Close) - Mean MAE & Correlation")
    lines.append("-" * 60)
    for ticker in TICKERS:
        for model in MODELS:
            vals = all_results.get((ticker, model, 'Float_Price'), [])
            maes = [r['mae'] for r in vals if 'mae' in r]
            corrs = [r['corr'] for r in vals if 'corr' in r]
            if maes:
                lines.append(f"  {ticker:5s} {model:8s}: MAE=${statistics.mean(maes):.2f} +/- {statistics.stdev(maes) if len(maes)>1 else 0:.2f}, CORR={statistics.mean(corrs):.3f} +/- {statistics.stdev(corrs) if len(corrs)>1 else 0:.3f}  (n={len(maes)})")
    lines.append("")

    # Delta Price summary
    lines.append("DELTA PRICE (Price Change) - Mean MAE")
    lines.append("-" * 60)
    for ticker in TICKERS:
        for model in MODELS:
            vals = all_results.get((ticker, model, 'Delta_Price'), [])
            maes = [r['mae'] for r in vals if 'mae' in r]
            if maes:
                lines.append(f"  {ticker:5s} {model:8s}: MAE=${statistics.mean(maes):.2f} +/- {statistics.stdev(maes) if len(maes)>1 else 0:.2f}  (n={len(maes)})")
    lines.append("")

    # Factor Price summary
    lines.append("FACTOR PRICE (% Change) - Mean MAE")
    lines.append("-" * 60)
    for ticker in TICKERS:
        for model in MODELS:
            vals = all_results.get((ticker, model, 'Factor_Price'), [])
            maes = [r['mae'] for r in vals if 'mae' in r]
            if maes:
                mean_mae = statistics.mean(maes)
                lines.append(f"  {ticker:5s} {model:8s}: MAE={mean_mae:.4f} ({mean_mae*100:.2f}%)  (n={len(maes)})")
    lines.append("")

    # Best combinations
    lines.append("BEST COMBINATIONS (by mean across seeds)")
    lines.append("-" * 60)

    # Best binary
    best_bin = max(
        [(t, m, statistics.mean([r['accuracy'] for r in all_results.get((t, m, 'Binary_Price'), []) if 'accuracy' in r]))
         for t in TICKERS for m in MODELS
         if any('accuracy' in r for r in all_results.get((t, m, 'Binary_Price'), []))],
        key=lambda x: x[2]
    )
    lines.append(f"  Best Binary:  {best_bin[0]}/{best_bin[1]} = {best_bin[2]:.1%}")

    # Best float corr
    best_corr = max(
        [(t, m, statistics.mean([r['corr'] for r in all_results.get((t, m, 'Float_Price'), []) if 'corr' in r]))
         for t in TICKERS for m in MODELS
         if any('corr' in r for r in all_results.get((t, m, 'Float_Price'), []))],
        key=lambda x: x[2]
    )
    lines.append(f"  Best Float CORR: {best_corr[0]}/{best_corr[1]} = {best_corr[2]:.3f}")

    lines.append("")
    lines.append(f"Total result files: {sum(len(v) for v in all_results.values())}")

    report = "\n".join(lines)
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "lstm_multiseed_report.txt"
    out.write_text(report)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

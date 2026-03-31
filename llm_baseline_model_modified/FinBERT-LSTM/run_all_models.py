"""
Orchestrator: run FinBERT-LSTM pipeline for all pending sentiment models.

Strategy (pipeline parallelism for 16 GB M5 MacBook Pro):
  - LLMs run ONE AT A TIME for step 4 (sentiment inference) — each needs ~3-4 GB
  - Steps 5-7 + analysis (TF/Keras LSTM, lightweight) run in BACKGROUND
    while the next model's step 4 starts immediately after
  - GPU acceleration: MPS (Apple Silicon) auto-detected in step 4

Execution order:
  1. Llama-FinSent  → resume from checkpoint if available
  2. Qwen 2.5-1.5B
  3. Gemma 3n-E2B

Pipeline overlap:
  [Llama step4] → [Llama steps5-7 BG] + [Qwen step4] → [Qwen steps5-7 BG] + [Gemma step4] ...

Usage:
  python run_all_models.py            # run all pending models
  python run_all_models.py --dry-run  # show what would run without executing
  python run_all_models.py --model llama   # run a specific model only
"""

import os
import sys
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Model definitions — order matters (Llama first to resume checkpoint)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_BASE = PROJECT_ROOT / "data" / "output"

MODELS = [
    {
        "name": "llama",
        "model_id": "oopere/Llama-FinSent-S",
        "tag": "llama_finsent_lstm_results",
        "label": "Llama-FinSent-S",
    },
    {
        "name": "qwen",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "tag": "qwen25_lstm_results",
        "label": "Qwen2.5-1.5B",
    },
    {
        "name": "gemma",
        "model_id": "google/gemma-3-1b-it",
        "tag": "gemma_3_1b_lstm_results",
        "label": "Gemma-3-1B",
    },
    {
        "name": "llama32",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "tag": "llama32_1b_lstm_results",
        "label": "Llama-3.2-1B",
    },
]

# Reference output dir (finbert, already complete) — used to copy shared data files
REFERENCE_TAG = "finbert_lstm_results"

# Steps that run once per model (sentiment inference — LLM bottleneck)
STEP4 = "4_news_sentiment_analysis.py"

# Steps that run after sentiment — lightweight TF/Keras training (can run in parallel)
TRAINING_STEPS = [
    "5_MLP_model.py",
    "6_LSTM_model.py",
    "7_lstm_model_bert.py",
]

# Post-training step — depends on training outputs, must run after TRAINING_STEPS complete
POST_STEP = "analysis.py"

# Map each training step to its output file (for step-level skip check)
_STEP_OUTPUT = {
    "5_MLP_model.py":       "mlp_results.csv",
    "6_LSTM_model.py":      "lstm_results.csv",
    "7_lstm_model_bert.py": "bert_lstm_results.csv",
    "analysis.py":          os.path.join("plots", "model_comparison.csv"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ts():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def is_complete(tag):
    """Model run is complete if bert_lstm_results.csv AND plots/model_comparison.csv exist."""
    out = OUTPUT_BASE / tag
    return (
        (out / "bert_lstm_results.csv").exists()
        and (out / "plots" / "model_comparison.csv").exists()
    )


def has_sentiment(tag):
    """Sentiment step is done if sentiment.csv exists."""
    return (OUTPUT_BASE / tag / "sentiment.csv").exists()


def has_checkpoint(tag):
    return (OUTPUT_BASE / tag / "_sentiment_checkpoint.json").exists()


def prep_output_dir(tag):
    """Pre-populate output dir with shared data files (news + stock) from finbert reference."""
    out_dir = OUTPUT_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_dir = OUTPUT_BASE / REFERENCE_TAG
    for fname in ("news_data.csv", "stock_price.csv"):
        src = ref_dir / fname
        dst = out_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            log(f"  Copied {fname} → {tag}/")


def run_step(script, model_id, tag, background=False):
    """Run a pipeline script with FINBERT_SENTIMENT_MODEL env var set.

    Returns subprocess.Popen if background=True, else waits and returns returncode.
    """
    env = {**os.environ, "FINBERT_SENTIMENT_MODEL": model_id}
    cmd = [sys.executable, str(SCRIPT_DIR / script)]

    log_path = OUTPUT_BASE / tag / f"_log_{Path(script).stem}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if background:
        log(f"  → {script} (background, log: {log_path.name})")
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(SCRIPT_DIR),
                stdout=logf, stderr=subprocess.STDOUT
            )
        return proc
    else:
        log(f"  → {script} (foreground)")
        with open(log_path, "w") as logf:
            result = subprocess.run(
                cmd, env=env, cwd=str(SCRIPT_DIR),
                stdout=logf, stderr=subprocess.STDOUT
            )
        # Always tail the last 20 lines of output to console
        if log_path.exists():
            lines = log_path.read_text().splitlines()
            for line in lines[-20:]:
                print(f"    {line}")
        return result.returncode


def wait_for_procs(procs, labels):
    """Wait for all background processes, reporting completion."""
    for proc, label in zip(procs, labels):
        proc.wait()
        rc = proc.returncode
        status = "done" if rc == 0 else f"FAILED (rc={rc})"
        log(f"  Background [{label}] {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dry_run = "--dry-run" in sys.argv
    filter_model = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            filter_model = arg.split("=", 1)[1].lower()
        elif arg == "--model" and sys.argv.index(arg) + 1 < len(sys.argv):
            filter_model = sys.argv[sys.argv.index(arg) + 1].lower()

    log("=" * 60)
    log("FinBERT-LSTM Multi-Model Orchestrator")
    log(f"Project root: {PROJECT_ROOT}")
    log("=" * 60)

    # --- Status check ---
    log("\nModel status:")
    pending = []
    for m in MODELS:
        if filter_model and m["name"] != filter_model:
            continue
        complete = is_complete(m["tag"])
        sent_done = has_sentiment(m["tag"])
        ckpt = has_checkpoint(m["tag"])
        status = "COMPLETE" if complete else ("sentiment-done" if sent_done else ("checkpoint" if ckpt else "not started"))
        log(f"  {m['label']:25s} → {status}")
        if not complete:
            pending.append(m)

    if not pending:
        log("\nAll models complete. Running comparison...")
        subprocess.run([sys.executable, str(SCRIPT_DIR / "compare_all_models.py")],
                       cwd=str(SCRIPT_DIR))
        return

    log(f"\n{len(pending)} model(s) to run: {[m['label'] for m in pending]}")

    if dry_run:
        log("\n[DRY RUN] No changes made.")
        return

    # --- Pipeline execution ---
    background_procs = []   # (Popen, label) for in-flight LSTM training jobs
    pending_analysis = []   # (model_id, tag, label) — run analysis.py after training completes

    for i, m in enumerate(pending):
        model_id = m["model_id"]
        tag = m["tag"]
        label = m["label"]

        log(f"\n{'─'*50}")
        log(f"Model {i+1}/{len(pending)}: {label}")
        log(f"{'─'*50}")

        # Ensure output dir has shared data files
        prep_output_dir(tag)

        # Step 4: sentiment inference (foreground — LLM must run alone)
        if has_sentiment(tag):
            log(f"  Step 4: SKIPPED (sentiment.csv already exists)")
        else:
            if has_checkpoint(tag):
                log(f"  Step 4: Resuming from checkpoint...")
            else:
                log(f"  Step 4: Starting fresh sentiment inference on MPS GPU...")
            t0 = time.time()
            rc = run_step(STEP4, model_id, tag, background=False)
            elapsed = time.time() - t0
            if rc != 0:
                log(f"  Step 4 FAILED (rc={rc}). Check log: {tag}/_log_4_news_sentiment_analysis.txt")
                log("  Skipping to next model.")
                continue
            log(f"  Step 4: done in {elapsed/60:.1f} min")

        # Steps 5-7: launch in background so next model's step 4 can start
        # Each script handles its own per-ticker checkpoint — skip if output already complete
        steps_to_run = []
        for step in TRAINING_STEPS:
            out_file = _STEP_OUTPUT.get(step, "")
            if out_file and (OUTPUT_BASE / tag / out_file).exists():
                log(f"  Skipping {step} (output already exists)")
            else:
                steps_to_run.append(step)

        if steps_to_run:
            log(f"  Training steps to run in background: {steps_to_run}")
            procs_for_this_model = []
            for step in steps_to_run:
                proc = run_step(step, model_id, tag, background=True)
                procs_for_this_model.append((proc, f"{label}/{step}"))
            background_procs.extend(procs_for_this_model)
            # Track which models need analysis.py after training completes
            pending_analysis.append((model_id, tag, label))
            log(f"  {len(steps_to_run)} training job(s) running in background.")
        else:
            # Training done — check if analysis still needs to run
            post_out = _STEP_OUTPUT.get(POST_STEP, "")
            if post_out and (OUTPUT_BASE / tag / post_out).exists():
                log(f"  All steps already complete for {label}.")
            else:
                pending_analysis.append((model_id, tag, label))

    # Wait for all background training jobs
    if background_procs:
        log(f"\nWaiting for {len(background_procs)} background training job(s)...")
        for proc, label in background_procs:
            proc.wait()
            rc = proc.returncode
            status = "done" if rc == 0 else f"FAILED (rc={rc})"
            log(f"  [{label}] {status}")

    # Run analysis.py for each model (depends on training outputs, must run after)
    if pending_analysis:
        log(f"\nRunning analysis for {len(pending_analysis)} model(s)...")
        for model_id, tag, label in pending_analysis:
            post_out = _STEP_OUTPUT.get(POST_STEP, "")
            if post_out and (OUTPUT_BASE / tag / post_out).exists():
                log(f"  [{label}] analysis already complete — skipping")
                continue
            rc = run_step(POST_STEP, model_id, tag, background=False)
            status = "done" if rc == 0 else f"FAILED (rc={rc})"
            log(f"  [{label}/analysis.py] {status}")

    log("\n" + "=" * 60)
    log("All models done. Running cross-model comparison...")
    log("=" * 60)
    subprocess.run([sys.executable, str(SCRIPT_DIR / "compare_all_models.py")],
                   cwd=str(SCRIPT_DIR))
    log("Complete.")


if __name__ == "__main__":
    main()

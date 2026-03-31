"""
collect_metrics.py — SafeGuard AI Evaluation Metrics Collector
==============================================================
Scans every model directory (HUMAN, NEW PPE, NEW TOOLS, PPE, TOOLS)
for completed training runs, then copies the key metric files and
performance charts into a single EVALUATION METRICS/ folder for easy
reference when writing the project report.

What gets collected per model run:
  - results.csv              — raw per-epoch training metrics
  - results.png              — YOLO training dashboard image
  - confusion_matrix*.png    — class confusion matrices
  - BoxF1_curve.png          — F1 score vs confidence
  - BoxPR_curve.png          — Precision-Recall curve
  - weights/best.pt          — the best trained model checkpoint

Also generates:
  - SUMMARY_ALL_MODELS.csv   — one-row-per-model comparison table
  - comparison_chart.png     — bar chart of mAP50 / Precision / Recall
  - <MODEL>/learning_curve.png  — per-model mAP50 curve across epochs
  - README.md                — human-readable summary with metric definitions

Run from the project root after all training is complete:
    python TRAINING/collect_metrics.py
"""

import os
import shutil
import csv
import re
from pathlib import Path
from datetime import datetime

# Project root and output destination
PROJECT_ROOT  = Path(r"E:\4TH YEAR PROJECT")
EVAL_OUT_DIR  = PROJECT_ROOT / "EVALUATION METRICS"
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M")

# Map of human-friendly model names to their dataset directories
MODEL_DIRS = {
    "HUMAN"    : PROJECT_ROOT / "HUMAN",
    "NEW_PPE"  : PROJECT_ROOT / "NEW PPE",
    "NEW_TOOLS": PROJECT_ROOT / "NEW TOOLS",
    "PPE"      : PROJECT_ROOT / "PPE",
    "TOOLS"    : PROJECT_ROOT / "TOOLS",
}

# Files to copy from each run directory into EVALUATION METRICS
METRIC_FILES_TO_COPY = [
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxF1_curve.png",
    "F1_curve.png",
    "BoxPR_curve.png",
    "PR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "P_curve.png",
    "R_curve.png",
    "args.yaml",
    "val_batch0_pred.jpg",
    "val_batch0_labels.jpg",
]

# Ensure the output directory exists before we start copying files
EVAL_OUT_DIR.mkdir(exist_ok=True)

print(f"\n{'=' * 60}")
print( "  SafeGuard AI — Evaluation Metrics Collector")
print(f"  Generated: {RUN_TIMESTAMP}")
print(f"{'=' * 60}\n")


def find_training_run_dirs(model_dir: Path):
    """
    Recursively search a model directory for completed training runs.

    A directory is considered a completed run if it contains a
    results.csv file (YOLO writes this after every epoch).

    Args:
        model_dir (Path): Root directory of a model (e.g., PROJECT_ROOT / "HUMAN")

    Returns:
        list[Path]: List of run directories, each containing results.csv
    """
    return [csv_file.parent for csv_file in model_dir.rglob("results.csv")]


def parse_best_epoch_metrics(results_csv_path: Path):
    """
    Parse a YOLO results.csv and return the metrics from the epoch
    that achieved the highest mAP@50.

    Args:
        results_csv_path (Path): Path to the results.csv file

    Returns:
        dict: Cleaned metric values (epochs, best_mAP50, precision, recall, fitness)
              All values are strings; empty string means the column was not found.
    """
    metrics = {
        "epochs"     : "",
        "best_mAP50" : "",
        "best_mAP50-95": "",
        "precision"  : "",
        "recall"     : "",
        "fitness"    : "",
    }

    try:
        with open(results_csv_path, newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = [{key.strip(): val.strip() for key, val in row.items()} for row in reader]

        if not rows:
            return metrics

        metrics["epochs"] = str(len(rows))

        # Identify which columns hold which metric (column names differ across YOLO versions)
        col_map50    = next((k for k in rows[0] if "map50" in k.lower() and "95" not in k.lower()), None)
        col_map5095  = next((k for k in rows[0] if "map50-95" in k.lower()), None)
        col_precision= next((k for k in rows[0] if "precision" in k.lower()), None)
        col_recall   = next((k for k in rows[0] if "recall" in k.lower()), None)
        col_fitness  = next((k for k in rows[0] if "fitness" in k.lower()), None)

        if col_map50:
            best_row = max(rows, key=lambda r: float(r.get(col_map50, 0) or 0))
            metrics["best_mAP50"]    = f"{float(best_row.get(col_map50, 0)):.4f}"
            if col_map5095:
                metrics["best_mAP50-95"] = f"{float(best_row.get(col_map5095, 0)):.4f}"
            if col_precision:
                metrics["precision"] = f"{float(best_row.get(col_precision, 0)):.4f}"
            if col_recall:
                metrics["recall"]    = f"{float(best_row.get(col_recall, 0)):.4f}"
            if col_fitness:
                metrics["fitness"]   = f"{float(best_row.get(col_fitness, 0)):.4f}"

    except Exception as parse_error:
        print(f"     Warning: Could not parse results.csv — {parse_error}")

    return metrics


# Track how many models were successfully collected vs not yet trained
collected_count = 0
skipped_count   = 0
summary_rows    = []

# Process each model directory
for model_name, model_dir in MODEL_DIRS.items():
    if not model_dir.exists():
        print(f"  Warning  {model_name}: directory not found — skipping")
        continue

    run_dirs = find_training_run_dirs(model_dir)

    if not run_dirs:
        print(f"  [ ] {model_name}: no completed training run found yet")
        skipped_count += 1
        summary_rows.append({
            "model"         : model_name,
            "run_path"      : "N/A",
            "status"        : "NOT TRAINED",
            "epochs"        : "",
            "best_mAP50"    : "",
            "best_mAP50-95" : "",
            "precision"     : "",
            "recall"        : "",
            "fitness"       : "",
            "collected_files": "0",
        })
        continue

    for run_dir in run_dirs:
        # Each run gets its own subfolder inside EVALUATION METRICS/
        dest_dir = EVAL_OUT_DIR / model_name / run_dir.name
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [+] {model_name} — run: {run_dir.name}")

        # Copy all available metric files
        copied_files = []
        for filename in METRIC_FILES_TO_COPY:
            source_file = run_dir / filename
            if source_file.exists():
                shutil.copy2(source_file, dest_dir / filename)
                copied_files.append(filename)
                print(f"     Copied: {filename}")
            else:
                # Some older YOLO versions omit the "Box" prefix
                alt_name = filename.replace("Box", "")
                alt_source = run_dir / alt_name
                if alt_name != filename and alt_source.exists():
                    shutil.copy2(alt_source, dest_dir / alt_name)
                    copied_files.append(alt_name)
                    print(f"     Copied: {alt_name} (alternative name)")

        # Always copy the best model checkpoint
        best_pt_src = run_dir / "weights" / "best.pt"
        if best_pt_src.exists():
            weights_dest = dest_dir / "weights"
            weights_dest.mkdir(exist_ok=True)
            shutil.copy2(best_pt_src, weights_dest / "best.pt")
            copied_files.append("weights/best.pt")
            print(f"     Copied: weights/best.pt")

        # Parse metrics from results.csv for the summary table
        epoch_metrics = parse_best_epoch_metrics(run_dir / "results.csv")
        print(
            f"     Metrics — Epochs: {epoch_metrics['epochs']} | "
            f"Best mAP@50: {epoch_metrics['best_mAP50']} | "
            f"Precision: {epoch_metrics['precision']} | "
            f"Recall: {epoch_metrics['recall']}"
        )

        summary_rows.append({
            "model"          : model_name,
            "run_path"       : str(run_dir),
            "status"         : "TRAINED",
            "epochs"         : epoch_metrics["epochs"],
            "best_mAP50"     : epoch_metrics["best_mAP50"],
            "best_mAP50-95"  : epoch_metrics["best_mAP50-95"],
            "precision"      : epoch_metrics["precision"],
            "recall"         : epoch_metrics["recall"],
            "fitness"        : epoch_metrics["fitness"],
            "collected_files": str(len(copied_files)),
        })
        collected_count += 1

# Write the combined summary CSV
summary_csv_path = EVAL_OUT_DIR / "SUMMARY_ALL_MODELS.csv"
csv_columns = [
    "model", "run_path", "status", "epochs",
    "best_mAP50", "best_mAP50-95", "precision", "recall", "fitness",
    "collected_files"
]
with open(summary_csv_path, "w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    writer.writeheader()
    writer.writerows(summary_rows)

print(f"\n{'=' * 60}")
print( "  COLLECTION SUMMARY")
print(f"  Models with results : {collected_count}")
print(f"  Models not trained  : {skipped_count}")
print(f"  Summary CSV         : {summary_csv_path}")
print(f"  Output folder       : {EVAL_OUT_DIR}")
print(f"{'=' * 60}\n")


# Generate comparison charts using matplotlib if available
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    trained_models = [r for r in summary_rows if r["status"] == "TRAINED" and r["best_mAP50"]]

    if trained_models:
        model_names    = [r["model"]       for r in trained_models]
        map50_values   = [float(r["best_mAP50"])  for r in trained_models]
        precision_vals = [float(r["precision"])    for r in trained_models]
        recall_vals    = [float(r["recall"])       for r in trained_models]

        # Colour palette — each model gets a distinct colour
        bar_colors = ["#00f5ff", "#7c3aed", "#f59e0b", "#ef4444", "#10b981"][:len(model_names)]

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.patch.set_facecolor("#0f172a")

        for ax, values, chart_title, y_label in zip(
            axes,
            [map50_values, precision_vals, recall_vals],
            ["Best mAP@50", "Precision", "Recall"],
            ["mAP@50",      "Precision", "Recall"]
        ):
            bars = ax.bar(model_names, values, color=bar_colors, width=0.55, edgecolor="none")
            ax.set_facecolor("#1e293b")
            ax.set_title(chart_title, color="#00f5ff", fontsize=13, fontweight="bold", pad=12)
            ax.set_ylabel(y_label, color="#94a3b8", fontsize=10)
            ax.set_ylim(0, 1)
            ax.tick_params(colors="#94a3b8", labelsize=9)
            ax.spines[:].set_color("#334155")
            ax.yaxis.set_tick_params(labelcolor="#94a3b8")

            # Add value label above each bar
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    color="white", fontsize=9, fontweight="bold"
                )

        plt.suptitle(
            "SafeGuard AI — Model Performance Comparison",
            color="white", fontsize=15, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        chart_output = EVAL_OUT_DIR / "comparison_chart.png"
        plt.savefig(chart_output, dpi=150, bbox_inches="tight",
                    facecolor="#0f172a", edgecolor="none")
        plt.close()
        print(f"  Chart saved: {chart_output}")

        # Per-model learning curves (mAP@50 vs epoch)
        for model_row in trained_models:
            run_results_csv = Path(model_row["run_path"]) / "results.csv"
            if not run_results_csv.exists():
                continue
            try:
                epoch_numbers, epoch_map50 = [], []
                with open(run_results_csv, newline="", encoding="utf-8") as f:
                    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]

                col_epoch = next((k for k in rows[0] if "epoch" in k.lower()), None)
                col_map50 = next((k for k in rows[0] if "map50" in k.lower() and "95" not in k.lower()), None)

                if col_epoch and col_map50:
                    for row in rows:
                        try:
                            epoch_numbers.append(int(float(row[col_epoch])))
                            epoch_map50.append(float(row.get(col_map50, 0)))
                        except (ValueError, TypeError):
                            pass

                if epoch_numbers:
                    fig2, ax2 = plt.subplots(figsize=(10, 5))
                    fig2.patch.set_facecolor("#0f172a")
                    ax2.set_facecolor("#1e293b")
                    ax2.plot(epoch_numbers, epoch_map50, color="#00f5ff", linewidth=2)
                    ax2.fill_between(epoch_numbers, epoch_map50, alpha=0.15, color="#00f5ff")

                    best_idx = int(np.argmax(epoch_map50))
                    ax2.scatter(
                        [epoch_numbers[best_idx]], [epoch_map50[best_idx]],
                        color="#f59e0b", s=80, zorder=5,
                        label=f"Best: {epoch_map50[best_idx]:.4f}"
                    )
                    ax2.set_title(
                        f"{model_row['model']} — mAP@50 Learning Curve",
                        color="#00f5ff", fontsize=13, fontweight="bold"
                    )
                    ax2.set_xlabel("Epoch", color="#94a3b8")
                    ax2.set_ylabel("mAP@50", color="#94a3b8")
                    ax2.tick_params(colors="#94a3b8")
                    ax2.spines[:].set_color("#334155")
                    ax2.legend(facecolor="#1e293b", edgecolor="#334155",
                               labelcolor="white", fontsize=9)
                    plt.tight_layout()

                    curve_path = EVAL_OUT_DIR / model_row["model"] / "learning_curve.png"
                    plt.savefig(curve_path, dpi=150, bbox_inches="tight",
                                facecolor="#0f172a", edgecolor="none")
                    plt.close()
                    print(f"  Learning curve saved: {curve_path}")

            except Exception as curve_error:
                print(f"  Warning: Could not plot learning curve for {model_row['model']}: {curve_error}")

except ImportError:
    print("  Note: matplotlib not installed — skipping charts (pip install matplotlib)")


# Write the README summary for the EVALUATION METRICS folder
readme_content = f"""# SafeGuard AI — Evaluation Metrics
Generated: {RUN_TIMESTAMP}

## Contents
- `SUMMARY_ALL_MODELS.csv` — Combined metrics table for all trained models
- `comparison_chart.png`   — Bar chart: mAP50 / Precision / Recall across all models
- `<MODEL_NAME>/`          — Per-model subfolder containing:
  - `results.csv`                        — Epoch-by-epoch training data
  - `results.png`                        — YOLO training dashboard image
  - `confusion_matrix.png`               — Class confusion matrix
  - `confusion_matrix_normalized.png`    — Normalised confusion matrix
  - `BoxF1_curve.png`, `BoxPR_curve.png` — Precision-Recall / F1 curves
  - `weights/best.pt`                    — Best model checkpoint
  - `learning_curve.png`                 — Custom mAP@50 per-epoch chart

## Model Results
| Model | Status | Epochs | Best mAP@50 | Precision | Recall |
|-------|--------|--------|-------------|-----------|--------|
"""

for row in summary_rows:
    readme_content += (
        f"| {row['model']} | {row['status']} | {row['epochs']} "
        f"| {row['best_mAP50']} | {row['precision']} | {row['recall']} |\n"
    )

readme_content += (
    "\n## How to Refresh\n"
    "```\n"
    "python TRAINING/collect_metrics.py\n"
    "```\n"
    "Run from `E:\\4TH YEAR PROJECT\\` after completing a new training run.\n"
)

(EVAL_OUT_DIR / "README.md").write_text(readme_content, encoding="utf-8")
print("  README.md written to EVALUATION METRICS/")
print(f"\n  Done — all metrics collected to: {EVAL_OUT_DIR}\n")

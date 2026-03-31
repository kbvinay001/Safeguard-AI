"""
generate_eval_metrics.py — Per-Model Evaluation Report Generator
================================================================
Reads the results.csv training logs for the HUMAN, TOOLS, and PPE
models and produces:

  1. A detailed human-readable evaluation report (.txt) per model
  2. A JSON summary file per model (machine-readable metrics)
  3. Copies of all key training plots (F1 curve, PR curve, confusion
     matrices, training dashboard) into EVALUATION METRICS/<model>/
  4. An updated SUMMARY_ALL_MODELS.csv combining metrics for all models
  5. A README.md with metric definitions and the summary table

Run from the project root after all three models have been trained:
    python TRAINING/generate_eval_metrics.py
"""

import csv
import json
import shutil
from pathlib import Path

# Project root and output base directory
PROJECT_ROOT = Path(r"E:\4TH YEAR PROJECT")
EVAL_BASE    = PROJECT_ROOT / "EVALUATION METRICS"

# Model configurations — each entry describes one trained model
MODEL_CONFIGS = {
    "HUMAN": {
        "csv"     : PROJECT_ROOT / "HUMAN"     / "runs" / "detect" / "train_fast"    / "results.csv",
        "plots"   : PROJECT_ROOT / "HUMAN"     / "runs" / "detect" / "train_fast",
        "epochs"  : 75,
        "classes" : ["person"],
        "dataset" : "15,357 industrial CCTV images (CCTV-Person + CrowdHuman combined)",
        "arch"    : "YOLOv11n (nano)",
        "device"  : "NVIDIA RTX 4060 · CUDA FP16",
        "task"    : "Person / worker detection in construction and industrial environments",
        "best_pt" : PROJECT_ROOT / "HUMAN"     / "runs" / "detect" / "train_fast"    / "weights" / "best.pt",
    },
    "TOOLS": {
        "csv"     : PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_fast"    / "results.csv",
        "plots"   : PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_fast",
        "epochs"  : 100,
        "classes" : ["drill", "hammer", "pliers", "screwdriver", "wrench"],
        "dataset" : "6,535 multi-class tool images from Roboflow (Mechanical-10000)",
        "arch"    : "YOLOv11n (nano)",
        "device"  : "NVIDIA RTX 4060 · CUDA FP16",
        "task"    : "Abandoned / misplaced industrial tool detection with temporal FSM alerts",
        "best_pt" : PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_fast"    / "weights" / "best.pt",
    },
    "PPE": {
        "csv"     : PROJECT_ROOT / "NEW PPE"   / "runs" / "detect" / "train_v2_nano" / "results.csv",
        "plots"   : PROJECT_ROOT / "NEW PPE"   / "runs" / "detect" / "train_v2_nano",
        "epochs"  : 200,
        "classes" : [
            "barehead", "helmet", "barechest", "highvis",
            "barearms", "sleeves", "barehands", "gloves",
            "barelegs", "pants", "boots"
        ],
        "dataset" : "Custom Industrial PPE Dataset (11 merged classes, Roboflow-annotated)",
        "arch"    : "YOLOv11n (nano)",
        "device"  : "NVIDIA RTX 4060 · CUDA FP16",
        "task"    : "Personal Protective Equipment compliance detection (11 body-region classes)",
        "best_pt" : PROJECT_ROOT / "NEW PPE"   / "runs" / "detect" / "train_v2_nano" / "weights" / "best.pt",
    },
}

# Chart images to copy from each model's run directory
PLOT_FILES_TO_COPY = [
    "results.png",
    "BoxF1_curve.png",
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
]


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def load_results_csv(csv_path: Path):
    """
    Load a YOLO results.csv and parse every numeric cell to float.

    Non-numeric or empty cells are stored as 0.0.

    Args:
        csv_path (Path): Path to the results.csv file

    Returns:
        list[dict]: List of row dictionaries with float values
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            cleaned = {}
            for key, val in raw_row.items():
                stripped = val.strip()
                try:
                    cleaned[key.strip()] = float(stripped) if stripped else 0.0
                except ValueError:
                    cleaned[key.strip()] = 0.0
            rows.append(cleaned)
    return rows


def find_best_epoch_row(rows: list, metric_column="metrics/mAP50(B)"):
    """Return the row (dict) where the specified metric column reaches its maximum value."""
    return max(rows, key=lambda row: row.get(metric_column, 0.0))


def assign_grade(map50_value: float, thresholds=(0.85, 0.75, 0.65)):
    """
    Assign a star grade based on the mAP@50 value.

    Thresholds can be overridden; defaults are calibrated for
    industrial detection tasks.
    """
    if   map50_value >= thresholds[0]: return "★★★  EXCELLENT"
    elif map50_value >= thresholds[1]: return "★★☆  GOOD"
    elif map50_value >= thresholds[2]: return "★☆☆  ACCEPTABLE"
    else:                              return "☆☆☆  NEEDS IMPROVEMENT"


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_model_report(model_name: str, config: dict, rows: list, output_dir: Path):
    """
    Write a detailed evaluation report for one trained model.

    Args:
        model_name (str):     E.g. "HUMAN", "TOOLS", or "PPE"
        config     (dict):    Model configuration entry from MODEL_CONFIGS
        rows       (list):    Parsed results.csv rows (list of dicts)
        output_dir (Path):    Folder where the report text file will be saved

    Returns:
        tuple: (report_file_path Path, metrics_summary dict)
    """
    total_epochs = len(rows)
    last_row     = rows[-1]
    best_row     = find_best_epoch_row(rows)
    best_epoch   = int(best_row.get("epoch", 0))

    map50_best  = best_row.get("metrics/mAP50(B)",    0.0)
    map95_best  = best_row.get("metrics/mAP50-95(B)", 0.0)
    prec_best   = best_row.get("metrics/precision(B)", 0.0)
    rec_best    = best_row.get("metrics/recall(B)",    0.0)
    f1_best     = 2 * prec_best * rec_best / max(prec_best + rec_best, 1e-6)

    # Training time is stored in seconds in the last row's "time" column
    train_hours = last_row.get("time", 0.0) / 3600

    # Snapshot rows for start / mid / end loss comparison
    start_row = rows[0]
    mid_row   = rows[total_epochs // 2]
    end_row   = rows[-1]

    lines = []

    def section(title):
        lines.append("\n" + "=" * 70)
        lines.append(f"  {title}")
        lines.append("=" * 70)

    def key_value(label, value):
        lines.append(f"  {label:<36} {value}")

    lines.append(f"SAFEGUARD AI — {model_name} MODEL EVALUATION REPORT")
    lines.append("Auto-generated from YOLO training results.csv")
    lines.append("-" * 70)

    section("1. MODEL OVERVIEW")
    key_value("Model Name",          f"SafeGuard-{model_name}")
    key_value("Architecture",        config["arch"])
    key_value("Task",                config["task"])
    key_value("Training Dataset",    config["dataset"])
    key_value("Detected Classes",    ", ".join(config["classes"]))
    key_value("Training Device",     config["device"])
    key_value("Epochs Run",          f"{total_epochs} / {config['epochs']}")
    key_value("Total Training Time", f"{train_hours:.2f} hours ({train_hours * 60:.0f} minutes)")
    key_value("Best Weights Path",   str(config["best_pt"]))

    section("2. PEAK PERFORMANCE  (Best Checkpoint)")
    key_value("Best Epoch",              f"Epoch {best_epoch}")
    key_value("mAP @ IoU 0.50",         f"{map50_best * 100:.2f}%   {assign_grade(map50_best)}")
    key_value("mAP @ IoU 0.50:0.95",    f"{map95_best * 100:.2f}%   {assign_grade(map95_best, (0.70, 0.55, 0.40))}")
    key_value("Precision",              f"{prec_best * 100:.2f}%")
    key_value("Recall",                 f"{rec_best  * 100:.2f}%")
    key_value("F1-Score",               f"{f1_best   * 100:.2f}%")

    section("3. TRAINING LOSS PROGRESSION")
    lines.append(f"  {'Epoch':<16} {'Box Loss':<14} {'Cls Loss':<14} {'DFL Loss':<14}")
    lines.append("  " + "-" * 54)
    for snapshot_row, label in [(start_row, "Start"), (mid_row, "Mid"), (end_row, "Final")]:
        ep = int(snapshot_row.get("epoch", 0))
        lines.append(
            f"  {label + f' (ep {ep})':<16}"
            f"{snapshot_row.get('train/box_loss', 0):<14.4f}"
            f"{snapshot_row.get('train/cls_loss', 0):<14.4f}"
            f"{snapshot_row.get('train/dfl_loss', 0):<14.4f}"
        )

    section("4. VALIDATION LOSS PROGRESSION")
    lines.append(f"  {'Epoch':<16} {'Val Box':<14} {'Val Cls':<14} {'Val DFL':<14}")
    lines.append("  " + "-" * 54)
    for snapshot_row, label in [(start_row, "Start"), (mid_row, "Mid"), (end_row, "Final")]:
        ep = int(snapshot_row.get("epoch", 0))
        lines.append(
            f"  {label + f' (ep {ep})':<16}"
            f"{snapshot_row.get('val/box_loss', 0):<14.4f}"
            f"{snapshot_row.get('val/cls_loss', 0):<14.4f}"
            f"{snapshot_row.get('val/dfl_loss', 0):<14.4f}"
        )

    section("5. mAP@50 MILESTONE TABLE  (every 10 epochs)")
    lines.append(f"  {'Epoch':<8} {'mAP@50':<12} {'mAP@50:95':<14} {'Precision':<12} {'Recall':<10} F1")
    lines.append("  " + "-" * 65)
    for row in rows:
        ep = int(row.get("epoch", 0))
        if ep % 10 == 0 or ep == 1 or ep == total_epochs:
            p = row.get("metrics/precision(B)", 0.0)
            r = row.get("metrics/recall(B)",    0.0)
            f = 2 * p * r / max(p + r, 1e-6)
            lines.append(
                f"  {ep:<8}"
                f"{row.get('metrics/mAP50(B)',    0) * 100:<12.2f}"
                f"{row.get('metrics/mAP50-95(B)', 0) * 100:<14.2f}"
                f"{p * 100:<12.2f}"
                f"{r * 100:<10.2f}"
                f"{f * 100:.2f}"
            )

    section("6. AVAILABLE EVALUATION PLOTS")
    lines.append("")
    for plot_file in PLOT_FILES_TO_COPY:
        source = config["plots"] / plot_file
        status = "Yes" if source.exists() else "Not found"
        lines.append(f"  {status:<10}  {plot_file}")

    section("7. INTERPRETATION & INSIGHTS")
    lines.append("")
    improvement = (map50_best - rows[0].get("metrics/mAP50(B)", 0.0)) * 100
    lines.append(f"  Training improved mAP@50 by {improvement:.1f} percentage points over {total_epochs} epochs.")
    lines.append("")

    if   map50_best >= 0.85: lines.append("  Result: EXCELLENT detection accuracy. Model is ready for production deployment.")
    elif map50_best >= 0.75: lines.append("  Result: GOOD accuracy. Suitable for deployment with periodic monitoring.")
    elif map50_best >= 0.65: lines.append("  Result: ACCEPTABLE accuracy. Consider more training data or additional epochs.")
    else:                    lines.append("  Result: Below target. More training data or a larger model architecture is recommended.")

    lines.append("")
    lines.append(f"  Precision {prec_best * 100:.1f}% means {prec_best * 100:.1f}% of all detections made are correct.")
    lines.append(f"  Recall {rec_best * 100:.1f}% means {rec_best * 100:.1f}% of real objects in the scene were found.")
    lines.append(f"  F1-Score {f1_best * 100:.1f}% is the harmonic mean of precision and recall.")

    box_drop = start_row.get("train/box_loss", 1) - end_row.get("train/box_loss", 1)
    cls_drop = start_row.get("train/cls_loss", 1) - end_row.get("train/cls_loss", 1)
    lines.append(f"  Box loss dropped by {box_drop:.4f}  |  Classification loss dropped by {cls_drop:.4f}")
    lines.append("")
    lines.append("  Note: All images were annotated via Roboflow. Training used YOLOv11n with")
    lines.append("        CUDA FP16 acceleration on the NVIDIA RTX 4060.")

    section("8. DEPLOYMENT RECOMMENDATION")
    lines.append("")
    lines.append(f"  Load weights : {config['best_pt']}")
    lines.append(f"  Confidence   : 0.45  (recommended for noisy industrial scenes)")
    lines.append(f"  IoU threshold: 0.45")
    lines.append(f"  Input size   : 640 × 640 pixels")
    lines.append(f"  Inference    : FP16 on CUDA for maximum throughput")
    lines.append("")
    if model_name == "TOOLS":
        lines.append("  Tool abandonment alert thresholds (Finite State Machine):")
        lines.append("    - WARNING  : tool stationary and unattended for 25 seconds")
        lines.append("    - CRITICAL : tool stationary and unattended for 35 seconds")
    elif model_name == "HUMAN":
        lines.append("  Used alongside the PPE model to pair detected workers with their safety gear status.")

    lines.append("\n" + "=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_path = output_dir / f"{model_name}_EVALUATION_REPORT.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    metrics_summary = {
        "model"        : f"SafeGuard-{model_name}",
        "epochs"       : total_epochs,
        "best_epoch"   : best_epoch,
        "mAP50_pct"    : round(map50_best * 100, 2),
        "mAP50_95_pct" : round(map95_best * 100, 2),
        "precision_pct": round(prec_best  * 100, 2),
        "recall_pct"   : round(rec_best   * 100, 2),
        "f1_pct"       : round(f1_best    * 100, 2),
        "train_hours"  : round(train_hours, 2),
        "grade"        : assign_grade(map50_best),
    }
    return report_path, metrics_summary


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    all_summaries = []

    for model_name, config in MODEL_CONFIGS.items():
        output_dir = EVAL_BASE / model_name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Processing {model_name} model...")

        # 1. Copy the results CSV file
        dest_csv = output_dir / "results.csv"
        shutil.copy2(config["csv"], dest_csv)
        print(f"  Copied results.csv  →  {dest_csv}")

        # 2. Copy training plots
        for plot_file in PLOT_FILES_TO_COPY:
            source = config["plots"] / plot_file
            if source.exists():
                shutil.copy2(source, output_dir / plot_file)
                print(f"  Copied {plot_file}")
            else:
                print(f"  Not found: {plot_file}")

        # 3. Parse results and write the human-readable report
        rows = load_results_csv(config["csv"])
        report_path, summary = build_model_report(model_name, config, rows, output_dir)
        print(f"  Report saved  →  {report_path}")

        # 4. Write the per-model JSON summary
        json_path = output_dir / f"{model_name}_metrics_summary.json"
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  JSON saved    →  {json_path}")

        all_summaries.append(summary)

    # 5. Update the top-level SUMMARY_ALL_MODELS.csv
    summary_csv_path = EVAL_BASE / "SUMMARY_ALL_MODELS.csv"
    csv_columns = [
        "model", "epochs", "best_epoch", "mAP50_pct", "mAP50_95_pct",
        "precision_pct", "recall_pct", "f1_pct", "train_hours", "grade"
    ]

    # Merge with any existing rows from other models not in this run
    existing_rows = []
    if summary_csv_path.exists():
        with open(summary_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = [
                row for row in reader
                if row.get("model", "") not in [s["model"] for s in all_summaries]
            ]

    all_rows = existing_rows + all_summaries
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  SUMMARY_ALL_MODELS.csv updated  →  {summary_csv_path}")

    # 6. Write the master README
    readme_content = """# SafeGuard AI — Evaluation Metrics

This folder contains training evaluation results for all three YOLOv11n models
used in the SafeGuard AI industrial safety detection system.

## Model Summary

| Model | Epochs | Best mAP@50 | F1 | Grade | Training Time |
|-------|--------|------------|-----|-------|---------------|
"""
    for row in all_rows:
        readme_content += (
            f"| {row['model']} | {row['epochs']} | {row['mAP50_pct']}% "
            f"| {row['f1_pct']}% | {row['grade']} | {row['train_hours']} hrs |\n"
        )

    readme_content += """
## Folder Structure

```
EVALUATION METRICS/
├── HUMAN/
│   ├── HUMAN_EVALUATION_REPORT.txt       — Full human-readable evaluation report
│   ├── HUMAN_metrics_summary.json        — Machine-readable metric summary
│   ├── results.csv                       — Raw epoch-by-epoch training data
│   ├── results.png                       — YOLO training dashboard image
│   ├── BoxF1_curve.png                   — F1 score vs confidence threshold
│   ├── BoxPR_curve.png                   — Precision-Recall curve
│   ├── BoxP_curve.png                    — Precision vs confidence
│   ├── BoxR_curve.png                    — Recall vs confidence
│   ├── confusion_matrix.png              — Raw confusion matrix
│   └── confusion_matrix_normalized.png   — Row-normalised confusion matrix
├── TOOLS/                                — Same structure as HUMAN
├── PPE/                                  — Same structure as HUMAN
├── SUMMARY_ALL_MODELS.csv                — Cross-model comparison table
└── README.md                             — This file
```

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| **mAP@50** | Mean Average Precision at IoU threshold 0.50 — primary YOLO metric |
| **mAP@50:95** | Stricter COCO-style mAP averaged across IoU thresholds 0.50–0.95 |
| **Precision** | TP / (TP + FP) — fraction of detections that are correct |
| **Recall** | TP / (TP + FN) — fraction of real objects that were detected |
| **F1** | Harmonic mean of Precision and Recall |
| **Box Loss** | Bounding-box regression loss (lower is better) |
| **Cls Loss** | Classification loss (lower is better) |
| **DFL Loss** | Distribution Focal Loss for precise box localisation |

## Grading Scale

| Grade | mAP@50 |
|-------|--------|
| ★★★ EXCELLENT | ≥ 85% |
| ★★☆ GOOD | ≥ 75% |
| ★☆☆ ACCEPTABLE | ≥ 65% |
| ☆☆☆ NEEDS IMPROVEMENT | < 65% |
"""
    (EVAL_BASE / "README.md").write_text(readme_content, encoding="utf-8")
    print(f"  README.md updated  →  {EVAL_BASE / 'README.md'}")

    print(f"\n{'=' * 60}")
    print("  ALL DONE. Files written to:")
    print(f"  {EVAL_BASE}")
    print()
    for row in all_summaries:
        print(f"  {row['model']:12s}  mAP@50={row['mAP50_pct']}%  F1={row['f1_pct']}%  {row['grade']}")
    print()


if __name__ == "__main__":
    main()

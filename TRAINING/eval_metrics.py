"""
eval_metrics.py — SafeGuard AI Training Evaluation Report Generator
====================================================================
Reads the results.csv files produced by YOLO training runs and
generates a human-readable summary report covering:
  - Best epoch, mAP@50, mAP@50:95, Precision, Recall, F1
  - Loss progression (box loss, classification loss) from start to end
  - ASCII sparkline of mAP@50 across all epochs
  - Grade based on mAP thresholds
  - Side-by-side comparison table of all trained models

Run this script after training completes:
    python TRAINING/eval_metrics.py

Output:
    E:\\4TH YEAR PROJECT\\training_evaluation_report.txt
    (also printed to the terminal)
"""

import csv
import math
from pathlib import Path
from datetime import datetime

# Project root — all CSV paths are relative to this
PROJECT_ROOT = Path(r"E:\4TH YEAR PROJECT")

# Paths to results.csv for each model's training run
RESULTS_CSV_PATHS = {
    "HUMAN": PROJECT_ROOT / "HUMAN"     / "runs" / "detect" / "train_fast"    / "results.csv",
    "TOOLS": PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_fast"    / "results.csv",
}

# PPE uses a separate run name from the sequential pipeline
PPE_RESULTS_CSV = PROJECT_ROOT / "NEW PPE" / "runs" / "detect" / "train_v2_nano" / "results.csv"

# Where to write the final report text file
REPORT_OUTPUT_FILE = PROJECT_ROOT / "training_evaluation_report.txt"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def read_results_csv(csv_path: Path):
    """
    Load a YOLO results.csv and return a list of row dicts with stripped keys.

    Args:
        csv_path (Path): Path to the results.csv file

    Returns:
        tuple: (list[dict] rows, None) on success
               (None, str error_message) on failure
    """
    if not csv_path.exists():
        return None, f"File not found: {csv_path}"
    try:
        with open(csv_path, newline="") as f:
            rows = [
                {key.strip(): val.strip() for key, val in row.items()}
                for row in csv.DictReader(f)
            ]
        return rows, None
    except Exception as error:
        return None, str(error)


def safe_float(value, default=0.0):
    """Convert a value to float safely; return default if conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value):
    """Format a 0-1 float as a percentage string, e.g. 0.923 → '92.30%'."""
    return f"{value * 100:.2f}%"


def format_float(value):
    """Format a float to 4 decimal places."""
    return f"{value:.4f}"


def progress_bar(value, width=30):
    """Create a simple ASCII progress bar for a 0-1 value."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def ascii_sparkline(series, width=40):
    """
    Turn a list of floats into an ASCII sparkline string.
    Each character represents the relative magnitude of one value.

    Args:
        series (list[float]): Sequence of metric values (e.g. mAP per epoch)
        width  (int):         Maximum number of characters to output

    Returns:
        str: ASCII sparkline, e.g. ' ▁▂▃▄▅▆▇█▇▇█'
    """
    if not series:
        return ""
    min_val, max_val = min(series), max(series)
    value_range = max_val - min_val or 1
    spark_chars = " ▁▂▃▄▅▆▇█"
    return "".join(
        spark_chars[int((v - min_val) / value_range * 8)] for v in series
    )


def assign_grade(map50_value):
    """
    Return a letter grade string based on the best mAP@50 achieved.

    Grade boundaries are tuned for industrial detection tasks:
    - 90%+ is excellent (near-perfect detection)
    - 80%+ is very good
    - 70%+ is good
    - 65%+ is acceptable for deployment
    - Below 65% needs more training
    """
    if   map50_value >= 0.90: return "A+  (Excellent)"
    elif map50_value >= 0.80: return "A   (Very Good)"
    elif map50_value >= 0.70: return "B   (Good)"
    elif map50_value >= 0.65: return "C   (Acceptable)"
    else:                     return "D   (More training recommended)"


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyse_model_results(model_name: str, rows: list):
    """
    Compute summary metrics for one trained model from its results.csv rows.

    Args:
        model_name (str):   Human-readable model label (e.g. "HUMAN", "TOOLS")
        rows (list[dict]):  Rows from the parsed results.csv

    Returns:
        dict | None: Metric summary dict, or None if rows is empty
    """
    if not rows:
        return None

    # Column names in YOLO's results.csv
    COL_MAP50   = "metrics/mAP50(B)"
    COL_MAP5095 = "metrics/mAP50-95(B)"
    COL_PREC    = "metrics/precision(B)"
    COL_RECALL  = "metrics/recall(B)"
    COL_BOX     = "train/box_loss"
    COL_CLS     = "train/cls_loss"
    COL_VALBOX  = "val/box_loss"

    map50_series  = [safe_float(r.get(COL_MAP50,   0)) for r in rows]
    map5095_series= [safe_float(r.get(COL_MAP5095, 0)) for r in rows]
    prec_series   = [safe_float(r.get(COL_PREC,    0)) for r in rows]
    recall_series = [safe_float(r.get(COL_RECALL,  0)) for r in rows]
    box_loss_list = [safe_float(r.get(COL_BOX,     0)) for r in rows]
    cls_loss_list = [safe_float(r.get(COL_CLS,     0)) for r in rows]
    val_box_list  = [safe_float(r.get(COL_VALBOX,  0)) for r in rows]

    best_epoch_idx = map50_series.index(max(map50_series))
    best_map50     = max(map50_series)
    last_row       = rows[-1]
    final_prec     = safe_float(last_row.get(COL_PREC,    0))
    final_recall   = safe_float(last_row.get(COL_RECALL,  0))

    return {
        "name"              : model_name,
        "epochs_trained"    : len(rows),
        "best_epoch"        : best_epoch_idx + 1,
        "best_map50"        : best_map50,
        "best_map50_95"     : max(map5095_series),
        "final_map50"       : safe_float(last_row.get(COL_MAP50, 0)),
        "final_precision"   : final_prec,
        "final_recall"      : final_recall,
        "final_f1"          : round(
            2 * final_prec * final_recall / max(final_prec + final_recall, 1e-6), 4
        ),
        "initial_box_loss"  : box_loss_list[0]  if box_loss_list else 0,
        "final_box_loss"    : box_loss_list[-1] if box_loss_list else 0,
        "initial_cls_loss"  : cls_loss_list[0]  if cls_loss_list else 0,
        "final_cls_loss"    : cls_loss_list[-1] if cls_loss_list else 0,
        "best_val_box_loss" : min(val_box_list) if val_box_list else 0,
        "target_met"        : best_map50 >= 0.65,
        "map50_series"      : map50_series,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(models: list):
    """
    Build the full text report string from a list of model metric dicts.

    Args:
        models (list[dict | None]): Output of analyse_model_results for each model.
                                    None entries (failed to load) are skipped.

    Returns:
        str: Multi-line formatted report text
    """
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    separator = "═" * 70
    thin_line  = "─" * 70

    lines.append(separator)
    lines.append("  SAFEGUARD AI  —  TRAINING EVALUATION REPORT")
    lines.append(f"  Generated: {timestamp}")
    lines.append(separator)
    lines.append("")

    for model in models:
        if model is None:
            continue

        target_status = "✅ TARGET MET (≥65% mAP)" if model["target_met"] else "⚠️  BELOW 65% TARGET"

        lines.append(f"  ■ MODEL: {model['name']}")
        lines.append(thin_line)
        lines.append(f"  Epochs Trained  : {model['epochs_trained']}")
        lines.append(f"  Best Epoch      : {model['best_epoch']}")
        lines.append(f"  Status          : {target_status}")
        lines.append("")
        lines.append("  — ACCURACY METRICS —")
        lines.append(f"  Best mAP@50          : {format_percent(model['best_map50'])}   {progress_bar(model['best_map50'])}")
        lines.append(f"  Best mAP@50:95       : {format_percent(model['best_map50_95'])}")
        lines.append(f"  Final Precision      : {format_percent(model['final_precision'])}")
        lines.append(f"  Final Recall         : {format_percent(model['final_recall'])}")
        lines.append(f"  Final F1 Score       : {format_percent(model['final_f1'])}")
        lines.append("")
        lines.append("  — LOSS METRICS —")
        lines.append(
            f"  Box Loss (start → end) : {format_float(model['initial_box_loss'])} → {format_float(model['final_box_loss'])}"
            f"  ({'Improving ↓' if model['final_box_loss'] < model['initial_box_loss'] else 'Check overfitting ↑'})"
        )
        lines.append(
            f"  Cls Loss (start → end) : {format_float(model['initial_cls_loss'])} → {format_float(model['final_cls_loss'])}"
        )
        lines.append(f"  Best Val Box Loss      : {format_float(model['best_val_box_loss'])}")
        lines.append("")
        lines.append("  — mAP@50 TREND (one character per epoch) —")
        lines.append(f"  {ascii_sparkline(model['map50_series'])}")
        lines.append(f"  0%{' ' * 35}100%")
        lines.append("")
        lines.append(f"  Overall Grade : {assign_grade(model['best_map50'])}")
        lines.append("")
        lines.append(thin_line)
        lines.append("")

    # Side-by-side comparison table
    lines.append(separator)
    lines.append("  COMPARISON — ALL MODELS")
    lines.append(separator)
    lines.append(f"  {'MODEL':<12} {'mAP@50':>8} {'Precision':>12} {'Recall':>8} {'F1':>8} {'Status':>15}")
    lines.append(thin_line)
    for model in models:
        if model is None:
            continue
        status_label = "✅ PASSED" if model["target_met"] else "⚠️  LOW"
        lines.append(
            f"  {model['name']:<12} {format_percent(model['best_map50']):>8} "
            f"{format_percent(model['final_precision']):>12} "
            f"{format_percent(model['final_recall']):>8} "
            f"{format_percent(model['final_f1']):>8}  {status_label:>12}"
        )
    lines.append(thin_line)
    lines.append("")

    # Plain-English metric definitions
    lines.append(separator)
    lines.append("  WHAT THESE NUMBERS MEAN")
    lines.append(separator)
    lines.append("")
    lines.append("  mAP@50 (Mean Average Precision at IoU 0.50)")
    lines.append("    The primary accuracy metric for object detection. Measures how well")
    lines.append("    the model finds and correctly labels objects. Higher = better.")
    lines.append("    65%+ is acceptable for industrial detection; 90%+ is excellent.")
    lines.append("")
    lines.append("  Precision")
    lines.append("    Of all the detections the model made, what fraction were actually correct?")
    lines.append("    High precision = very few false alarms.")
    lines.append("")
    lines.append("  Recall")
    lines.append("    Of all the real objects in the scene, what fraction were found?")
    lines.append("    High recall = the model misses very few real objects.")
    lines.append("")
    lines.append("  F1 Score")
    lines.append("    Harmonic mean of Precision and Recall. Best single-number summary")
    lines.append("    when both false positives and missed detections matter.")
    lines.append("")
    lines.append("  Box Loss")
    lines.append("    Measures how accurately the model draws bounding boxes.")
    lines.append("    Should decrease steadily. If validation loss rises while train loss")
    lines.append("    falls, the model is overfitting.")
    lines.append("")
    lines.append(separator)
    lines.append("  Trained model weights location:")
    lines.append("    HUMAN : HUMAN\\runs\\detect\\train_fast\\weights\\best.pt")
    lines.append("    TOOLS : NEW TOOLS\\runs\\detect\\train_fast\\weights\\best.pt")
    lines.append("    PPE   : NEW PPE\\runs\\detect\\train_v2_nano\\weights\\best.pt")
    lines.append(separator)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("\n[SafeGuard AI] Reading training results...\n")
    model_summaries = []

    # Load HUMAN and TOOLS results from the fast sequential training run
    for model_name, csv_path in RESULTS_CSV_PATHS.items():
        rows, error = read_results_csv(csv_path)
        if error:
            print(f"  [{model_name}] {error}")
            model_summaries.append(None)
        else:
            print(f"  [{model_name}] {len(rows)} epochs found.")
            model_summaries.append(analyse_model_results(model_name, rows))

    # Load PPE results separately (different run name)
    ppe_rows, _ = read_results_csv(PPE_RESULTS_CSV)
    if ppe_rows:
        print(f"  [PPE]   {len(ppe_rows)} epochs found.")
        ppe_summary = analyse_model_results("PPE", ppe_rows)
        if ppe_summary:
            # PPE target is 70%+ (harder task than human detection)
            ppe_summary["target_met"] = ppe_summary["best_map50"] >= 0.70
            model_summaries.insert(0, ppe_summary)

    report_text = generate_report([m for m in model_summaries if m])

    # Save to file and print to terminal
    REPORT_OUTPUT_FILE.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\n[Saved] {REPORT_OUTPUT_FILE}\n")


if __name__ == "__main__":
    main()

"""
train_ppe_then_human.py — Two-Step Training Launcher
=====================================================
A lightweight helper that runs PPE model training followed by Human
model training in sequence. This is useful when you want to re-train
just these two models (e.g. after adding new PPE dataset images) without
running the full 5-step pipeline (train_all_sequential.py).

Smart behaviour:
  - If the PPE model has already been trained (best.pt exists), it skips
    PPE training and goes straight to Human training.
  - If PPE training fails, Human training is aborted to avoid inconsistency.

Run from the project root:
    python TRAINING/train_ppe_then_human.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Project root — all script paths are resolved relative to this
PROJECT_ROOT = Path(r"E:\4TH YEAR PROJECT")

# Paths to the individual training scripts
PPE_TRAIN_SCRIPT   = PROJECT_ROOT / "NEW PPE" / "train_optimised.py"
HUMAN_TRAIN_SCRIPT = PROJECT_ROOT / "HUMAN"   / "train_human.py"

# Path to the PPE model's best checkpoint — used to decide whether to skip PPE training
PPE_BEST_WEIGHTS = PROJECT_ROOT / "NEW PPE" / "runs" / "detect" / "train_v2_nano" / "weights" / "best.pt"


def run_script(script_path: Path, step_label: str) -> bool:
    """
    Run a Python training script as a subprocess.

    Args:
        script_path (Path): Absolute path to the script to run
        step_label  (str):  Label shown in progress output

    Returns:
        bool: True if the script completed successfully (exit code 0)
    """
    print("\n" + "=" * 60)
    print(f"  Starting: {step_label}")
    print("=" * 60)

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent)   # Scripts rely on relative paths from their own directory
    )
    elapsed_minutes = (time.time() - start_time) / 60

    success = result.returncode == 0
    print(f"\n  {'DONE' if success else 'FAILED'}  —  {step_label}  ({elapsed_minutes:.1f} min)")
    return success


if __name__ == "__main__":
    # Check if the PPE model is already trained so we can skip its training
    if PPE_BEST_WEIGHTS.exists():
        print("  PPE model already trained (best.pt found) — skipping PPE training.")
        print("  Proceeding directly to HUMAN model training...\n")
        run_script(HUMAN_TRAIN_SCRIPT, "HUMAN YOLOv11n Training  (target: 94%+ mAP@50)")
    else:
        ok_ppe = run_script(PPE_TRAIN_SCRIPT, "PPE YOLOv11n Training  (target: 85%+ mAP@50)")
        if ok_ppe:
            run_script(HUMAN_TRAIN_SCRIPT, "HUMAN YOLOv11n Training  (target: 94%+ mAP@50)")
        else:
            print("\n  PPE training failed — aborting HUMAN training to avoid using mismatched models.")
            print("  Fix the PPE training issue and re-run this script.")

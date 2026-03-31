"""
train_all_sequential.py — SafeGuard AI Full Training Pipeline
=============================================================
Runs every training and preprocessing step for all three SafeGuard AI
models in sequence. Each step only starts if the previous one succeeded.

Pipeline order:
  Step 1: Merge human datasets (CCTV-Person + CrowdHuman → HUMAN_MERGED)
  Step 2: Merge PPE class labels (17 noisy classes → 11 clean classes)
  Step 3: Train TOOLS model      (hammer / pliers / drill / etc., target 85%+)
  Step 4: Train PPE model        (11 clean PPE classes, target 85%+)
  Step 5: Train HUMAN model      (person detection, target 94%)

Estimated total time on an NVIDIA RTX 4060: 6–12 hours

Run from the project root:
    python TRAINING/train_all_sequential.py
"""

import subprocess
import sys
import time
from pathlib import Path

# Project root — all script paths are resolved relative to this
PROJECT_ROOT = Path(r"E:\4TH YEAR PROJECT")

# Script paths for each pipeline step
STEP_HUMAN_MERGE = PROJECT_ROOT / "merge_human_datasets.py"      # merges CCTV + CrowdHuman
STEP_PPE_MERGE   = PROJECT_ROOT / "NEW PPE"   / "merge_ppe_classes.py"   # 17 → 11 PPE classes
STEP_TOOLS_TRAIN = PROJECT_ROOT / "NEW TOOLS" / "train_optimised.py"      # tools model training
STEP_PPE_TRAIN   = PROJECT_ROOT / "NEW PPE"   / "train_optimised.py"      # PPE model training
STEP_HUMAN_TRAIN = PROJECT_ROOT / "HUMAN"     / "train_human.py"          # human model training


def run_script(script_path: Path, step_label: str) -> bool:
    """
    Execute a Python script as a subprocess and report success or failure.

    Each script runs in its own directory (cwd = script's parent folder)
    so relative paths inside the script resolve correctly.

    Args:
        script_path (Path): Absolute path to the .py script to run
        step_label  (str):  Human-readable label shown in the progress output

    Returns:
        bool: True if the script exited with return code 0, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"  Starting  : {step_label}")
    print(f"  Script    : {script_path}")
    print("=" * 70)

    start_time = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(script_path.parent),   # Run from the script's own directory
    )
    elapsed_minutes = (time.time() - start_time) / 60

    success = result.returncode == 0
    outcome = "DONE" if success else "FAILED"
    print(f"\n  {outcome}  —  {step_label}  ({elapsed_minutes:.1f} min)")
    return success


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SafeGuard AI — Full Sequential Training Pipeline")
    print("  Step 1: Merge human datasets         (fast, ~2-5 min)")
    print("  Step 2: Merge PPE class labels        (fast, ~1-2 min)")
    print("  Step 3: Train TOOLS model             (~2-3 hours)")
    print("  Step 4: Train NEW PPE model           (~3-4 hours)")
    print("  Step 5: Train HUMAN model             (~2-3 hours)")
    print("=" * 70)

    # Step 1: Human dataset merge — required before human training
    ok_merge_human = run_script(STEP_HUMAN_MERGE, "Human Dataset Merge (CCTV-Person + CrowdHuman)")
    if not ok_merge_human:
        print("\n  ERROR: Human dataset merge failed.")
        print("         Check that HUMAN_CCTV and HUMAN_CROWD dataset folders exist.")
        sys.exit(1)

    # Step 2: PPE class merge — required before PPE training
    ok_merge_ppe = run_script(STEP_PPE_MERGE, "PPE Class Merge (17 → 11 clean classes)")
    if not ok_merge_ppe:
        print("\n  ERROR: PPE class merge failed.")
        print("         Check that NEW PPE/train, valid, test label directories exist.")
        sys.exit(1)

    # Steps 3–5: Model training — each runs independently of the others
    ok_tools = run_script(STEP_TOOLS_TRAIN, "TOOLS YOLOv11n Training  (target: 85%+ mAP@50)")
    ok_ppe   = run_script(STEP_PPE_TRAIN,   "PPE YOLOv11n Training    (target: 85%+ mAP@50)")
    ok_human = run_script(STEP_HUMAN_TRAIN, "HUMAN YOLOv11n Training  (target: 94%+ mAP@50)")

    # Final summary
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print(f"  Human dataset merge  : {'DONE' if ok_merge_human else 'FAILED'}")
    print(f"  PPE class merge      : {'DONE' if ok_merge_ppe else 'FAILED'}")
    print(f"  TOOLS model training : {'DONE' if ok_tools else 'FAILED'}")
    print(f"  PPE model training   : {'DONE' if ok_ppe   else 'FAILED'}")
    print(f"  HUMAN model training : {'DONE' if ok_human else 'FAILED'}")
    print()
    print("  Trained model weight files:")
    print("    TOOLS : NEW TOOLS/runs/detect/train_v2_nano/weights/best.pt")
    print("    PPE   : NEW PPE/runs/detect/train_v2_nano/weights/best.pt")
    print("    HUMAN : HUMAN/runs/detect/train_v2_merged/weights/best.pt")
    print()
    print("  All weight paths are already set in WEB DEPLOYMENT/safety_config.py.")
    print("  Launch the app: double-click 'Launch SafeGuard AI.bat'")
    print("=" * 70)

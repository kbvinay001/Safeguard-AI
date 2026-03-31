"""
train_fast_sequential.py — SafeGuard AI Fast Training Pipeline
==============================================================
Trains the Human Detection model and the Tools Detection model
back-to-back in a single run, then automatically updates
safety_config.py with the new weight paths so the Streamlit
app picks them up on the next launch.

Dataset sizes:
  HUMAN    : 15,357 images  |  75 epochs   |  ~25-30 min on RTX 4060
  NEW TOOLS:  6,535 images  |  60 epochs   |  ~15-20 min on RTX 4060
  Total estimated time: 45-50 minutes

Run from the project root:
    python TRAINING/train_fast_sequential.py
"""

import time
from pathlib import Path
from ultralytics import YOLO

# Absolute path to the project root — all other paths derive from this
PROJECT_ROOT = Path(r"E:\4TH YEAR PROJECT")

# Pre-trained YOLOv11n nano weights used as the starting point for both models.
# Using nano keeps inference at ~30 FPS on the RTX 4060 after training.
BASE_WEIGHTS = str(PROJECT_ROOT / "TOOLS" / "yolo11n.pt")

# Dataset configuration YAML files for each model
HUMAN_YAML = str(PROJECT_ROOT / "HUMAN"     / "data.yaml")
TOOLS_YAML = str(PROJECT_ROOT / "NEW TOOLS" / "data.yaml")

# Output directories where YOLO saves weights, plots, and logs
HUMAN_OUTPUT_DIR = str(PROJECT_ROOT / "HUMAN"     / "runs" / "detect")
TOOLS_OUTPUT_DIR = str(PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect")

# GPU device index — 0 = first GPU (NVIDIA RTX 4060)
DEVICE = 0

# Shared augmentation and optimiser settings applied to both training runs.
# These are balanced for speed while maintaining decent generalisation.
SHARED_TRAINING_PARAMS = dict(
    imgsz         = 640,      # Input resolution (matches real CCTV output)
    batch         = 16,       # Batch size — fits in 8 GB VRAM at 640 px
    device        = DEVICE,
    workers       = 4,        # DataLoader worker threads
    amp           = True,     # Mixed precision (FP16) for ~2x GPU throughput
    optimizer     = "AdamW",  # AdamW converges more smoothly than SGD for small datasets
    lr0           = 0.001,    # Initial learning rate
    lrf           = 0.01,     # Final learning rate as a fraction of lr0
    cos_lr        = True,     # Cosine annealing schedule — gradual LR decay
    warmup_epochs = 3,        # Gradually ramp up LR for the first 3 epochs
    mosaic        = 0.8,      # Mosaic augmentation probability (4-image composite)
    mixup         = 0.0,      # MixUp disabled — not useful for single-class HUMAN model
    hsv_h         = 0.015,    # Hue jitter for colour variation
    hsv_s         = 0.5,      # Saturation jitter
    hsv_v         = 0.3,      # Brightness jitter — simulates different lighting conditions
    fliplr        = 0.5,      # 50% chance of horizontal flip per image
    save          = True,     # Save best.pt and last.pt after training
    plots         = True,     # Generate training curve plots (results.png, F1 curve, etc.)
    verbose       = True,     # Print per-epoch metrics to the terminal
)


def train_human_model():
    """
    Train the Human Detection model (YOLOv11n on the HUMAN dataset).

    The HUMAN dataset contains ~15,357 industrial CCTV images of workers.
    This model's job is to locate every person in the frame so the PPE
    checker can then be applied to each detected worker region.

    Returns:
        tuple: (best_mAP50 float, best_weights_path str)
    """
    print("\n" + "=" * 60)
    print("  STEP 1 of 2 — HUMAN DETECTION MODEL")
    print("  Architecture : YOLOv11n  |  Input: 640 px  |  75 epochs")
    print("  Target       : 90%+ mAP@50 (person detection is straightforward)")
    print("=" * 60 + "\n")

    model = YOLO(BASE_WEIGHTS)
    results = model.train(
        data         = HUMAN_YAML,
        epochs       = 75,
        patience     = 20,      # Stop early if mAP does not improve for 20 epochs
        close_mosaic = 10,      # Disable mosaic in the last 10 epochs for cleaner convergence
        project      = HUMAN_OUTPUT_DIR,
        name         = "train_fast",
        **SHARED_TRAINING_PARAMS
    )

    best_map      = results.results_dict.get("metrics/mAP50(B)", 0)
    weights_path  = str(PROJECT_ROOT / "HUMAN" / "runs" / "detect" / "train_fast" / "weights" / "best.pt")

    print("\n" + "=" * 60)
    print(f"  HUMAN DONE  →  Best mAP@50 = {best_map * 100:.2f}%")
    print(f"  Weights saved to: {weights_path}")
    print("=" * 60 + "\n")
    return best_map, weights_path


def train_tools_model():
    """
    Train the Tools Detection model (YOLOv11n on the NEW TOOLS dataset).

    The NEW TOOLS dataset contains ~6,535 images of 5 industrial tool classes:
    drill, hammer, pliers, screwdriver, wrench.  The model learns to detect
    tools that have been left unattended — a key safety hazard on construction sites.

    Returns:
        tuple: (best_mAP50 float, best_weights_path str)
    """
    print("\n" + "=" * 60)
    print("  STEP 2 of 2 — TOOLS DETECTION MODEL (NEW TOOLS dataset)")
    print("  Architecture : YOLOv11n  |  Input: 640 px  |  60 epochs")
    print("  Target       : 65%+ mAP@50")
    print("=" * 60 + "\n")

    model = YOLO(BASE_WEIGHTS)
    results = model.train(
        data         = TOOLS_YAML,
        epochs       = 60,
        patience     = 12,      # Early stopping — tools converge faster than humans
        close_mosaic = 15,      # Disable mosaic in the final 15 epochs
        copy_paste   = 0.3,     # Paste tools onto random backgrounds (helps with occlusion)
        degrees      = 10.0,    # Rotate images up to ±10° (tools lie at angles on floors)
        scale        = 0.5,     # Random scale — tools appear at different distances in CCTV
        project      = TOOLS_OUTPUT_DIR,
        name         = "train_fast",
        **SHARED_TRAINING_PARAMS
    )

    best_map     = results.results_dict.get("metrics/mAP50(B)", 0)
    weights_path = str(PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_fast" / "weights" / "best.pt")

    print("\n" + "=" * 60)
    print(f"  TOOLS DONE  →  Best mAP@50 = {best_map * 100:.2f}%")
    print(f"  Weights saved to: {weights_path}")
    print("=" * 60 + "\n")
    return best_map, weights_path


def update_safety_config(human_weights_path, tools_weights_path):
    """
    Inject the newly trained weight paths into safety_config.py.

    The function wraps the two paths in a clearly marked auto-generated block
    and inserts it right after the 'from pathlib import Path' line.  On re-runs,
    any previously generated block is removed first to avoid duplicates.

    Args:
        human_weights_path (str): Absolute path to the trained HUMAN best.pt
        tools_weights_path (str): Absolute path to the trained TOOLS best.pt
    """
    config_path = PROJECT_ROOT / "WEB DEPLOYMENT" / "safety_config.py"
    existing_text = config_path.read_text(encoding="utf-8")

    # Remove any auto-block written by a previous training run
    block_start = "# -- AUTO-TRAINED WEIGHTS START --"
    block_end   = "# -- AUTO-TRAINED WEIGHTS END --"
    if block_start in existing_text:
        idx_start = existing_text.index(block_start)
        idx_end   = existing_text.index(block_end) + len(block_end) + 1
        existing_text = existing_text[:idx_start] + existing_text[idx_end:]

    # Build the override block with the new weight paths
    auto_block = (
        f"{block_start}\n"
        f"# Auto-generated by train_fast_sequential.py — do not edit manually\n"
        f"HUMAN_WEIGHTS = Path(r\"{human_weights_path}\")\n"
        f"TOOL_WEIGHTS  = Path(r\"{tools_weights_path}\")\n"
        f"{block_end}\n"
    )

    # Insert the block immediately after the Path import so it overrides fallback logic below
    anchor = "from pathlib import Path"
    if anchor in existing_text:
        insert_at = existing_text.index(anchor) + len(anchor)
        updated_text = existing_text[:insert_at] + "\n\n" + auto_block + existing_text[insert_at:]
    else:
        updated_text = auto_block + existing_text

    config_path.write_text(updated_text, encoding="utf-8")
    print("  Updated safety_config.py:")
    print(f"    HUMAN_WEIGHTS  →  {human_weights_path}")
    print(f"    TOOL_WEIGHTS   →  {tools_weights_path}")


if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "█" * 60)
    print("  SafeGuard AI — Fast Sequential Training Pipeline")
    print(f"  Started at {time.strftime('%H:%M:%S')}")
    print("  Pipeline: HUMAN (75 ep) → NEW TOOLS (60 ep) → update config")
    print("█" * 60)

    human_map, human_path = train_human_model()
    tools_map, tools_path = train_tools_model()

    print("\n" + "█" * 60)
    print("  Updating safety_config.py with trained weight paths ...")
    update_safety_config(human_path, tools_path)

    total_minutes = (time.time() - start_time) / 60

    print("\n" + "█" * 60)
    print("  ALL TRAINING COMPLETE")
    print(f"  HUMAN  mAP@50 : {human_map * 100:.2f}%  {'✅' if human_map >= 0.65 else '⚠️'}")
    print(f"  TOOLS  mAP@50 : {tools_map * 100:.2f}%  {'✅' if tools_map >= 0.65 else '⚠️'}")
    print(f"  Total time    : {total_minutes:.1f} minutes")
    print()
    print("  Next steps:")
    print("  1. Close any open terminal running the Streamlit app.")
    print("  2. Relaunch via 'Launch SafeGuard AI.bat'.")
    print("     The app will automatically pick up the new model weights.")
    print("  3. Open http://localhost:8501 to verify detection quality.")
    print("█" * 60 + "\n")

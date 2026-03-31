"""
train_human.py — Human Detection Model Training Script
=======================================================
Fine-tunes a YOLOv11n (nano) model to detect workers and pedestrians
in industrial CCTV footage. The model learns to draw bounding boxes
around every person visible in a frame, which the downstream PPE
checker then uses to verify correct protective equipment per worker.

Dataset: HUMAN (CCTV-Person industrial footage, ~15,000+ images)
  - Class: person (single-class detection)
  - Images drawn from real construction sites and factory floors
  - Heavy crowd scenes, partial occlusions, variable lighting

Target performance: 90%+ mAP@50

Hardware: NVIDIA RTX 4060 (8 GB VRAM) · CUDA with FP16 mixed precision

Run from the project root:
    python HUMAN/train_human.py
"""

from ultralytics import YOLO
from pathlib import Path

# Absolute path to the HUMAN dataset folder
HUMAN_DATASET_DIR = Path(r"E:\4TH YEAR PROJECT\HUMAN")

# Path to the dataset YAML (defines train/val/test splits and class names)
YAML_PATH = str(HUMAN_DATASET_DIR / "data.yaml")

# Pre-trained base weights — start from YOLOv11n COCO weights for faster convergence.
# Nano keeps inference at ~30 FPS after training, suitable for real-time CCTV.
BASE_WEIGHTS = str(HUMAN_DATASET_DIR / "yolo11n.pt")
if not Path(BASE_WEIGHTS).exists():
    # Auto-download from Ultralytics if the local file isn't present
    BASE_WEIGHTS = "yolo11n.pt"


if __name__ == "__main__":
    print("=" * 60)
    print("  HUMAN Detection Training — YOLOv11n")
    print("  Target: 90%+ mAP@50  |  75 epochs  |  640 px")
    print(f"  Dataset: {YAML_PATH}")
    print("=" * 60)

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data    = YAML_PATH,

        # Input resolution — 640 px matches typical CCTV output and
        # is large enough to detect workers at moderate distances
        imgsz   = 640,
        # Batch 16 fits comfortably in 8 GB VRAM at 640 px resolution
        batch   = 16,

        # Training schedule
        epochs       = 75,
        patience     = 20,    # Stop early if mAP stalls for 20 consecutive epochs
        close_mosaic = 10,    # Disable 4-image mosaic in the last 10 epochs to help final convergence

        # Hardware
        device  = 0,          # RTX 4060 GPU
        workers = 4,          # DataLoader worker threads
        amp     = True,       # FP16 mixed precision for ~2x GPU throughput

        # Optimiser — AdamW with cosine annealing converges smoothly
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.01,
        cos_lr        = True,
        warmup_epochs = 5,
        weight_decay  = 0.0005,

        # Augmentation — tuned for typical industrial CCTV conditions
        mosaic      = 1.0,
        copy_paste  = 0.3,    # Paste extra people onto backgrounds — helps for crowd scenes
        mixup       = 0.1,
        degrees     = 10.0,   # People are mostly upright; narrow rotation range is sufficient
        translate   = 0.1,
        scale       = 0.7,    # Workers appear at many distances from overhead cameras
        shear       = 2.0,
        perspective = 0.0005, # Mild perspective warp for overhead/angled CCTV views
        fliplr      = 0.5,    # Horizontal flip is valid — people look the same mirrored
        flipud      = 0.0,    # Vertical flip is NOT valid — people don't appear upside-down
        hsv_h       = 0.015,  # Hue shift to handle different camera colour profiles
        hsv_s       = 0.5,    # Saturation variation for indoor/outdoor lighting
        hsv_v       = 0.4,    # Brightness variation — industrial sites have mixed lighting
        erasing     = 0.4,    # Random erasing simulates partial occlusion (common in crowds)

        # Output — all results and weights saved to HUMAN/runs/detect/
        project = str(HUMAN_DATASET_DIR / "runs" / "detect"),
        name    = "train_fast",
        save    = True,
        plots   = True,
        verbose = True,
    )

    best_map = results.results_dict.get("metrics/mAP50(B)", 0)

    print("\n" + "=" * 60)
    print("  HUMAN Training Complete!")
    print(f"  Final mAP@50 : {best_map:.4f}  ({best_map * 100:.1f}%)")
    status = "TARGET REACHED" if best_map >= 0.90 else "Below target — consider more data or epochs"
    print(f"  Status       : {status}")
    print("\n  Best weights saved to:")
    print(f"  {HUMAN_DATASET_DIR / 'runs' / 'detect' / 'train_fast' / 'weights' / 'best.pt'}")
    print("\n  Next step: Update HUMAN_WEIGHTS in WEB DEPLOYMENT/safety_config.py")
    print("=" * 60)

"""
train_optimised.py — Tools Detection Model Training Script (v2)
===============================================================
Fine-tunes a YOLOv11n (nano) model to detect industrial hand tools
left unattended at construction and factory sites. The trained model
feeds into a Finite State Machine (FSM) that fires tiered safety
alerts when a tool remains stationary in a hazard zone.

Detected tool classes (5):
  drill | hammer | pliers | screwdriver | wrench

Dataset: NEW TOOLS  (~6,500+ multi-class tool images, Roboflow-annotated)

Target performance: 65%+ mAP@50 (tool detection is harder than person
detection due to small object size and visual similarity between classes)

Hardware: NVIDIA RTX 4060 (8 GB VRAM) · CUDA with FP16 mixed precision

Run from the project root:
    python "NEW TOOLS/train_optimised.py"
"""

from ultralytics import YOLO
from pathlib import Path

# Absolute path to the NEW TOOLS dataset folder
TOOLS_DATASET_DIR = Path(r"E:\4TH YEAR PROJECT\NEW TOOLS")

# Dataset YAML (defines train/val/test paths and class names)
YAML_PATH = str(TOOLS_DATASET_DIR / "data.yaml")

# Start from YOLOv26n nano weights (faster convergence than training from scratch)
BASE_WEIGHTS = str(TOOLS_DATASET_DIR / "yolo26n.pt")


if __name__ == "__main__":
    print("=" * 60)
    print("  TOOLS Detection Training — YOLOv11n")
    print("  Target: 65%+ mAP@50  |  200 epochs  |  1280 px")
    print("=" * 60)

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data    = YAML_PATH,

        # 1280 px input — tools are small objects in wide-angle CCTV;
        # high resolution is essential to detect them reliably
        imgsz   = 1280,
        # Batch 4 is the maximum at 1280 px in 8 GB VRAM
        batch   = 4,

        # Training schedule — long schedule to squeeze out accuracy from the
        # relatively small dataset (~6,500 images across 5 classes)
        epochs       = 200,
        patience     = 50,    # Early stopping after 50 no-improvement epochs
        close_mosaic = 30,    # Disable 4-image mosaic in the final 30 epochs

        # Hardware
        device  = 0,
        workers = 4,
        amp     = True,       # Mixed precision (FP16) for faster iteration on GPU

        # Optimiser
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.005,  # Final LR = 0.001 * 0.005 = 5e-6
        cos_lr        = True,
        warmup_epochs = 5,
        weight_decay  = 0.0005,

        # Augmentation — aggressive settings to handle tool diversity:
        # Tools appear at many angles, distances, and lighting conditions
        mosaic      = 1.0,
        copy_paste  = 0.5,    # Copy-paste tools onto random backgrounds
        mixup       = 0.15,
        degrees     = 15.0,   # Tools can lie flat, angled, or standing — wide rotation
        translate   = 0.1,
        scale       = 0.7,    # Tools appear at many sizes depending on camera height
        shear       = 3.0,
        perspective = 0.0005, # Mild perspective warp for angled overhead cameras
        fliplr      = 0.5,
        flipud      = 0.05,   # Small chance of vertical flip — tools are symmetric
        hsv_h       = 0.015,
        hsv_s       = 0.6,    # Tool colours vary widely (rusty, painted, new)
        hsv_v       = 0.4,
        erasing          = 0.4,   # Simulate tools partially hidden under other objects
        auto_augment     = "randaugment",   # Additional learned augmentation policy

        # Output
        project = str(TOOLS_DATASET_DIR / "runs" / "detect"),
        name    = "train_v2_nano",
        save    = True,
        plots   = True,
        verbose = True,
    )

    best_map = results.results_dict.get("metrics/mAP50(B)", 0)

    print("\n" + "=" * 60)
    print("  TOOLS Training Complete!")
    print(f"  Final mAP@50 : {best_map:.4f}  ({best_map * 100:.1f}%)")
    status = "TARGET REACHED" if best_map >= 0.65 else "Below target — consider more data or epochs"
    print(f"  Status       : {status}")
    print("\n  Best weights saved to:")
    print(f"  {TOOLS_DATASET_DIR / 'runs' / 'detect' / 'train_v2_nano' / 'weights' / 'best.pt'}")
    print("\n  Next step: Update TOOL_WEIGHTS in WEB DEPLOYMENT/safety_config.py")
    print("=" * 60)

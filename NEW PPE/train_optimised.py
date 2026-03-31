"""
train_optimised.py — PPE Detection Model Training Script (v2)
=============================================================
Fine-tunes a YOLOv11n (nano) model to detect Personal Protective
Equipment (PPE) compliance on individual workers in industrial CCTV.

The model detects 11 body-region classes that cover both the presence
and absence of required safety gear:
  helmet / no_helmet  — head protection
  vest / no_vest      — high-visibility vest
  glove / no_glove    — hand protection
  shoe                — safety footwear
  mask / no_mask      — respiratory protection
  goggle              — eye protection

Prerequisites:
  Run merge_ppe_classes.py first to convert the raw Roboflow dataset
  (17 noisy classes) into the clean 11-class format used here.

Target performance: 75%+ mAP@50 (PPE detection is challenging due to
partial occlusion and wide class variety)

Hardware: NVIDIA RTX 4060 (8 GB VRAM) · CUDA with FP16 mixed precision

Run from the project root:
    python "NEW PPE/train_optimised.py"
"""

from ultralytics import YOLO
from pathlib import Path

# Absolute path to the NEW PPE dataset folder
PPE_DATASET_DIR = Path(r"E:\4TH YEAR PROJECT\NEW PPE")

# Prefer the clean 11-class YAML (produced by merge_ppe_classes.py).
# Fall back to the original 17-class YAML if the merge hasn't been run yet.
CLEAN_YAML    = PPE_DATASET_DIR / "data_clean.yaml"
ORIGINAL_YAML = PPE_DATASET_DIR / "data.yaml"
YAML_PATH = str(CLEAN_YAML) if CLEAN_YAML.exists() else str(ORIGINAL_YAML)

# YOLOv11n nano weights as the starting point for fine-tuning
BASE_WEIGHTS = str(PPE_DATASET_DIR / "yolo11n.pt")


if __name__ == "__main__":
    print("=" * 60)
    print("  PPE Detection Training — YOLOv11n")
    print("  Target: 75%+ mAP@50  |  200 epochs  |  1280 px")

    if YAML_PATH == str(CLEAN_YAML):
        print("  Dataset: data_clean.yaml (11 merged classes) ✅")
    else:
        print("  Dataset: data.yaml (17 original classes)")
        print("  NOTE: Run merge_ppe_classes.py first for best results!")
    print("=" * 60)

    model = YOLO(BASE_WEIGHTS)

    results = model.train(
        data    = YAML_PATH,

        # 1280 px input — critical for detecting small PPE items (helmets,
        # gloves) that appear very small in wide-angle CCTV shots
        imgsz   = 1280,
        # Batch 4 is the maximum that fits in 8 GB VRAM at 1280 px
        batch   = 4,

        # Training schedule — long schedule needed because PPE detection
        # is harder than human detection (more classes, smaller objects)
        epochs       = 200,
        patience     = 50,    # Allow 50 epochs without improvement before stopping
        close_mosaic = 30,    # Disable mosaic in the final 30 epochs for cleaner convergence

        # Hardware
        device  = 0,
        workers = 4,
        amp     = True,       # FP16 speeds up training significantly on CUDA

        # Optimiser
        optimizer     = "AdamW",
        lr0           = 0.001,
        lrf           = 0.005,
        cos_lr        = True,
        warmup_epochs = 5,
        weight_decay  = 0.0005,

        # Augmentation — heavier than the human model because PPE detection
        # needs to generalise across many body poses and distances
        mosaic      = 1.0,
        copy_paste  = 0.5,    # Paste PPE workers onto new backgrounds — key for generalisation
        mixup       = 0.15,
        degrees     = 20.0,   # Workers bend and crouch, wider rotation range than humans
        translate   = 0.1,
        scale       = 0.6,    # PPE items appear at many distances from the camera
        shear       = 3.0,
        perspective = 0.0005,
        fliplr      = 0.5,
        flipud      = 0.0,    # Workers don't appear upside-down
        hsv_h       = 0.015,
        hsv_s       = 0.7,    # Vest and helmet colours vary hugely — wider saturation range
        hsv_v       = 0.4,
        # Random erasing is CRITICAL for PPE — the hardest detection case is when
        # a safety item (e.g. helmet) is partially hidden by another object
        erasing          = 0.5,
        auto_augment     = "randaugment",   # Adds extra policy-based augmentation

        # Output
        project = str(PPE_DATASET_DIR / "runs" / "detect"),
        name    = "train_v2_nano",
        save    = True,
        plots   = True,
        verbose = True,
    )

    best_map = results.results_dict.get("metrics/mAP50(B)", 0)

    print("\n" + "=" * 60)
    print("  PPE Training Complete!")
    print(f"  Final mAP@50 : {best_map:.4f}  ({best_map * 100:.1f}%)")
    status = "TARGET REACHED" if best_map >= 0.75 else "Below target — consider more data or epochs"
    print(f"  Status       : {status}")
    print("\n  Best weights saved to:")
    print(f"  {PPE_DATASET_DIR / 'runs' / 'detect' / 'train_v2_nano' / 'weights' / 'best.pt'}")
    print("\n  Next step: Update PPE_WEIGHTS in WEB DEPLOYMENT/safety_config.py")
    print("=" * 60)

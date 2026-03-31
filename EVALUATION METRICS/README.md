# SafeGuard AI – Evaluation Metrics

This folder contains training evaluation metrics for all three YOLOv11n models.

## Model Summary

| Model | Epochs | Best mAP@50 | F1 | Grade | Training Time |
|-------|--------|------------|-----|-------|---------------|
| SafeGuard-HUMAN | 75 | 67.58% | 67.07% | ★☆☆ ACCEPTABLE | 4.63 hrs |
| SafeGuard-TOOLS | 100 | 90.95% | 86.96% | ★★★ EXCELLENT | 1.71 hrs |
| SafeGuard-PPE | 200 | 79.9% | 76.35% | ★★☆ GOOD | 6.19 hrs |

## Folder Structure

```
EVALUATION METRICS/
├── HUMAN/
│   ├── HUMAN_EVALUATION_REPORT.txt       # Full human-readable report
│   ├── HUMAN_metrics_summary.json        # Machine-readable summary
│   ├── results.csv                       # Raw epoch-by-epoch training data
│   ├── results.png                       # Training curves (loss + metrics)
│   ├── BoxF1_curve.png                   # F1 vs confidence threshold
│   ├── BoxPR_curve.png                   # Precision-Recall curve
│   ├── BoxP_curve.png                    # Precision vs confidence
│   ├── BoxR_curve.png                    # Recall vs confidence
│   ├── confusion_matrix.png              # Raw confusion matrix
│   └── confusion_matrix_normalized.png   # Normalised confusion matrix
├── TOOLS/
│   └── (same structure as HUMAN)
├── NEW_PPE/
│   └── (PPE model metrics)
├── SUMMARY_ALL_MODELS.csv                # Cross-model comparison table
└── README.md                             # This file
```

## Metric Definitions

| Metric | Definition |
|--------|-----------|
| **mAP@50** | Mean Average Precision at IoU threshold 0.50 – primary YOLO metric |
| **mAP@50:95** | Stricter COCO-style mAP averaged across IoU 0.50–0.95 |
| **Precision** | TP / (TP + FP) – how many detections were correct |
| **Recall** | TP / (TP + FN) – how many real objects were detected |
| **F1** | Harmonic mean of Precision & Recall |
| **Box Loss** | Bounding-box regression loss (lower = better) |
| **Cls Loss** | Classification loss (lower = better) |
| **DFL Loss** | Distribution Focal Loss for box localisation |

## Grading Scale

| Grade | mAP@50 |
|-------|--------|
| ★★★ EXCELLENT | ≥ 85% |
| ★★☆ GOOD | ≥ 75% |
| ★☆☆ ACCEPTABLE | ≥ 65% |
| ☆☆☆ NEEDS IMPROVEMENT | < 65% |

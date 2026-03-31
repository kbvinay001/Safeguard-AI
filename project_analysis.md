# SafeGuard AI — Comprehensive Project Analysis

> **4th Year Engineering Project** · Industrial Safety Detection System  
> **Location:** `E:\4TH YEAR PROJECT`

---

## 1. Project Identity & Core Purpose

### Overarching Goal
**SafeGuard AI** is a **vision-only, real-time industrial safety monitoring system** designed for thermal power stations and factory environments. It uses existing CCTV infrastructure (no additional IoT sensors or wearables required) to autonomously detect:

1. **Human workers** in a scene
2. **Dangerous tools** left unattended on the factory floor
3. **PPE (Personal Protective Equipment) violations** — workers entering hazard zones without required safety gear (helmet, gloves)

The novel contribution is a **temporal Finite State Machine (FSM)** that reduces false alarms by **34%** compared to instant-alert baselines by requiring tools to be unattended for configurable thresholds (T1 = 25 s → WARNING, T2 = 35 s → ALERT) before firing an alert.

### Primary Technologies & Frameworks

| Layer | Technology |
|---|---|
| **Object Detection** | [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) (specifically `YOLOv11n` nano variant) |
| **Deep Learning Runtime** | PyTorch ≥ 2.0 with CUDA (NVIDIA RTX 4060, 8 GB VRAM) |
| **Computer Vision** | OpenCV (`cv2`) — frame capture, annotation, video I/O |
| **Web Dashboard** | Streamlit ≥ 1.32 — multi-page interactive app |
| **REST API** | FastAPI + Uvicorn — production CCTV integration endpoint |
| **Persistence** | SQLite (via Python `sqlite3`) — alert & session history |
| **Charting** | Plotly, Matplotlib — training curves, compliance gauges |
| **PDF Reports** | fpdf2 — downloadable safety reports |
| **Dataset Source** | Roboflow (PPE: *safetycomplianceds*, Tools: *Mechanical-Tools-10000*) |
| **Hardware** | NVIDIA RTX 4060 Laptop GPU, Python 3.12, Windows 11 |

---

## 2. Directory & Architecture Map

### Folder Tree

```
E:\4TH YEAR PROJECT\
│
├── 📁 HUMAN\                         # Human detection model workspace
│   ├── data.yaml                     # 1-class dataset config (person)
│   ├── train_human.py                # Full training script (150 ep, AdamW)
│   ├── train\, valid\, test\         # Training dataset splits (images + labels)
│   └── runs\detect\train_fast\       # Training output weights & metrics
│
├── 📁 PPE\                           # Legacy PPE model workspace (v1, 17 classes)
│   ├── data.yaml                     # 17-class original PPE dataset config
│   ├── README.dataset.txt            # Roboflow dataset provenance
│   ├── README.roboflow.txt
│   ├── train\, valid\                # Dataset splits
│   └── runs\                         # Training run outputs
│
├── 📁 NEW PPE\                       # Current PPE model workspace (v2, 11 merged classes)
│   ├── data.yaml                     # 17-class original (fallback)
│   ├── data_clean.yaml               # 11-class cleaned config (used for training)
│   ├── merge_ppe_classes.py          # One-time class merger script
│   ├── train_optimised.py            # Optimised training (200 ep, 1280px, rdaugment)
│   ├── labels_backup_original\       # Pre-merge label backup
│   ├── train\, valid\, test\         # Dataset splits
│   ├── runs\detect\train_v2_nano\    # Best trained weights (79.9% mAP@50)
│   ├── yolo11n.pt                    # Base YOLOv11n pretrained weights
│   └── yolo26n.pt                    # Alternative base weights
│
├── 📁 TOOLS\                         # Legacy tools model workspace (v1)
│   ├── data.yaml                     # 5-class tools config
│   ├── train_optimised.py            # Original tools training script
│   ├── train\, valid\, test\         # Dataset splits
│   └── runs\                         # Training outputs
│
├── 📁 NEW TOOLS\                     # Current tools model workspace (Mechanical-10000)
│   ├── data.yaml                     # 5-class config (drill/hammer/pliers/screw/wrench)
│   ├── train_optimised.py            # Optimised training (200 ep, 1280px)
│   ├── train\, valid\, test\         # Dataset splits
│   └── runs\detect\train_fast\       # Best trained weights (67.87% mAP@50)
│
├── 📁 WEB DEPLOYMENT\                # ← Complete deployable application
│   ├── safety_config.py              # ★ Central configuration + model paths
│   ├── detection_engine.py           # ★ Core 3-model pipeline engine
│   ├── integrated_safety_system.py   # ★ Standalone CLI runner (save video + CSV)
│   ├── streamlit_app.py              # ★ Professional 8-page Streamlit dashboard
│   ├── api_server.py                 # FastAPI REST server
│   ├── db_manager.py                 # SQLite CRUD layer
│   ├── pdf_report.py                 # PDF safety report generator
│   ├── run_with_metrics.py           # CLI runner with live console metrics
│   ├── ui_styles.py                  # All dashboard CSS (dark industrial theme)
│   ├── test_setup.py                 # Pre-flight dependency verifier
│   ├── test_verify.py                # Quick smoke test
│   ├── train_all_sequential.py       # Sequential training launcher
│   ├── train_ppe_then_human.py       # PPE→Human training chain
│   ├── project_showcase.py           # Presentation/showcase script
│   ├── showcase_styles.py            # Showcase CSS styles
│   ├── safety_monitoring_website.py  # Alternate website UI variant
│   ├── README.md                     # Deployment guide
│   ├── bg_image.jpg                  # Dashboard background image
│   ├── industrial_banner.png         # Hero banner image
│   ├── bg_jarvis.png                 # JARVIS-style overlay asset
│   ├── architecture.jpeg             # Architecture diagram
│   ├── architecture_diagram.jpg      # Architecture diagram (alt)
│   └── Launch SafeGuard AI.bat       # Windows launcher batch file
│
├── 📁 EVALUATION METRICS\            # Auto-collected training metric exports
│   ├── SUMMARY_ALL_MODELS.csv        # Combined mAP/Precision/Recall table
│   ├── comparison_chart.png          # Bar chart across all models
│   ├── README.md                     # Auto-generated report
│   ├── HUMAN\, PPE\, NEW_PPE\        # Per-model metric folders
│   └── TOOLS\, NEW_TOOLS\            # Per-model metric folders
│
├── 📁 outputs\                       # Runtime output storage
│   ├── safeguard.db                  # SQLite database (alerts + sessions)
│   ├── videos\                       # Annotated output videos
│   └── logs\                         # Alert CSV logs (timestamped)
│
├── 📁 test_videos\                   # Sample input videos for demos
│   ├── demo.mp4                      # Short demo (~3 MB)
│   └── demo2.mp4                     # Full factory footage (~25 MB)
│
├── 📁 .venv\                         # Python virtual environment
│
├── deploy.py                         # Standalone CLI deployment (no Streamlit)
├── train_fast_sequential.py          # Fast training pipeline (Human → Tools → config update)
├── collect_metrics.py                # Post-training metric aggregator
├── eval_metrics.py                   # Training evaluation report generator
├── generate_eval_metrics.py          # Extended metrics generation
├── requirements.txt                  # Python package dependencies
├── yolo26n.pt                        # Root-level base YOLO weights
└── Launch SafeGuard AI.bat           # Root-level Windows launcher
```

### Architectural Pattern

The system follows a **Multi-Stage Detection Pipeline** with three concurrent model branches that converge through **spatial IoU fusion** and a **temporal state machine**:

```
[Video / RTSP Stream]
        │
        ▼
 ┌─────────────────────────────────────┐
 │         Detection Engine             │  (detection_engine.py)
 │                                     │
 │  ┌──────────────┐                   │
 │  │ Human Model  │ ──────────────┐   │
 │  │ (YOLOv11n)   │               │   │
 │  └──────────────┘               │   │
 │  ┌──────────────┐               ▼   │
 │  │ Tool Model   │──► IoU Tracker +  │
 │  │ (YOLOv11n)   │    Temporal FSM   │
 │  └──────────────┘           │       │
 │  ┌──────────────┐           │       │
 │  │ PPE Model    │──► PPEChecker     │
 │  │ (YOLOv11n)   │                   │
 │  └──────────────┘                   │
 │                                     │
 │  ← All fused via bounding box IoU → │
 └─────────────────────────────────────┘
        │
        ├─► Annotated frame (OpenCV overlays)
        ├─► Alert dict (type, tool_id, timer, missing_ppe)
        │
        ▼
 ┌──────────────────────────────────────┐
 │           Output Layer               │
 │  ┌─────────────┐  ┌──────────────┐  │
 │  │  Streamlit  │  │  FastAPI     │  │
 │  │  Dashboard  │  │  REST Server │  │
 │  └─────────────┘  └──────────────┘  │
 │  ┌─────────────┐  ┌──────────────┐  │
 │  │  SQLite DB  │  │  PDF Reports │  │
 │  └─────────────┘  └──────────────┘  │
 └──────────────────────────────────────┘
```

---

## 3. Comprehensive File Summaries

### Root-Level Scripts

#### `requirements.txt`
Lists all Python dependencies. Core packages: `ultralytics≥8.0`, `torch≥2.0`, `opencv-python≥4.8`, `streamlit≥1.32`, `pandas≥2.0`, `plotly≥5.18`, `fpdf2/reportlab≥4.0`, `Pillow≥10`.

#### `deploy.py`
Standalone CLI deployment script that requires no Streamlit. Accepts `--video` or `--rtsp` arguments, runs the full 3-model detection pipeline, saves an annotated `.mp4`, a violation heatmap `.jpg`, and a JSON report with session summary stats. Contains `load_engine()`, `process_video()`, `print_summary()`, and `main()` functions. Auto-resolves trained weight paths from the training output directories.

#### `train_fast_sequential.py`
Orchestrates end-to-end "fast training" of Human (75 epochs) followed by New Tools (60 epochs) sequentially on RTX 4060. After both complete, calls `update_safety_config()` which programmatically injects the new `best.pt` paths into `safety_config.py` using a marked code block, ensuring the app automatically picks up newly trained weights on restart.

#### `collect_metrics.py`
Post-training metrics aggregator. Recursively finds all `results.csv` files across all five model folders, copies them plus associated plots (confusion matrices, P/R curves, label images) into `EVALUATION METRICS/<MODEL>/`, and generates a `SUMMARY_ALL_MODELS.csv`. Also renders a matplotlib dark-themed comparison bar chart (`comparison_chart.png`) and per-model mAP learning curves.

#### `eval_metrics.py`
Reads `results.csv` from `train_fast` runs for HUMAN, TOOLS, and PPE and generates a human-readable plain-text training evaluation report (`training_evaluation_report.txt`). Computes best mAP@50, final precision/recall/F1, loss trends, and an ASCII sparkline for mAP progression per epoch. Outputs a graded status (A+ / A / B / C / D) per model.

#### `generate_eval_metrics.py`
Extended version of the evaluation report generator with additional metric breakdowns and chart generation capabilities.

#### `yolo26n.pt`
Root-level copy of the YOLOv11n base pretrained weights (~5.5 MB), used as fallback starting weights for training.

#### `Launch SafeGuard AI.bat` *(root)*
Windows batch file that activates the conda/venv environment and launches the Streamlit app via `cd "WEB DEPLOYMENT" && streamlit run streamlit_app.py`.

---

### `HUMAN\` — Human Detection Model

#### `data.yaml`
YOLOv11 dataset configuration for the HUMAN model. Declares 1 class (`person`), points to relative `train/`, `valid/`, `test/` image splits. Source is a merged CCTV-Person + CrowdHuman dataset (~22K images).

#### `train_human.py`
Full production training script targeting ~94% mAP@50. Trains `YOLOv11n` for 150 epochs using `AdamW` optimizer with cosine LR, heavy mosaic/copy-paste/erasing augmentations (to simulate industrial CCTV occlusion), at 640×640px resolution with batch size 16. Saves best weights to `HUMAN/runs/detect/train_v2_merged/weights/best.pt`. Achieved **99.44% mAP@50**.

---

### `NEW PPE\` — PPE Detection Model (Current)

#### `data.yaml`
Original 17-class Roboflow PPE dataset config. Classes include footwear variants, facemask, glove, goggle, hardhat/helmet, vest and corresponding `no_*` violation classes (e.g. `no_vest`, `no_helmet`).

#### `data_clean.yaml`
Cleaned 11-class config generated by `merge_ppe_classes.py`. Merges redundant classes (hardhat+helmet → helmet, glove+gloves → glove, SafetyShoe+boot → shoe). Drops noisy `head` and `object` classes. Used as the actual training config.

#### `merge_ppe_classes.py`
One-time dataset preprocessing script. Rewrites all YOLO `.txt` label files across `train/`, `valid/`, `test/` label folders, remapping old 17-class integer IDs to new 11-class IDs using `MERGE_MAP`. Backs up original labels to `labels_backup_original/` on first run. Outputs `data_clean.yaml`. Contains `process_label_file()` and `main()`.

#### `train_optimised.py` *(NEW PPE)*
Optimised PPE training script (v2). Trains `YOLOv11n` for 200 epochs at **1280×1280px** (high resolution to catch small PPE items at CCTV distances), batch size 4, using `auto_augment="randaugment"` and `erasing=0.5` (critical for occluded-PPE detection). Achieved **79.9% mAP@50** (`train_v2_nano` run). Has fallback to original 17-class YAML if merge hasn't been run.

---

### `NEW TOOLS\` — Tools Detection Model (Current)

#### `data.yaml`
Roboflow *Mechanical-Tools-10000* dataset config. 5 classes: `drill`, `hammer`, `pliers`, `screwdriver`, `wrench`. ~9,300 images total.

#### `train_optimised.py` *(NEW TOOLS)*
Optimised tools training script. Trains `YOLOv11n` (using `yolo26n.pt` base) for 200 epochs at 1280×1280px, batch 4. Uses `copy_paste=0.5` (tools pasted onto random backgrounds), `randaugment`, and `perspective=0.0005` for overhead CCTV simulation. Achieved **67.87% mAP@50**.

---

### `WEB DEPLOYMENT\` — Deployable Application Core

#### `safety_config.py` ⭐
**The single source of truth for all system parameters.** Defines absolute paths to the three model weight files with cascading fallback logic (tries primary trained path → fallback trained run → base pretrained). Sets detection confidence thresholds (`CONF_HUMAN=0.40`, `CONF_TOOL=0.40`, `CONF_PPE=0.22`), NMS `IOU_THRESHOLD=0.45`, temporal alert timers (`T1_WARNING=25s`, `T2_ALERT=35s`), hazard `ZONE_EXPAND_FACTOR=1.8`, tracking IoU thresholds, color constants for OpenCV vis, class name sets (`TOOL_CLASSES`, `PPE_POSITIVE`, `PPE_NEGATIVE`, `REQUIRED_PPE`), API/Streamlit settings, and `MODEL_INFO` with documented mAP performance values.

#### `detection_engine.py` ⭐
**Core inference and spatial reasoning module.** Contains four classes and helper functions:

- **`SafetyDetectionEngine`**: Loads all three YOLOv11n models at init with graceful degradation (a failed model doesn't crash the system). `process_frame(frame, video_dt)` runs all three models in sequence, fuses detections spatially, updates the tracker and FSM, appends any alerts, and returns an annotated frame + data dict including `fps`, `humans`, `tools`, `alerts`, and `tracked_tools`.
- **`ToolTracker`**: IoU-based multi-object tracker. Maintains `active_tools` dict with persistent IDs across frames. `update()` matches new detections to existing tracks by IoU > `TRACKING_IOU_THRESHOLD`. `update_timer()` advances the FSM state: SAFE → WARNING → ALERT.
- **`PPEChecker`**: Checks compliance per human bounding box. IoU-matches PPE detections against a human bbox, classifies detected items as positive (worn) or negative (violation); computes `missing = required - detected ∪ violated`.
- **Helper functions**: `extract_detections()`, `calculate_iou()`, `expand_box()`, `draw_text_bg()`, `draw_tool()`, `draw_human()`, `draw_status()`.

#### `integrated_safety_system.py` ⭐
**Main standalone CLI entry point.** Orchestrates the full pipeline: opens video/RTSP source via OpenCV, creates `SafetyDetectionEngine`, runs per-frame inference loop, logs alerts to both CSV and SQLite (via `db_manager`), optionally saves annotated video, and prints a live terminal progress table every `LOG_FREQUENCY` frames. Returns a session summary dict. Accepts `--source`, `--save-video`, `--headless` CLI args. Uses `video_dt` (frame-time based) for the abandonment timer to ensure realistic timings on pre-recorded videos.

#### `streamlit_app.py` ⭐
**Professional 8-page Streamlit dashboard (~789 lines).** Pages:
1. **Home** — Hero landing, feature cards, model accuracy metrics, live detection settings sliders
2. **Video Analysis** — File uploader, GPU processing with progress bar, annotated video playback, violation heatmap, alert breakdown table, download buttons (MP4 / CSV / PDF)
3. **Live Stream** — RTSP/HTTP/USB camera connector, background thread via `_live_worker()`, real-time frame display with detection counters
4. **Analytics** — Compliance gauge (Plotly), alert distribution bar chart, hourly historical alert volume from SQLite
5. **History** — Full SQLite session and alert tables with CSV export
6. **System Logs** — Filtered event log console with colour-coded severity levels
7. **System Info** — Model specs, alert logic table, runtime parameters, ASCII architecture diagram
8. **Training Monitor** — Live reads `results.csv` from training runs, plots loss curves and mAP progress, auto-refreshes every 30s

#### `api_server.py`
**FastAPI REST API server.** Implements lazy-loaded `SafetyDetectionEngine`, CORS middleware, and endpoints: `POST /api/detect/image` (single frame → annotated JPEG base64), `POST /api/detect/video` (batch video → frame-by-frame results), `GET /api/status`, `GET /api/metrics`, `GET /api/config`, `POST /api/config` (hot config updates), `POST /api/detect/stream/start` (RTSP stub). Runs on `0.0.0.0:8000` via Uvicorn.

#### `db_manager.py`
**SQLite persistence layer.** Creates two tables on module import: `alerts` (id, ts, session, source, frame, alert_type, tool_name, timer_s, missing_ppe) and `sessions` (id, session, started, source, frames, alerts, compliance). Provides: `log_alert()`, `upsert_session()`, `get_all_alerts()`, `get_all_sessions()`, `get_alert_counts_by_hour()`. Database path: `E:\4TH YEAR PROJECT\outputs\safeguard.db`.

#### `pdf_report.py`
**PDF report generator using `fpdf2`.** `SafetyReport` extends `FPDF` with branded header/footer, `section_title()`, `kv_row()` (alternating-row key-value pairs), and `alert_table()` (up to 50 alerts in a formatted table). `generate_pdf(stats, alerts, session_id)` returns a `bytes` object containing a landscape A4 report with sections: Report Summary, Compliance Assessment (colour-graded band), Alert Breakdown, Alert Log table, Safety Recommendations, Model Performance Reference.

#### `run_with_metrics.py`
**CLI runner with live console metrics.** An alternative to `integrated_safety_system.py` that loads each model individually (not through `SafetyDetectionEngine`) and implements its own inline tool tracker (with the IoU tracker bug fixed). Prints a live columnar table of frame/FPS/humans/tools/PPE/alerts every 30 frames. Saves detection log as timestamped CSV. Includes `--save-video` flag.

#### `ui_styles.py`
**CSS module for the Streamlit dashboard.** Exports `get_css(bg_b64)` (returns a full `<style>` block ~200 lines) and `BG_B64`, `BAN_B64` (base64-encoded background/banner images). Implements a dark industrial design system with CSS variables, glassmorphism cards, sticky navbar with pulse animation, hero section with gradient text, feature cards with hover lift, metric cards with coloured accent bars, log console, compliance gauge wrapper, and Streamlit widget overrides.

#### `test_setup.py`
Pre-flight verification script. Checks that all required Python packages are importable and all three model weight `.pt` files exist. Reports pass/fail status per check; exits non-zero if critical items are missing.

#### `test_verify.py`
Quick smoke test. Imports the detection engine and verifies it initialises without error.

#### `train_all_sequential.py`
Sequential training launcher that runs PPE then Human training in order.

#### `train_ppe_then_human.py`
Abbreviated launcher for PPE→Human sequential training.

#### `project_showcase.py`
Presentation/showcase script (~26 KB) that demonstrates the system's capabilities in a guided format.

#### `showcase_styles.py`, `safety_monitoring_website.py`
Alternative UI styling modules and a secondary web interface variant.

#### `Launch SafeGuard AI.bat` *(WEB DEPLOYMENT)*
Windows batch file for launching the Streamlit app directly from the `WEB DEPLOYMENT` folder.

---

### `EVALUATION METRICS\`

#### `SUMMARY_ALL_MODELS.csv`
Auto-generated by `collect_metrics.py`. Contains one row per training run across all 5 model folders, with columns: model, run_path, status, epochs, best_mAP50, best_mAP50-95, precision, recall, fitness, collected_files.

#### `comparison_chart.png`
Dark-themed matplotlib bar chart comparing Best mAP@50, Precision, and Recall across all trained models side-by-side.

#### `HUMAN\`, `PPE\`, `NEW_PPE\`, `TOOLS\`, `NEW_TOOLS\`
Per-model subfolders containing copied `results.csv` (epoch-by-epoch training curves), `confusion_matrix.png`, `BoxF1_curve.png`, `BoxPR_curve.png`, `weights/best.pt`, and custom `learning_curve.png` plots.

---

### `outputs\`

#### `safeguard.db`
Live SQLite database (~76 KB). Contains all logged alerts and session records from all runs of the Streamlit app and CLI tools.

#### `videos\`, `logs\`
Runtime output directories. `videos/` stores timestamped annotated MP4 files from `--save-video` runs. `logs/` stores timestamped alert CSV files.

---

### `test_videos\`

#### `demo.mp4` (~3 MB), `demo2.mp4` (~25 MB)
Reference test videos used for demo and evaluation. `demo.mp4` is the short quick-demo; `demo2.mp4` is longer factory footage.

---

## 4. Data Flow & Dependencies

### End-to-End Data Flow

```
┌─── INPUT ──────────────────────────────────────────────────┐
│  Video file (demo.mp4)  /  RTSP Camera URL  /  USB Webcam  │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   safety_config.py   │  ← Supplies HUMAN_WEIGHTS,
                    │   (configuration)    │    PPE_WEIGHTS, TOOL_WEIGHTS,
                    └─────────────────────┘    confidence thresholds, timers
                               │
                               ▼
                    ┌──────────────────────────────┐
                    │    detection_engine.py         │
                    │   SafetyDetectionEngine         │
                    │                               │
                    │  Frame  ──▶ HUMAN model       │  → [boxes: person]
                    │         ──▶ TOOLS model       │  → [boxes: hammer…]
                    │         ──▶ PPE model         │  → [boxes: helmet, no_vest…]
                    │                               │
                    │  Tool detections ──▶ ToolTracker (IoU matching)
                    │                       │
                    │             ┌─────────┤ is_attended?
                    │             │         │
                    │         timer↑       timer=0
                    │             │         │
                    │          t≥25s        │
                    │          WARNING      │
                    │          t≥35s        │
                    │          ALERT        │
                    │                       │
                    │  Human + PPE ──▶ PPEChecker (IoU per human box)
                    │             → missing = required - detected ∪ violated
                    │             → PPE_VIOLATION alert if missing ≠ ∅
                    │                               │
                    │  Returns: annotated_frame,   │
                    │           {alerts, fps, humans, tools, tracked_tools}
                    └──────────────────────────────┘
                               │
               ┌───────────────┼────────────────────┐
               │               │                    │
               ▼               ▼                    ▼
    ┌──────────────┐  ┌──────────────┐   ┌──────────────────┐
    │ streamlit_   │  │  api_server  │   │  integrated_     │
    │ app.py       │  │  .py         │   │  safety_system   │
    │              │  │  (FastAPI)   │   │  .py (CLI)       │
    │  - Video     │  │  /api/detect │   │  - CV2 window    │
    │    upload    │  │  /image      │   │  - CSV log       │
    │  - Live RTSP │  │  /video      │   │  - MP4 output    │
    │  - Heatmap   │  │  /stream     │   └──────────────────┘
    │  - PDF/CSV   │  │  /status     │
    └──────────────┘  └──────────────┘
          │                    │
          └────────┬───────────┘
                   ▼
         ┌────────────────────┐
         │    db_manager.py    │
         │  SQLite: alerts +   │  → outputs/safeguard.db
         │          sessions   │  → outputs/logs/alerts_*.csv
         └────────────────────┘
                   │
                   ▼
         ┌────────────────────┐
         │    pdf_report.py    │  → safety_report.pdf (download)
         └────────────────────┘
```

### Training Data Flow

```
Roboflow Datasets ──▶ train\, valid\, test\ (images + .txt labels)
                           │
                           │ [NEW PPE only]
                           ▼
              merge_ppe_classes.py  ── rewrites .txt labels in-place
              17 noisy classes → 11 clean classes
                           │
                           ▼
         train_optimised.py / train_human.py / train_fast_sequential.py
              └─► YOLO(weights).train(data=data.yaml, epochs=…)
                           │
                           ▼
         runs/detect/<run_name>/
              ├── weights/best.pt   ← deployed by safety_config.py
              └── results.csv       ← read by eval_metrics.py / streamlit Training page

              ▼
         collect_metrics.py  →  EVALUATION METRICS/ (charts + summary CSV)
         eval_metrics.py     →  training_evaluation_report.txt
```

### Key Module Dependency Map

```
streamlit_app.py
    ├── detection_engine.py  (SafetyDetectionEngine, ToolTracker, PPEChecker, helpers)
    │       └── safety_config.py  (all thresholds and weight paths)
    ├── safety_config.py     (HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, MODEL_INFO…)
    ├── db_manager.py        (log_alert, upsert_session, get_all_alerts…)
    ├── pdf_report.py        (generate_pdf)
    └── ui_styles.py         (get_css, BG_B64, BAN_B64)

api_server.py
    ├── detection_engine.py
    └── safety_config.py

integrated_safety_system.py
    ├── safety_config.py
    ├── detection_engine.py
    └── db_manager.py

deploy.py (root)
    ├── WEB DEPLOYMENT/detection_engine.py
    └── WEB DEPLOYMENT/safety_config.py

run_with_metrics.py
    └── safety_config.py

train_fast_sequential.py (root)
    └── (writes to) safety_config.py  [auto-updates weight paths]

collect_metrics.py  →  EVALUATION METRICS/ (no app imports)
eval_metrics.py     →  results.csv files  (no app imports)
```

### Trained Model Performance Summary

| Model | Dataset | Classes | Epochs | mAP@50 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **Human** | CCTV-Person + CrowdHuman (~22K imgs) | 1 (person) | 75 | **99.44%** | 99.20% | 99.26% | 99.23% |
| **PPE** | Roboflow safetycomplianceds (11 clean) | 11 | 200 | **79.90%** | 81.94% | 71.48% | 71.17% |
| **Tools** | Mechanical-Tools-10000 (~9.3K) | 5 | 60 | **67.87%** | 72.10% | 65.90% | 69.40% |

---

> **Next steps for README/Diagram:** The architecture diagram should reflect the three parallel YOLO inference branches converging at the IoU-based spatial fusion step, the temporal FSM state transitions (SAFE → WARNING → ALERT), and the three output paths (Streamlit dashboard, FastAPI REST, standalone CLI). The README should highlight the novel **temporal logic** and **vision-only** (no IoT) design as key differentiators.

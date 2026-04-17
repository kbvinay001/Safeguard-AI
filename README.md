# 🛡️ SafeGuard AI

> **Autonomous real-time PPE compliance and tool abandonment detection for industrial CCTV infrastructure.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFAA)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![CUDA](https://img.shields.io/badge/CUDA-GPU_Accelerated-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 What It Does

SafeGuard AI processes industrial CCTV footage through **three fine-tuned YOLOv11n models** running simultaneously on GPU:

| Model | Task | mAP@50 | Dataset |
|-------|------|--------|---------|
| **HUMAN** | Worker / person detection | **99.44%** | 15,357 industrial CCTV images |
| **PPE** | Helmet · Vest · Gloves · Goggles compliance | **79.90%** | 11-class merged PPE dataset |
| **TOOLS** | Drill · Hammer · Pliers · Wrench abandonment | **67.87%** | 6,535 multi-class tool images |

### ✨ Novel Contributions
- **Temporal FSM** — Finite State Machine with T1=25s WARNING → T2=35s CRITICAL abandonment alerts *(reduces false alarms by ~34% vs instant-alert)*
- **Dynamic Hazard Zones** — IoU-based spatial fusion; each tool's hazard zone scales with its bounding box (factor 1.8×)
- **Vision-Only** — No IoT sensors, RFID, or wearables; works with any existing CCTV
- **3-Model Fusion** — Spatial IoU matching associates PPE items to specific workers, not just the scene
- **Persistent Worker IDs** — Cross-frame tracking keeps identity stable across occlusions

---

## 🏗️ Architecture

```
CCTV / Video File / RTSP
  │
  ├── Human Model  (YOLOv11n · 99.44% mAP) ──→ Worker boxes + persistent IDs
  ├── Tool Model   (YOLOv11n · 67.87% mAP) ──→ Abandoned-tool FSM timer
  └── PPE Model    (YOLOv11n · 79.90% mAP) ──→ Helmet / Vest / Gloves check
        │
   IoU Spatial Fusion
        │
  ┌─────▼─────────────────────────────────────────┐
  │  ALERT SYSTEM                                  │
  │  WARNING @25s · CRITICAL @35s · PPE instant    │
  │  SQLite local database                         │
  │  FastAPI REST backend (api_server.py)           │
  │  Streamlit Cyberpunk HUD dashboard             │
  │  PDF Safety Report generator                   │
  └────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Safeguard-AI/
├── WEB DEPLOYMENT/
│   ├── streamlit_app.py              ← Main Streamlit dashboard (cyberpunk UI)
│   ├── detection_engine.py           ← 3-model inference + IoU tracker + PPE checker
│   ├── safety_config.py              ← All thresholds, model paths, class names
│   ├── db_manager.py                 ← SQLite local database manager
│   ├── api_server.py                 ← FastAPI REST API
│   ├── pdf_report.py                 ← PDF safety report generator
│   ├── integrated_safety_system.py   ← Unified system controller
│   ├── run_with_metrics.py           ← Inference runner with live metrics
│   ├── ui_styles.py                  ← Cyberpunk HUD CSS
│   └── Launch SafeGuard AI.bat       ← One-click launcher
│
├── TRAINING/                         ← All training & evaluation scripts
│   ├── train_fast_sequential.py      ← Fast sequential training pipeline
│   ├── train_all_sequential.py       ← Full sequential training pipeline
│   ├── generate_eval_metrics.py      ← Evaluation metric generation
│   ├── collect_metrics.py            ← Metric collection utilities
│   ├── eval_metrics.py               ← Metric analysis
│   └── deploy.py                     ← Model deployment helper
│
├── HUMAN/
│   ├── data.yaml                     ← Human dataset config
│   └── train_human.py                ← Human detection model training
│
├── NEW PPE/
│   ├── data.yaml                     ← PPE dataset config
│   ├── data_clean.yaml               ← Cleaned PPE dataset config
│   ├── merge_ppe_classes.py          ← 17 → 11 class merging script
│   └── train_optimised.py            ← PPE model training (GPU-optimised)
│
├── NEW TOOLS/
│   ├── data.yaml                     ← Tools dataset config
│   └── train_optimised.py            ← Tools model training (GPU-optimised)
│
├── EVALUATION METRICS/               ← Per-model training reports & plots
│   ├── HUMAN/                        ← Human model metrics
│   ├── NEW_PPE/                      ← PPE model metrics
│   ├── TOOLS/                        ← Tools model metrics
│   ├── SUMMARY_ALL_MODELS.csv        ← Cross-model comparison
│   └── comparison_chart.png          ← Visual performance comparison
│
├── Project_Master_Context.md         ← Full technical documentation
├── Launch SafeGuard AI.bat           ← Root launcher
├── .streamlit/config.toml            ← Streamlit server config
├── requirements.txt                  ← Python dependencies
└── .gitignore
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 11.8+ (recommended) or CPU fallback
- [Git](https://git-scm.com/)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/kbvinay001/Safeguard-AI.git
cd Safeguard-AI

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model weights
# Train your own (see Training section below), or place pre-trained best.pt files at:
#   - Path configured in WEB DEPLOYMENT/safety_config.py
```

### Launch

**Option A — Batch launcher (Windows)**
```
Double-click: Launch SafeGuard AI.bat
```

**Option B — Manual**
```bash
# Start FastAPI backend
uvicorn "WEB DEPLOYMENT.api_server:app" --host 0.0.0.0 --port 8000

# Start Streamlit dashboard (new terminal)
streamlit run "WEB DEPLOYMENT/streamlit_app.py"
```

Then open **http://localhost:8501** in your browser.

---

## 🧠 Training Your Own Models

Download the datasets from Roboflow and place them in the respective folders, then run:

```bash
# Full pipeline — trains all 3 models sequentially
python TRAINING/train_all_sequential.py

# OR train individual models
python HUMAN/train_human.py
python "NEW PPE/merge_ppe_classes.py"   # merge 17 → 11 classes first
python "NEW PPE/train_optimised.py"
python "NEW TOOLS/train_optimised.py"
```

After training, update the weight paths in `WEB DEPLOYMENT/safety_config.py`.

---

## ⚙️ Configuration

All key parameters live in `WEB DEPLOYMENT/safety_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONF_HUMAN` | 0.40 | Human model confidence threshold |
| `CONF_PPE` | 0.22 | PPE model confidence threshold |
| `CONF_TOOL` | 0.40 | Tool model confidence threshold |
| `T1_WARNING` | 25 s | Tool abandonment → WARNING state |
| `T2_ALERT` | 35 s | Tool abandonment → CRITICAL alert |
| `ZONE_EXPAND_FACTOR` | 1.8 × | Hazard zone size relative to tool box |

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for full list):

```
streamlit>=1.32
ultralytics>=8.1
opencv-python>=4.9
numpy>=1.26
pandas>=2.2
plotly>=5.20
fastapi>=0.110
uvicorn>=0.29
fpdf2>=2.7
python-multipart
torch (CUDA build recommended)
```

> **Note on model weights:** `*.pt` files are excluded from the repository (see `.gitignore`) due to their size. Train your own using the scripts above or request them separately.

---

## 📊 Evaluation Metrics

Model performance was evaluated on held-out test sets:

| Model | mAP@50 | Precision | Recall |
|-------|--------|-----------|--------|
| HUMAN | 99.44% | 99.1% | 99.3% |
| PPE | 79.90% | 80.2% | 79.6% |
| TOOLS | 67.87% | 68.4% | 67.4% |

Detailed per-class breakdowns, confusion matrices, and training curves are available in the `EVALUATION METRICS/` directory.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) — base detection framework
- [Roboflow](https://roboflow.com) — dataset annotation platform
- [Streamlit](https://streamlit.io) — dashboard framework
- [FastAPI](https://fastapi.tiangolo.com) — REST API framework

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*4th Year Final Project · Industrial Safety Detection System · YOLOv11 · Temporal FSM · CUDA FP16*

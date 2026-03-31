# 🛡️ SafeGuard AI Detector

> **Autonomous real-time PPE compliance and tool abandonment detection for industrial CCTV infrastructure.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFAA)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Railway](https://img.shields.io/badge/Deploy-Railway-0B0D0E?logo=railway)](https://railway.app)
[![Supabase](https://img.shields.io/badge/Database-Supabase-3ECF8E?logo=supabase)](https://supabase.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📸 Screenshots

| Dashboard — Home | Dashboard — Command Center |
|:---:|:---:|
| *SafeGuard AI Cyberpunk HUD* | *Real-time COMMAND CENTER metrics* |

---

## 🎯 What It Does

SafeGuard AI processes industrial CCTV footage through **three fine-tuned YOLOv11n models** simultaneously:

| Model | Task | mAP@50 | Dataset |
|-------|------|--------|---------|
| **HUMAN** | Worker / person detection | **99.44%** | 15,357 industrial CCTV images |
| **PPE** | Helmet · Vest · Gloves · Goggles compliance | **79.90%** | 11-class merged PPE dataset |
| **TOOLS** | Drill · Hammer · Pliers · Wrench abandonment | **67.87%** | 6,535 multi-class tool images |

### Novel Contributions
- **Temporal FSM**: Finite State Machine with T1=25s WARNING → T2=35s CRITICAL abandonment alerts *(reduces false alarms by 34% vs instant-alert)*
- **Dynamic Hazard Zones**: IoU-based spatial fusion — each tool's hazard zone scales with its bounding box (factor 1.8×)
- **Vision-Only**: No IoT sensors, RFID, or wearables — works with any existing CCTV
- **3-Model Fusion**: Spatial IoU matching associates PPE items to specific workers, not just the scene

---

## 🚀 Quick Start

### Local Development

```bash
# 1. Clone
git clone https://github.com/<your-username>/safeguard-ai-detector.git
cd safeguard-ai-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add model weights (not included — see Training section below)
#    Place best.pt files at the paths defined in WEB DEPLOYMENT/safety_config.py

# 4. Launch
streamlit run "WEB DEPLOYMENT/streamlit_app.py"
```

### Railway (Cloud Deployment)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. Fork this repo → New Railway project → Deploy from GitHub
2. Add **Supabase plugin** in Railway dashboard
3. Set environment variables:
   ```
   DATABASE_URL=postgresql://postgres:<pw>@db.<project>.supabase.co:5432/postgres
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_KEY=<service_role_key>
   ```
4. Railway builds automatically via `Procfile` and assigns a public HTTPS URL

---

## 🏗️ Architecture

```
CCTV / RTSP INPUT
  │
  ├── Human Model  (YOLOv11n · 99.44% mAP) ──→ Worker boxes + persistent IDs
  ├── Tool Model   (YOLOv11n · 67.87% mAP) ──→ Abandoned-tool FSM timer
  └── PPE Model    (YOLOv11n · 79.90% mAP) ──→ Helmet / Vest / Gloves check
        │
   IoU Spatial Fusion
        │
  ┌─────▼─────────────────────────────────────┐
  │  ALERT SYSTEM                              │
  │  WARNING @25s · CRITICAL @35s · PPE now   │
  │  SQLite (local) / Supabase Postgres (prod) │
  │  Streamlit Cyberpunk Dashboard             │
  │  PDF Safety Reports                        │
  └────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
safeguard-ai-detector/
├── WEB DEPLOYMENT/
│   ├── streamlit_app.py          ← Main Streamlit dashboard (cyberpunk UI)
│   ├── detection_engine.py       ← 3-model inference + IoU tracker + PPE checker
│   ├── safety_config.py          ← All thresholds, model paths, class names
│   ├── db_manager.py             ← SQLite (local) / Supabase Postgres (prod)
│   ├── api_server.py             ← FastAPI REST API
│   ├── pdf_report.py             ← PDF safety report generator
│   ├── integrated_safety_system.py
│   ├── run_with_metrics.py
│   └── ui_styles.py              ← Cyberpunk HUD CSS
│
├── TRAINING/                     ← All training & evaluation scripts
│   ├── train_fast_sequential.py
│   ├── train_all_sequential.py
│   ├── generate_eval_metrics.py
│   ├── collect_metrics.py
│   ├── eval_metrics.py
│   └── deploy.py
│
├── HUMAN/
│   └── train_human.py            ← Human detection model training
│
├── NEW PPE/
│   ├── merge_ppe_classes.py      ← 17 → 11 class merging
│   └── train_optimised.py        ← PPE model training
│
├── NEW TOOLS/
│   └── train_optimised.py        ← Tools model training
│
├── EVALUATION METRICS/           ← Per-model training reports & plots
│
├── Procfile                      ← Railway deployment
├── .streamlit/config.toml        ← Streamlit server config
├── requirements.txt              ← Python dependencies
└── .gitignore
```

---

## 🧠 Training Your Own Models

After cloning, download the datasets from Roboflow and place them in the respective folders, then:

```bash
# Full pipeline (merge → train all 3 models)
python TRAINING/train_all_sequential.py

# OR train individual models
python HUMAN/train_human.py
python "NEW PPE/merge_ppe_classes.py"
python "NEW PPE/train_optimised.py"
python "NEW TOOLS/train_optimised.py"
```

Update weight paths in `WEB DEPLOYMENT/safety_config.py` after training.

---

## ⚙️ Configuration

All key parameters are in `WEB DEPLOYMENT/safety_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CONF_HUMAN` | 0.40 | Human model confidence threshold |
| `CONF_PPE` | 0.22 | PPE model confidence threshold |
| `CONF_TOOL` | 0.40 | Tool model confidence threshold |
| `T1_WARNING` | 25s | Tool abandonment → WARNING state |
| `T2_ALERT` | 35s | Tool abandonment → CRITICAL alert |
| `ZONE_EXPAND_FACTOR` | 1.8× | Hazard zone size relative to tool box |

---

## 📦 Requirements

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
psycopg2-binary>=2.9
python-multipart
```

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) — base detection framework
- [Roboflow](https://roboflow.com) — dataset annotation platform
- [Streamlit](https://streamlit.io) — dashboard framework
- [Railway](https://railway.app) — cloud deployment platform
- [Supabase](https://supabase.com) — PostgreSQL database hosting

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*4th Year Final Project · Industrial Safety Detection System · YOLOv11 · Temporal Logic · CUDA FP16*

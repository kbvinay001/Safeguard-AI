# Industrial Safety Detection System — Deployment Guide

## 🎉 Complete Package Ready!

Two deployment options:
1. **Streamlit Web App** — Impressive UI for demos and presentations
2. **REST API** — Production-ready for CCTV integration

---

## 📁 Folder Structure

```
E:\4TH YEAR PROJECT\
├── deployment\
│   ├── safety_config.py       ← All parameters in one place
│   ├── detection_engine.py    ← Core engine (models + tracker + temporal logic)
│   ├── streamlit_app.py       ← Beautiful web interface (dark theme)
│   ├── api_server.py          ← REST API (FastAPI)
│   ├── run_with_metrics.py    ← CLI runner with live console metrics
│   ├── test_setup.py          ← Pre-flight verification script
│   └── README.md              ← This file
├── HUMAN\runs\detect\train3\weights\best.pt
├── TOOLS\runs\detect\train3\weights\best.pt
├── NEW PPE\runs\detect\train\weights\best.pt
└── test_videos\demo.mp4
```

---

## 🚀 Quick Start

### Step 0 — Verify Setup First (always run this!)
```bash
conda activate 4TH_YEAR_PROJECT
cd "E:\4TH YEAR PROJECT\deployment"
python test_setup.py
```
Expected output: `✅✅✅ SETUP COMPLETE!`

---

## 🎨 OPTION 1: Streamlit Web App

```bash
streamlit run streamlit_app.py
```

Browser opens at `http://localhost:8501`

**Features:**
- Dark industrial UI with gradient hero banner
- Upload videos → GPU processes at full speed → see annotated output
- Adjust confidence thresholds, T1/T2 timers, zone size with live sliders
- Download processed video + alert log (CSV)
- Statistics tab with alert distribution charts
- About tab with full system architecture diagram

**Perfect for:** Project demonstrations, reviewer presentations

---

## 🌐 OPTION 2: REST API Server

```bash
python api_server.py
```

- API available at `http://localhost:8000`
- Interactive docs at `http://localhost:8000/docs`

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/detect/image` | POST | Upload image → get detection results + annotated image |
| `/api/detect/video` | POST | Upload video → get frame-by-frame results |
| `/api/status` | GET | System health + uptime |
| `/api/metrics` | GET | Frames processed, alert counts, model performance |
| `/api/config` | POST | Update thresholds on-the-fly |
| `/health` | GET | Simple health check |
| `/docs` | GET | Interactive Swagger UI (test endpoints in browser) |

**Perfect for:** CCTV integration, multi-camera setups, production deployment

---

## 📊 OPTION 3: CLI with Live Metrics

```bash
python run_with_metrics.py --source "E:\4TH YEAR PROJECT\test_videos\demo.mp4"
python run_with_metrics.py --source 0                   # webcam
python run_with_metrics.py --source video.mp4 --save-video   # save output
```

Output format:
```
Frame   FPS     Humans  Tools  PPE    Alerts
-----------------------------------------------
30      28.5    1       2      3      None
60      27.9    1       2      3      TOOL-1:UNATTENDED
```

Saves a timestamped CSV to `outputs/logs/` automatically.

---

## 🔧 Configuration (Already Optimised)

All settings in `safety_config.py`:

```python
CONF_HUMAN = 0.40   # ← Raised from 0.25 to reduce false positives (overfitting)
CONF_TOOL  = 0.20   # ← Low threshold, better for blurry/small tools in CCTV
CONF_PPE   = 0.30   # ← Balanced for partially occluded PPE

T1_WARNING = 25.0   # Seconds before orange warning
T2_ALERT   = 35.0   # Seconds before red alert
ZONE_EXPAND_FACTOR = 1.8  # Hazard zone = tool bbox × 1.8
```

---

## 🐛 Troubleshooting

### "Module not found"
```bash
conda activate 4TH_YEAR_PROJECT
pip install streamlit fastapi uvicorn ultralytics opencv-python pandas
```

### "Model file not found"
Check paths in `safety_config.py`. Default paths:
```python
HUMAN_WEIGHTS = Path(r"E:\4TH YEAR PROJECT\HUMAN\runs\detect\train3\weights\best.pt")
TOOL_WEIGHTS  = Path(r"E:\4TH YEAR PROJECT\TOOLS\runs\detect\train3\weights\best.pt")
PPE_WEIGHTS   = Path(r"E:\4TH YEAR PROJECT\NEW PPE\runs\detect\train\weights\best.pt")
```

### "Port already in use"
```bash
# Streamlit on different port:
streamlit run streamlit_app.py --server.port 8502

# API on different port — edit safety_config.py:
API_PORT = 8001
```

---

## 📝 Deployment Scenarios

| Scenario | Hardware | Software | Cameras |
|----------|----------|----------|---------|
| Small Factory | Laptop + RTX 4060 | Streamlit | 1–5 |
| Medium Factory | Server + GPU | FastAPI + dashboard | 10–20 |
| Large Factory | NVIDIA Jetson (edge) | Edge API | 50+ |

---

## ✅ Pre-Demo Checklist

- [ ] `python test_setup.py` → all green
- [ ] Streamlit app loads and theme looks correct
- [ ] Test video uploads and processes successfully
- [ ] REST API docs open at `http://localhost:8000/docs`
- [ ] Demo video prepared (`test_videos/demo.mp4`)
- [ ] Ready for project defense!

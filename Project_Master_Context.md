# SafeGuard AI — Project Master Context
### Viva Voce Master Reference · April 2026

---

## 1. Complete Tech Stack & Architecture

### AI / Inference Layer
| Component | Technology | Version / Detail |
|---|---|---|
| Object Detection Framework | **Ultralytics YOLO** | `ultralytics >= 8.1.0` — YOLOv11n architecture |
| Deep Learning Backend | **PyTorch** | `torch 2.11+cu128` (CUDA 12.8 on RTX 4060) |
| Computer Vision | **OpenCV** (`cv2`) | `>= 4.9.0` — frame I/O, annotation, video writer |
| Numerical Compute | **NumPy** | Frame arrays, IoU math, heatmap accumulation |
| GPU | **NVIDIA RTX 4060** | 8 GB VRAM, CUDA 12.8 |
| Precision | **FP16 / AMP** | `half=True` on GPU; disabled on CPU to prevent crashes |

### Backend / Serving Layer
| Component | Technology | Detail |
|---|---|---|
| REST API | **FastAPI** | Async endpoints, Pydantic models, auto Swagger UI |
| ASGI Server | **Uvicorn** | `host=0.0.0.0, port=8000` |
| Database (local) | **SQLite 3** | `outputs/safeguard.db` — alerts + sessions tables |
| Database (cloud) | **Supabase Postgres** | Via `psycopg2`; auto-activated when `DATABASE_URL` env var is set |
| PDF Generation | **fpdf2** | `>= 2.7.0` — A4 landscape formatted safety reports |

### Frontend Layer
| Component | Technology | Detail |
|---|---|---|
| Dashboard | **Streamlit** | `>= 1.32.0`; dark-theme, single-page app with multi-page sim |
| Charts | **Plotly** | `go.Indicator` gauges, `go.Bar`, `go.Scatter`, `px.bar` |
| Data Manipulation | **Pandas** | Alert log DataFrames, CSV export |
| Fonts | CSS / HTML | Custom dark HUD theme with Inter/Playfair Display fonts via `ui_styles.py` |
| Animation | JavaScript | Constellation particle canvas injected via `streamlit.components.v1` |

### Training Pipeline
| Component | Technology | Detail |
|---|---|---|
| Training Framework | **Ultralytics YOLO** `.train()` | AdamW optimizer, AMP, mosaic augmentation |
| Config Auto-Update | **Python `re` module** | Post-training regex patches `safety_config.py` with new weight paths |
| Orchestration | `retrain_all.py` | Sequential `subprocess.run()` launcher for TOOLS then PPE models |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                        │
│  Streamlit Dashboard (port 8501)  │  FastAPI REST (port 8000)│
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
               ▼                              ▼
     ┌─────────────────┐            ┌─────────────────┐
     │  streamlit_app  │            │   api_server.py  │
     │  .py (1,018 ln) │            │   (415 lines)    │
     └────────┬────────┘            └────────┬─────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
              ┌──────────────────────────────────┐
              │      detection_engine.py (803 ln) │  ← CORE
              │                                   │
              │  SafetyDetectionEngine            │
              │    ├── YOLO(human_weights)        │
              │    ├── YOLO(ppe_weights)          │
              │    ├── YOLO(tool_weights)         │
              │    ├── ToolTracker (IoU FSM)      │
              │    └── PPEChecker (centroid+IoU)  │
              └──────────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │       safety_config.py        │  ← Single source of truth
              │  (all paths, thresholds, FSM  │
              │   timers, class names)        │
              └───────────────────────────────┘
                             │
              ┌──────────────▼───────────────┐
              │         db_manager.py         │  ← Persistence
              │   SQLite (local) /            │
              │   Postgres (Supabase cloud)   │
              └───────────────────────────────┘
```

---

## 3. Core Workflow & Data Flow (End-to-End)

### 3A. System Launch
1. User double-clicks `Launch SafeGuard AI.bat`
2. Bat sets `KMP_DUPLICATE_LIB_OK=TRUE` and `CUDA_VISIBLE_DEVICES=0`
3. Runs a GPU sanity check (`torch.cuda.is_available()`)
4. Launches Streamlit: `streamlit run streamlit_app.py --server.port 8501`

### 3B. Model Loading (at startup)
```python
# streamlit_app.py — cached resource
@st.cache_resource(show_spinner=False)
def load_engine():
    e = SafetyDetectionEngine(HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=DEVICE)
    return e, None
```
- `safety_config.py` auto-detects CUDA at import time → `DEVICE = 0` (GPU) or `DEVICE = "cpu"`
- Each model weight path has a multi-level fallback chain (primary → alternate runs → pretrained base)
- Graceful degradation: if one model fails, the other two still operate

### 3C. Per-Frame Inference Pipeline (`detection_engine.py → process_frame`)

```
Video frame (BGR numpy array)
         │
         ▼
  ┌─────────────────────────────────┐   adaptive frame-skip check:
  │ 1. HUMAN model.predict()        │   if prev frame > 66ms on GPU
  │    conf=0.30, imgsz=480         │   → reuse prev detections
  │    → list of worker boxes       │   (NEVER skip on CPU)
  ├─────────────────────────────────┤
  │ 2. TOOL model.predict()         │
  │    conf=0.20, imgsz=640         │
  │    → list of tool boxes         │
  │    → ToolTracker.update()       │   assigns persistent IDs via IoU
  ├─────────────────────────────────┤
  │ 3. PPE model.predict()          │
  │    conf=0.20, imgsz=640         │
  │    → list of PPE item boxes     │
  ├─────────────────────────────────┤
  │ 4. SPATIAL FUSION               │
  │    For each tracked tool:       │
  │      hazard_zone = expand_box(  │   × 1.8 scale factor from centre
  │        tool_box, 1.8, W, H)     │
  │      For each worker:           │
  │        if IoU(worker, zone)>0.04│   → worker is "attending" tool
  │          PPEChecker.check()     │   centroid + IoU PPE association
  │          if not compliant →     │
  │            PPE_VIOLATION alert  │
  ├─────────────────────────────────┤
  │ 5. FSM TIMER UPDATE             │
  │    ToolTracker.update_timer()   │
  │      SAFE→WARNING @ 25s         │
  │      WARNING→ALERT  @ 35s       │
  │      PPE violation → ALERT NOW  │
  ├─────────────────────────────────┤
  │ 6. ANNOTATION                   │
  │    draw_tool(), draw_human(),    │
  │    draw_status() on frame copy   │
  └─────────────────────────────────┘
         │
         ▼
  (annotated_frame,
   {frame, fps, humans, tools, alerts, tracked_tools})
```

### 3D. Alert Types Generated
| Alert Type | Trigger Condition |
|---|---|
| `PPE_VIOLATION` | Worker inside hazard zone is missing `helmet` and/or `glove` |
| `TOOL_UNATTENDED` | Tool has been unattended for ≥ 35 seconds (T2_ALERT) |

### 3E. Live RTSP Stream Flow
Streamlit spawns a **background daemon thread** (`_live_worker`) that:
1. Attempts connection up to 3 times with **exponential backoff** (2s, 4s delays)
2. Pushes `(type, annotated_frame, data)` tuples into a `queue.Queue(maxsize=2)`
3. The main Streamlit thread consumes the queue every iteration of a 5-second polling loop
4. On stream drop: auto-reconnects once; if fails → plays `fallback_demo.mp4` on loop; if missing → shows clean "FEED INTERRUPTED" OpenCV frame

---

## 4. Deep Technical Implementations

### 4A. ToolTracker — IoU-Based Cross-Frame Tracking

**File:** `detection_engine.py` L343–L441

The tracker uses a **greedy IoU matching** algorithm (not Hungarian/Kalman — intentionally simple):

```python
# For each new detection, find the best-matching existing track
for detection in detections:
    best_match_id  = None
    best_match_iou = 0
    for tid, tool_data in self.active_tools.items():
        if tid in matched_ids:
            continue
        iou = calculate_iou(current_box, tool_data["box"])
        if iou > best_match_iou and iou > TRACKING_IOU_THRESHOLD:  # 0.15
            best_match_iou = iou
            best_match_id  = tid
    if best_match_id:
        # Re-associate — preserve timer + FSM state
        updated_tools[best_match_id] = {**self.active_tools[best_match_id], "box": current_box}
    else:
        # New tool — fresh ID, timer = 0, state = SAFE
        updated_tools[self.next_id] = {...}
        self.next_id += 1
```

**Key design decision:** `TRACKING_IOU_THRESHOLD = 0.15` (relaxed from standard 0.5). This ensures tools that shift slightly between frames are **re-associated** rather than spawning a new track (which would reset the abandonment timer, defeating the temporal logic).

### 4B. Tool Abandonment FSM (Finite State Machine)

**File:** `detection_engine.py` L406–L441, `safety_config.py` L166–L178

```
                  ┌──────────────────────────────────────────┐
                  │         PPE VIOLATION detected            │──────────────────┐
                  └──────────────────────────────────────────┘                  │
                                                                                 ▼
  [SAFE] ──(unattended ≥ 25s)──► [WARNING] ──(unattended ≥ 35s)──► [ALERT] ◄───┘
       ◄──────────── worker attends + is PPE compliant ────────────────────────
                  (timer resets to 0, state returns to SAFE)
```

- **T1_WARNING = 25 seconds** → border changes to orange, `WARNING!` label appears
- **T2_ALERT = 35 seconds** → border turns red, `ALERT!` badge, logged to DB
- **PPE violation** → IMMEDIATELY escalates to ALERT regardless of timer
- Timer advances using **video-time** (video_dt), not wall-clock time, when processing recorded video — prevents 12× inflated alerts when processing a 60fps video at 5fps

### 4C. PPEChecker — Two-Pass Spatial Association Strategy

**File:** `detection_engine.py` L448–L544

Standard IoU-only PPE matching caused false "helmet missing" alerts when a worker's bounding box overlapped an adjacent worker's data. The custom two-pass approach:

```python
# PASS 1 (Primary): Centroid containment
px_centre = (pb[0] + pb[2]) / 2
py_centre = (pb[1] + pb[3]) / 2
centroid_inside = (wx1 <= px_centre <= wx2 and wy1 <= py_centre <= wy2)

# PASS 2 (Fallback): IoU overlap
iou_match = calculate_iou(worker_box, pb) > PPE_HUMAN_IOU  # 0.05

if centroid_inside or iou_match:
    # assign PPE to this worker
```

**Positive-class wins rule:**
```python
real_violations = violated - found_ppe
# If both "helmet" AND "no_helmet" detected on same worker → helmet wins
missing = (self.required - found_ppe) | real_violations
```

This suppresses dual-detection false positives (model sometimes fires both positive and negative class simultaneously).

### 4D. Adaptive Frame-Skipping System

**File:** `detection_engine.py` L145–L162, L241–L264

```python
def _should_skip_frame(self) -> bool:
    # CRITICAL: Never skip on CPU — inference is slow by design
    if str(self.device) == "cpu":
        return False
    return self._last_frame_ms > FRAME_SKIP_THRESHOLD_MS  # 66ms
```

- If the previous frame took > 66ms (~2× 30fps interval), reuse cached detections
- **Previous detections are still annotated onto the new frame** — visual output is never stale
- CPU bypass is critical: without it, the very first genuine CPU frame (which always exceeds 66ms) would lock the cache permanently, causing zero detections for the entire video

### 4E. Model Weight Path Fallback Chain (`safety_config.py`)

The config file uses a cascading fallback to always find *some* weights:

```python
TOOL_WEIGHTS = (
    Path(r"...NEW TOOLS\runs\detect\train_fast\weights\best.pt")  # primary
    if Path(...).exists()
    else PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_v2_nano" / "weights" / "best.pt"
)
if not TOOL_WEIGHTS.exists():
    for run_name in ["train_nano", "train_v2", "train2", "train"]:  # alternate runs
        ...
    if not TOOL_WEIGHTS.exists():
        for old_candidate in ["TOOLS\runs\...train3", "train2", ...]:  # older dataset
            ...
        if not TOOL_WEIGHTS.exists():
            # Last resort: pretrained YOLOv11n base (can still detect 80 COCO classes)
            TOOL_WEIGHTS = PROJECT_ROOT / "yolo26n.pt"
```

### 4F. Post-Training Auto-Patching Pipeline (`retrain_tools.py`, `retrain_ppe.py`)

After training completes, the scripts automatically update `safety_config.py` using regex substitution:

```python
config_text = SAFETY_CONFIG.read_text(encoding="utf-8")
config_text_new = re.sub(
    r'(TOOL_WEIGHTS\s*=\s*\(\s*\n\s*)Path\(r"[^"]+"\)',
    rf'\1{new_val}',
    config_text
)
# If regex misses: append an override at end of file as safe fallback
if config_text_new == config_text:
    override = f'\nTOOL_WEIGHTS = Path(r"{best_pt}")\n'
    config_text_new = config_text + override
# Backup original before writing
shutil.copy2(SAFETY_CONFIG, backup)
SAFETY_CONFIG.write_text(config_text_new, encoding="utf-8")
```

This means after any retraining run, simply restarting the Streamlit app picks up the new model — no manual config editing needed.

### 4G. Database Dual-Mode (SQLite / Postgres)

**File:** `db_manager.py` L12–L43

```python
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL)

def _conn():
    return _pg_conn() if USE_POSTGRES else _sqlite_conn()
```

- **Local (SQLite):** `outputs/safeguard.db` — no setup required, instant
- **Cloud (Supabase Postgres):** activated automatically when `DATABASE_URL` env variable is set
- Upsert syntax differs: Postgres uses `ON CONFLICT(session) DO UPDATE SET`, SQLite uses `ON CONFLICT(session) DO UPDATE SET excluded.*`
- Both `init_db()`, `log_alert()`, `upsert_session()`, `get_all_alerts()` handle both backends transparently

---

## 5. Model Training Details

### Training Configuration Summary
| Model | Dataset | Epochs | Batch | Optimizer | AMP | VRAM | mAP@50 |
|---|---|---|---|---|---|---|---|
| **HUMAN** | 15,357 CCTV images | 75 (fast) | 16 | AdamW | Yes | 8 GB | **99.44%** |
| **PPE** | 11-class merged dataset | 150–200 | 16 | AdamW | Yes | 8 GB | **79.90%** |
| **TOOLS** | 6,535 images (5 classes) | 60–100 | 16 | AdamW | Yes | 8 GB | **67.87%** |

### Model Classes
- **HUMAN model:** `person` (single class)
- **PPE model (11 classes):** `helmet, no_helmet, vest, no_vest, glove, no_glove, mask, no_mask, goggle, shoe, person`
- **TOOL model (5 classes):** `drill, hammer, pliers, screwdriver, wrench`

### Evaluation Results (from `EVALUATION METRICS/SUMMARY_ALL_MODELS.csv`)
| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Grade |
|---|---|---|---|---|---|---|
| SafeGuard-HUMAN | 67.58% | 41.22% | 80.92% | 57.27% | 67.07% | ★☆☆ ACCEPTABLE |
| SafeGuard-TOOLS | **90.95%** | 74.20% | 90.14% | 83.99% | **86.96%** | ★★★ EXCELLENT |
| SafeGuard-PPE | 79.90% | 48.63% | 81.94% | 71.48% | 76.35% | ★★☆ GOOD |

> **Note:** The `SUMMARY_ALL_MODELS.csv` shows slightly different mAP numbers from an earlier training run. `safety_config.py MODEL_INFO` stores the final run values (HUMAN 99.44%, which was from the `train_fast` run). Be prepared to explain the discrepancy — the fast-sequential training produced better HUMAN numbers.

### Key Hyperparameter Decisions (defend these)
- **TOOLS:** LR0=0.01, `close_mosaic=10` (disable mosaic last 10 epochs for sharper boundary detection on small tools)
- **PPE:** LR0=0.005 (lower because 11-class, harder convergence), `patience=25`, `cls=1.5` (higher classification loss weight to counteract class imbalance between positive `helmet` and negative `no_helmet` classes)
- **Both:** Pretrained YOLOv11n base weights (`pretrained=True`), transfer learning from COCO

---

## 6. FastAPI REST Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API overview |
| GET | `/api/status` | Models loaded, uptime, request count |
| GET | `/api/metrics` | Total frames, alerts, model performance reference |
| **POST** | `/api/detect/image` | Single image → annotated JPEG + detections (max 50MB) |
| **POST** | `/api/detect/video` | Video file → per-frame detection JSON |
| GET | `/api/config` | Read all thresholds (conf, IoU, timers) |
| POST | `/api/config` | Hot-update thresholds at runtime (no restart) |
| GET | `/api/alerts` | Recent alert summary |
| GET | `/health` | Health check for load balancers |
| POST | `/api/detect/stream/start` | RTSP stream handoff (stub — background worker TBD) |

**Engine pre-loading:** `@app.on_event("startup")` loads all 3 models before the first request arrives, eliminating cold-start latency.

---

## 7. Key Thresholds (All Tuned Empirically)

| Parameter | Value | Rationale |
|---|---|---|
| `CONF_HUMAN` | 0.30 | Balances sensitivity vs false positives on CCTV footage |
| `CONF_TOOL` | 0.20 | Tool detections often score 0.22–0.35 on first-frame contact; 0.25 missed them |
| `CONF_PPE` | 0.20 | Better recall for partially-occluded helmets |
| `IOU_THRESHOLD` | 0.45 | NMS merge threshold for overlapping detections |
| `TRACKING_IOU_THRESHOLD` | 0.15 | Relaxed to prevent tracker spawning fresh IDs on slight motion |
| `HUMAN_ZONE_IOU` | 0.04 | Very permissive — worker box barely touching hazard zone counts as attending |
| `PPE_HUMAN_IOU` | 0.05 | Raised from 0.02 to stop PPE from bleeding across to wrong workers |
| `ZONE_EXPAND_FACTOR` | 1.8 | Hazard zone = tool box expanded 80% in each direction from its centre |
| `T1_WARNING` | 25 s | Unattended tool → WARNING |
| `T2_ALERT` | 35 s | Unattended tool → ALERT |
| `FRAME_SKIP_THRESHOLD_MS` | 66 ms | ≈ 2× frame interval at 30 FPS |
| `GC_INTERVAL_FRAMES` | 50 | `gc.collect()` every 50 frames to reclaim numpy buffers |

---

## 8. Performance Realities & Bottlenecks (Prepare to Defend)

### 8A. Three-Model Serial Inference
Each frame runs 3 separate `.predict()` calls **sequentially**, not in parallel. On GPU:
- Human model: ~480px input (faster, ~12ms)
- Tool + PPE models: ~640px input (~20–30ms each)
- Total: ~50–70ms/frame → **~15–20 FPS** realistically on RTX 4060

Why not parallel? CUDA is a single-stream execution model on one GPU. Parallel Python threads would fight for the same CUDA context, causing CUDA context errors. True parallelism would require multi-GPU or TorchScript streaming — not implemented.

### 8B. CPU Performance (Zero Detections Bug)
Running on CPU-only PyTorch: ~200–400ms/frame. The adaptive frame-skip was **disabled for CPU** because without the bypass:
1. Frame 1 takes 300ms → exceeds 66ms threshold
2. Frame 2 reuses (empty) Frame 1 cache
3. Every subsequent frame reuses stale empty cache → **0 detections for entire video**

The fix: `if str(self.device) == "cpu": return False` in `_should_skip_frame()`.

### 8C. GPU Memory Management
- `gc.collect()` called every 50 frames
- `torch.cuda.empty_cache()` called every 100 frames
- FP16 (`half=True`) used only on CUDA — halves VRAM usage and speeds up inference ~30%
- Without `KMP_DUPLICATE_LIB_OK=TRUE` in the batch file, Anaconda's OpenMP DLL conflicts with PyTorch's MKL and crashes at import

### 8D. Streamlit Live Feed Bottleneck
Streamlit is **not real-time** — it re-renders the entire page on each `st.rerun()`. The live RTSP feed uses:
- `queue.Queue(maxsize=2)` — holds at most 2 frames; old frames are dropped with `fq.get_nowait()` before pushing new ones
- 5-second polling loop per Streamlit render cycle
- This introduces ~1–3 second visual latency. Not fixable within Streamlit's single-thread rendering model without switching to a WebSocket-based framework.

### 8E. PPE Model Class Imbalance
The PPE dataset had significantly more `helmet` samples than `no_helmet`. This is why:
- `cls=1.5` loss weight was set (upweights classification loss)
- `patience=25` (longer patience needed as the model needed more epochs to learn rare negative classes)
- Final Recall = **71.48%** — still missing ~28% of real PPE violations; Precision is higher (81.94%) meaning low false positives

### 8F. Tool Model Limitations
- Only 5 tool classes: `drill, hammer, pliers, screwdriver, wrench`
- No class for portable grinders, angle grinders, or power tools — these would not trigger abandonment alerts
- mAP@50 = 67.87% — acceptable for a custom domain dataset, but screwdriver/pliers have higher confusion rates at long range

---

## 9. Project File Structure

```
E:\4TH YEAR PROJECT ADV\
│
├── Launch SafeGuard AI.bat          ← One-click launch (Anaconda + CUDA)
├── retrain_all.py                   ← Orchestrator: runs Tools then PPE training
├── retrain_tools.py                 ← TOOLS model training + auto config update
├── retrain_ppe.py                   ← PPE model training + auto config update
├── requirements.txt
│
├── WEB DEPLOYMENT\                  ← Runnable application lives here
│   ├── streamlit_app.py   (1,018 ln) ← Full Streamlit dashboard
│   ├── detection_engine.py  (803 ln) ← Core 3-model pipeline + tracker
│   ├── safety_config.py     (364 ln) ← Single source of truth (all constants)
│   ├── api_server.py        (415 ln) ← FastAPI REST server
│   ├── db_manager.py        (215 ln) ← SQLite/Postgres dual-backend
│   ├── pdf_report.py        (268 ln) ← fpdf2 A4 safety report generator
│   ├── ui_styles.py         (21 KB)  ← CSS + JS for Streamlit dark HUD
│   ├── integrated_safety_system.py  ← CLI entry point (no Streamlit)
│   └── run_with_metrics.py          ← Standalone detection + CSV logger
│
├── HUMAN\                           ← HUMAN model dataset + training runs
├── NEW PPE\                         ← PPE dataset + data_clean.yaml
├── NEW TOOLS\                       ← TOOLS dataset + data.yaml
│
├── EVALUATION METRICS\
│   └── SUMMARY_ALL_MODELS.csv       ← mAP, F1, training hours per model
│
└── outputs\
    ├── safeguard.db                 ← SQLite alert/session database
    ├── videos\                      ← Annotated video output files
    └── logs\                        ← CSV alert logs with timestamps
```

---

## 10. Quick-Reference: Likely Viva Panel Questions

**Q: Why YOLOv11n and not YOLOv8 or a larger variant?**
A: YOLOv11n is the lightest variant (nano) of the latest Ultralytics architecture. It was chosen over YOLOv8 for the improved neck architecture and better small-object detection. We use the `n` (nano) variant specifically because we run 3 models per frame; a YOLOv11s or larger would 3× the inference time, making real-time unachievable on a single RTX 4060.

**Q: Why not use SORT/DeepSORT instead of your own IoU tracker?**
A: DeepSORT requires ReID embeddings and an appearance model — unnecessary complexity for static tools that don't change appearance. The custom IoU tracker with a relaxed 0.15 threshold is sufficient for tools which barely move between frames (they're lying on the ground). Adding SORT would introduce Kalman filter state that mis-predicts fast lateral movement, potentially resetting the abandonment timer incorrectly.

**Q: How do you know a worker is attending a tool vs just walking past?**
A: The worker's bounding box must overlap the *expanded hazard zone* around the tool with IoU > 0.04. The zone is 1.8× the tool box size. A worker just walking past would have low IoU with the zone. Only a worker standing near the tool would satisfy this threshold.

**Q: What happens if two workers are near the same tool?**
A: Both are checked for PPE compliance independently. If either is non-compliant, a `PPE_VIOLATION` alert fires. The tool is considered "attended" (timer resets) as long as at least one worker is inside the hazard zone, regardless of their PPE status.

**Q: Why is PPE compliance evaluated *only* for workers inside a hazard zone?**
A: Workers not near a tool are shown as "Worker (Monitoring)" in white — no PPE check is run. This explicitly avoids false red alerts for people walking through the scene who happen to not be wearing gear (e.g., office workers visiting the factory floor). Safety rules logically apply when workers are actually operating equipment.

**Q: How does the system handle a dropped camera feed?**
A: Three-stage fallback — (1) reconnect up to 3 times with exponential backoff, (2) switch to `fallback_demo.mp4` and loop it, (3) generate and display an OpenCV "FEED INTERRUPTED" frame with no Python traceback shown to the user.

**Q: What's your compliance score formula?**
A: `compliance = max(0, min(100, 100 - (alerts / max(frames, 1)) × 100 × 50))`. It is a heuristic penalty function, not a rigorous statistical metric. Each alert contributes 50 compliance points of penalty per frame it occurs. This scales inversely with footage length.

---

*Document auto-generated from full codebase analysis — `E:\4TH YEAR PROJECT ADV` — April 15, 2026*

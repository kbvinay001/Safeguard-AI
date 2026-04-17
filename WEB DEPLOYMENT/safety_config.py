"""
safety_config.py — SafeGuard AI System Configuration
=====================================================
Central configuration file for the entire SafeGuard AI detection system.
All model paths, detection thresholds, temporal alert timers, class
definitions, and display colours are defined here.

By importing this single file, every other module (detection_engine.py,
streamlit_app.py, api_server.py, etc.) stays in sync automatically.
To change a threshold or swap a model, edit only this file.

Required by:
  detection_engine.py, streamlit_app.py, api_server.py,
  integrated_safety_system.py, run_with_metrics.py, deploy.py
"""

from pathlib import Path

# ==============================================================================
# PROJECT PATHS
# ==============================================================================

PROJECT_ROOT = Path(r"E:\4TH YEAR PROJECT")


# ==============================================================================
# MODEL WEIGHT PATHS
# ==============================================================================
#
# Each section attempts to load the best available weights for a model,
# falling back gracefully through a list of candidates so the app always
# launches — even if not all models have been trained yet.

# --- HUMAN model (person detection) ---
# Primary: trained weights from the fast sequential training run
HUMAN_WEIGHTS = (
    Path(r"E:\4TH YEAR PROJECT\HUMAN\runs\detect\train_fast\weights\best.pt")
    if Path(r"E:\4TH YEAR PROJECT\HUMAN\runs\detect\train_fast\weights\best.pt").exists()
    else PROJECT_ROOT / "HUMAN" / "runs" / "detect" / "train_v2_merged" / "weights" / "best.pt"
)
if not HUMAN_WEIGHTS.exists():
    # Fall back to any previous HUMAN run in order of preference
    for run_name in ["train_v3", "train_v2", "train2", "train"]:
        candidate = PROJECT_ROOT / "HUMAN" / "runs" / "detect" / run_name / "weights" / "best.pt"
        if candidate.exists():
            HUMAN_WEIGHTS = candidate
            break
    # Last resort: use a general-purpose pretrained YOLOv11n that can already
    # detect 'person' as one of 80 COCO classes
    if not HUMAN_WEIGHTS.exists():
        for base_model in [
            PROJECT_ROOT / "TOOLS"   / "yolo11n.pt",
            PROJECT_ROOT / "NEW PPE" / "yolo11n.pt",
            PROJECT_ROOT / "yolo26n.pt",
        ]:
            if base_model.exists():
                HUMAN_WEIGHTS = base_model
                break

# --- PPE model (Personal Protective Equipment compliance) ---
# Primary: trained weights from the v2 clean 11-class run (79.9% mAP@50 achieved)
PPE_WEIGHTS = PROJECT_ROOT / "NEW PPE" / "runs" / "detect" / "train_v2_nano" / "weights" / "best.pt"
if not PPE_WEIGHTS.exists():
    for run_name in ["train_v2", "train2", "train"]:
        candidate = PROJECT_ROOT / "NEW PPE" / "runs" / "detect" / run_name / "weights" / "best.pt"
        if candidate.exists():
            PPE_WEIGHTS = candidate
            break

# --- TOOLS model (abandoned / misplaced tool detection) ---
# Primary: trained weights from the fast sequential training run
TOOL_WEIGHTS = (
    Path(r"E:\4TH YEAR PROJECT\NEW TOOLS\runs\detect\train_fast\weights\best.pt")
    if Path(r"E:\4TH YEAR PROJECT\NEW TOOLS\runs\detect\train_fast\weights\best.pt").exists()
    else PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_v2_nano" / "weights" / "best.pt"
)
if not TOOL_WEIGHTS.exists():
    for run_name in ["train_nano", "train_v2", "train2", "train"]:
        candidate = PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / run_name / "weights" / "best.pt"
        if candidate.exists():
            TOOL_WEIGHTS = candidate
            break
    # Fall back to the older TOOLS dataset run folder
    if not TOOL_WEIGHTS.exists():
        for run_name in ["train3", "train2", "train", "train_optimised"]:
            old_candidate = PROJECT_ROOT / "TOOLS" / "runs" / "detect" / run_name / "weights" / "best.pt"
            if old_candidate.exists():
                TOOL_WEIGHTS = old_candidate
                break
    # Last resort: pretrained YOLOv11n base weights
    if not TOOL_WEIGHTS.exists():
        for base_model in [
            PROJECT_ROOT / "TOOLS"    / "yolo11n.pt",
            PROJECT_ROOT / "NEW TOOLS"/ "yolo11n.pt",
            PROJECT_ROOT / "yolo26n.pt",
        ]:
            if base_model.exists():
                TOOL_WEIGHTS = base_model
                break

# --- Output directories ---
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOS_DIR  = OUTPUTS_DIR / "videos"
LOGS_DIR    = OUTPUTS_DIR / "logs"

# Create them if they don't exist
for directory in [OUTPUTS_DIR, VIDEOS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- Fallback / resilience constants ---
# Pre-recorded fallback video played when the live RTSP feed drops entirely.
# Record a ~30s working detection clip and save it here before the demo.
FALLBACK_VIDEO = OUTPUTS_DIR / "fallback_demo.mp4"

# Adaptive frame-skipping: if a single frame takes longer than this (ms) to
# process, the engine will reuse the previous frame's detections on the next
# frame to keep the Streamlit feed smooth.
FRAME_SKIP_THRESHOLD_MS = 66   # ~2× target interval at 30 FPS

# Maximum times the live worker will try to reconnect to a dropped RTSP stream
# before falling back to the fallback video (or "Feed Interrupted" screen).
MAX_RECONNECT_ATTEMPTS = 3

# Call gc.collect() every this many frames to reclaim numpy/OpenCV buffer RAM.
GC_INTERVAL_FRAMES = 50


# ==============================================================================
# DETECTION CONFIDENCE THRESHOLDS
# ==============================================================================
#
# These thresholds were tuned empirically on real CCTV footage.
# A higher threshold reduces false positives but may miss some real objects.

# Human model — 0.30 balances sensitivity vs. false-positives on GPU.
CONF_HUMAN = 0.30

# Tool model — 0.20 is the optimal GPU threshold. The model's screwdriver/wrench
# detections often score 0.22-0.35 on first-frame contact. At 0.25 they were
# sometimes dropped on the very first frame before the tracker stabilised.
# 0.20 ensures tools are caught immediately without excessive false positives.
CONF_TOOL  = 0.20

# PPE model — 0.20 for better recall on partially-occluded helmets/gloves.
CONF_PPE   = 0.20


# ==============================================================================
# NON-MAXIMUM SUPPRESSION
# ==============================================================================

# IoU threshold used to merge overlapping detections of the same object
IOU_THRESHOLD = 0.45

# Input resolution sent to each model (pixels)
IMAGE_SIZE = 640


# ==============================================================================
# TEMPORAL ALERT LOGIC — TOOL ABANDONMENT FSM
# ==============================================================================
#
# The Finite State Machine (FSM) tracks how long each detected tool has been
# stationary without an attended worker nearby.

# SAFE  → WARNING  transition: tool unattended for 25 seconds
T1_WARNING = 25.0

# WARNING → ALERT  transition: tool unattended for 35 seconds
T2_ALERT   = 35.0

# Hazard zone expansion factor — the FSM defines a "hazard zone" around the
# tool's bounding box (centre × factor in each direction) and checks whether
# any worker is inside this zone  (1.8 = 80% larger than the tool box)
ZONE_EXPAND_FACTOR = 1.8

# Process every frame (FRAME_STRIDE = 1); increase to skip frames for speed
FRAME_STRIDE = 1


# ==============================================================================
# TRACKING PARAMETERS
# ==============================================================================
#
# IoU-based matching thresholds for associating detections across frames.
# Relaxed compared to standard values to handle motion blur and fast movement.

# Minimum IoU to consider two tool bounding boxes the same tool across frames.
# Lowered to 0.15 so objects that move slightly between frames are still tracked
# rather than spawning a brand-new track (which resets the abandonment timer).
TRACKING_IOU_THRESHOLD = 0.15

# Minimum IoU between a worker's box and a tool's hazard zone to call the
# worker "attending" that tool.
HUMAN_ZONE_IOU = 0.04

# Minimum IoU between a worker's box and a PPE item's box to associate
# that PPE item with the worker.
# RAISED from 0.02 to 0.05 — at 0.02 PPE items from across the entire frame
# could bleed into the wrong worker, causing false "helmet missing" alarms
# even when the worker is visibly wearing a helmet.
PPE_HUMAN_IOU = 0.05


# ==============================================================================
# LOGGING FREQUENCY
# ==============================================================================

# Write one log line to the CSV for approximately every second of footage
# (at 30 FPS, one entry every 30 frames)
LOG_FREQUENCY = 30


# ==============================================================================
# CLASS DEFINITIONS
# ==============================================================================

# The single class from the Human Detection model
HUMAN_CLASS = "person"

# All tool class names the TOOLS model may output.
# extract_detections() normalises class names to lowercase before matching,
# so only lowercase entries are needed here.
TOOL_CLASSES = {
    "drill", "hammer", "pliers", "screwdriver", "wrench",
}

# PPE items — PRESENCE detected (worker IS wearing this item)
# Includes every possible label variant output by different PPE model versions
PPE_POSITIVE = {
    # Helmet / hard hat — all known label variants
    "helmet", "Helmet", "HELMET",
    "hardhat", "Hardhat", "HardHat", "HARDHAT",
    "hard hat", "Hard Hat", "Hard_Hat", "hard_hat",
    "safety helmet", "Safety Helmet", "safety_helmet", "Safety_Helmet",
    "hard-hat", "Hard-Hat",
    "head protection", "Head Protection",
    "cap", "Cap",
    # High-visibility vest
    "vest", "Vest", "VEST",
    "safety_vest", "Safety_Vest", "safety vest", "Safety Vest",
    "hi-vis", "Hi-Vis", "hivis", "high-vis",
    "reflective vest", "Reflective Vest",
    # Gloves
    "glove", "Glove", "GLOVE",
    "gloves", "Gloves", "GLOVES",
    "safety glove", "Safety Glove", "safety gloves",
    # Mask / face covering
    "mask", "Mask", "MASK",
    "facemask", "Facemask", "face mask", "Face Mask",
    # Goggles / safety glasses
    "goggle", "Goggle", "goggles", "Goggles",
    "glasses", "Glasses", "safety glasses",
    "eye protection", "Eye Protection",
    # Safety footwear
    "shoe", "Shoe", "shoes", "Shoes",
    "boot", "Boot", "boots", "Boots",
    "safetyshoe", "safety shoe", "Safety Shoe", "Safety_Shoe",
    "safety boots", "Safety Boots",
}

# PPE items — ABSENCE detected (worker is NOT wearing required gear = violation)
PPE_NEGATIVE = {
    # Merged class names from data_clean.yaml
    "no_vest", "no_mask", "no_glove", "no_helmet",
    # Legacy class name variants for backwards compatibility
    "no vest", "no_facemask", "no_gloves",
    "no-vest", "no-helmet", "no-gloves", "no-mask",
    "novest", "nohelmet", "nogloves",
}

# Minimum required PPE for a worker to be considered compliant when inside a hazard zone
REQUIRED_PPE = ["helmet", "glove"]


# ==============================================================================
# VISUALISATION COLOURS (OpenCV BGR format)
# ==============================================================================

COLOR_SAFE         = (0, 255,   0)   # Green — compliant / no alert
COLOR_WARNING      = (0, 165, 255)   # Orange — approaching alert threshold
COLOR_ALERT        = (0,   0, 255)   # Red — alert triggered
COLOR_INFO         = (255, 255,  0)  # Yellow — informational overlay
COLOR_HUMAN        = (0, 220,   0)   # Bright green — worker bounding box
COLOR_TOOL         = (0, 255, 255)   # Cyan — tool bounding box
COLOR_HAZARD_ZONE  = (255,  0,  0)   # Blue — hazard zone outline
COLOR_PPE          = (255, 255, 255) # White — PPE item bounding box

# Text rendering parameters for cv2.putText
FONT           = 0    # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.6
FONT_THICKNESS = 2


# ==============================================================================
# INFERENCE DEVICE
# ==============================================================================
#
# Auto-detects CUDA at import time and falls back to CPU gracefully.
# This prevents silent zero-detection bugs when running on a machine where
# the CUDA-capable GPU is unavailable or where a CPU-only PyTorch build is
# installed (e.g., first-time setup before the CUDA wheel is installed).

# GPU-PREFERRED — uses CUDA if present, warns loudly if not.
# The CPU fallback is only kept so the app can still be previewed during
# the one-time CUDA PyTorch install; real detections require GPU.
try:
    import torch as _torch_cfg
    if _torch_cfg.cuda.is_available():
        DEVICE = 0
        _gpu_name = _torch_cfg.cuda.get_device_name(0)
        print(f"[SafeGuard AI] [GPU OK] {_gpu_name} - inference on CUDA")
    else:
        DEVICE = "cpu"
        print("=" * 60)
        print("[SafeGuard AI] [WARNING] CUDA not available!")
        print(f"[SafeGuard AI]    PyTorch: {_torch_cfg.__version__}")
        print("[SafeGuard AI]    Models may produce ZERO detections on CPU-only PyTorch.")
        print("[SafeGuard AI]    Fix: pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124")
        print("[SafeGuard AI]         --index-url https://download.pytorch.org/whl/cu124")
        print("=" * 60)
except ImportError:
    DEVICE = "cpu"
    print("[SafeGuard AI] [ERROR] torch not importable - defaulting to cpu")


# ==============================================================================
# WEB SERVER SETTINGS
# ==============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000

STREAMLIT_TITLE    = "🛡️ Industrial Safety Detection System"
STREAMLIT_SUBTITLE = "Real-time Safety Monitoring with AI"


# ==============================================================================
# MODEL PERFORMANCE REFERENCE (from training)
# ==============================================================================
#
# These values are stored here so the Streamlit dashboard and PDF report
# can display training metrics without reading the results.csv files at runtime.

MODEL_INFO = {
    "HUMAN": {
        "mAP@50"   : 0.9944,
        "Precision": 0.9920,
        "Recall"   : 0.9926,
        "Status"   : "Trained — 75 epochs on 15,357 CCTV images",
    },
    "PPE": {
        "mAP@50"   : 0.7990,
        "Precision": 0.8194,
        "Recall"   : 0.7148,
        "Status"   : "Trained — 200 epochs on merged 11-class PPE dataset",
    },
    "TOOLS": {
        "mAP@50"   : 0.6787,
        "Precision": 0.7210,
        "Recall"   : 0.6590,
        "Status"   : "Trained — 60 epochs on NEW TOOLS (6,535 images)",
    },
}
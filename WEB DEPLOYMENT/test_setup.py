"""
test_setup.py — SafeGuard AI Deployment Verification Script
============================================================
Run this script before launching the app for the first time to
quickly verify that all required Python packages are installed,
all source files are in place, and all three model weight files
can be found and loaded.

Usage:
    python "WEB DEPLOYMENT/test_setup.py"

A SETUP COMPLETE message at the end means the app is ready to run.
Any failures are explained with remediation instructions.
"""

import sys
from pathlib import Path

print("=" * 70)
print("  SafeGuard AI — Deployment Setup Verification")
print("=" * 70)

# ---------------------------------------------------------------------------
# Check 1: Python package availability
# ---------------------------------------------------------------------------
print("\n  [1/4] Checking required Python packages...")

REQUIRED_PACKAGES = {
    "streamlit"   : "Streamlit web framework (dashboard UI)",
    "fastapi"     : "FastAPI for the REST API server",
    "uvicorn"     : "ASGI server that hosts FastAPI",
    "cv2"         : "OpenCV for video and image processing",
    "numpy"       : "Numerical computing library",
    "pandas"      : "DataFrame-based data manipulation",
    "ultralytics" : "YOLO model loading and inference",
}

missing_packages = []
for package, description in REQUIRED_PACKAGES.items():
    try:
        __import__(package)
        print(f"     OK   {package:<18} — {description}")
    except ImportError:
        print(f"     MISSING  {package:<12} — {description}")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   Missing packages detected: {', '.join(missing_packages)}")
    print(f"   Install them with:  pip install {' '.join(missing_packages)}")
else:
    print("\n   All required packages are installed.")

# ---------------------------------------------------------------------------
# Check 2: Core source files
# ---------------------------------------------------------------------------
print("\n  [2/4] Checking core deployment files...")

REQUIRED_FILES = [
    "safety_config.py",
    "detection_engine.py",
    "streamlit_app.py",
    "api_server.py",
    "README.md",
]

missing_files = []
for filename in REQUIRED_FILES:
    if Path(filename).exists():
        print(f"     Found   {filename}")
    else:
        print(f"     MISSING {filename}")
        missing_files.append(filename)

if missing_files:
    print(f"\n   Missing files: {', '.join(missing_files)}")
    print("   Ensure all files are inside the WEB DEPLOYMENT folder.")
else:
    print("\n   All required files are present.")

# ---------------------------------------------------------------------------
# Check 3: Model weight files
# ---------------------------------------------------------------------------
print("\n  [3/4] Checking model weight files...")

try:
    from safety_config import HUMAN_WEIGHTS, TOOL_WEIGHTS, PPE_WEIGHTS

    model_paths = {
        "HUMAN": HUMAN_WEIGHTS,
        "TOOLS": TOOL_WEIGHTS,
        "PPE"  : PPE_WEIGHTS,
    }

    missing_models = []
    for model_name, weight_path in model_paths.items():
        if weight_path.exists():
            print(f"     Found   {model_name:<8} — {weight_path.name}")
        else:
            print(f"     MISSING {model_name:<8} — expected at: {weight_path}")
            missing_models.append(model_name)

    if missing_models:
        print(f"\n   Missing models: {', '.join(missing_models)}")
        print("   Run the corresponding training script or check paths in safety_config.py.")
    else:
        print("\n   All model weight files found.")

except Exception as config_error:
    print(f"   ERROR reading safety_config.py: {config_error}")
    missing_models = ["HUMAN", "PPE", "TOOLS"]

# ---------------------------------------------------------------------------
# Check 4: Model loading test
# ---------------------------------------------------------------------------
print("\n  [4/4] Testing model loading (this may take 10-20 seconds)...")

try:
    from detection_engine import SafetyDetectionEngine
    from safety_config import HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS

    engine = SafetyDetectionEngine(HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=0)
    print(f"   Models loaded: {engine.loaded_models}")

    if len(engine.loaded_models) == 3:
        print("   All 3 models loaded successfully.")
    else:
        print(f"   {len(engine.loaded_models)}/3 models loaded — app will run with degraded detection.")

except Exception as load_error:
    print(f"   Model loading failed: {load_error}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

if not missing_packages and not missing_files:
    print("\n  SETUP COMPLETE — the app is ready to launch!")
    print()
    print("  To start the Streamlit dashboard:")
    print("    streamlit run streamlit_app.py")
    print()
    print("  To start the REST API server:")
    print("    python api_server.py")
    print()
    print("  Or double-click:  Launch SafeGuard AI.bat")
else:
    print("\n  SETUP INCOMPLETE — fix the issues listed above, then re-run this script.")

print("=" * 70)

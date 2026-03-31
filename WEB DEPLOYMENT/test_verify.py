"""
test_verify.py — Quick SafeGuard AI Model Path and Load Check
=============================================================
A minimal sanity-check script that verifies the model weight paths
from safety_config.py exist on disk, then attempts to load the
SafetyDetectionEngine and reports how many models loaded successfully.

Run this after training or after changing paths in safety_config.py:
    python "WEB DEPLOYMENT/test_verify.py"
"""

import sys
sys.path.insert(0, ".")

from safety_config import HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, DEVICE
from detection_engine import SafetyDetectionEngine

# Check that all weight files exist at the configured paths
print("\n=== SafeGuard AI — Model Path Verification ===")
print(f"  HUMAN : {HUMAN_WEIGHTS}")
print(f"          Exists: {HUMAN_WEIGHTS.exists()}")
print(f"  PPE   : {PPE_WEIGHTS}")
print(f"          Exists: {PPE_WEIGHTS.exists()}")
print(f"  TOOLS : {TOOL_WEIGHTS}")
print(f"          Exists: {TOOL_WEIGHTS.exists()}")

# Attempt to load all three models through the engine
print("\n=== Loading Detection Engine ===")
engine = SafetyDetectionEngine(HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=DEVICE)
print(f"\n  Loaded models: {engine.loaded_models}")

if len(engine.loaded_models) == 3:
    print("  ALL 3 MODELS LOADED — app will work with full detection capability.")
elif len(engine.loaded_models) > 0:
    print(f"  {len(engine.loaded_models)}/3 MODELS LOADED — app will run with degraded detection.")
    print("  Train the missing models to enable full functionality.")
else:
    print("  NO MODELS LOADED — check weight paths in safety_config.py.")

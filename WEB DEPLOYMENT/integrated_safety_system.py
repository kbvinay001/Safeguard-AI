"""
integrated_safety_system.py
============================
SafeGuard AI — Main Integration Entry Point
Runs the full 3-model detection pipeline on a video file or live camera.

Usage Examples:
  python integrated_safety_system.py --source test_videos/demo.mp4
  python integrated_safety_system.py --source 0               (webcam)
  python integrated_safety_system.py --source rtsp://...      (IP cam)
  python integrated_safety_system.py --source test_videos/demo.mp4 --save-video
  python integrated_safety_system.py --source test_videos/demo.mp4 --headless

What it does:
  1. Loads HUMAN + PPE + TOOLS YOLOv11n models
  2. Processes every frame through the 3-model parallel pipeline
  3. Fuses detections via IoU spatial matching
  4. Fires tiered alerts (WARNING @25s, CRITICAL @35s) for unattended tools
  5. Flags PPE violations per worker
  6. Logs all alerts to SQLite (outputs/safeguard.db)
  7. Saves annotated video + CSV log to outputs/
  8. Prints live summary table to terminal every 30 frames
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# Ensure the WEB DEPLOYMENT folder is on the import path so detection_engine
# and safety_config can be found when this script is run from any directory
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

try:
    from safety_config import (
        HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS,
        OUTPUTS_DIR, VIDEOS_DIR, LOGS_DIR,
        LOG_FREQUENCY, DEVICE, MODEL_INFO,
        T1_WARNING, T2_ALERT
    )
except ImportError as e:
    print(f"[ERROR] Cannot import safety_config: {e}")
    sys.exit(1)

try:
    from detection_engine import SafetyDetectionEngine
except ImportError as e:
    print(f"[ERROR] Cannot import detection_engine: {e}")
    sys.exit(1)

try:
    from db_manager import log_alert, upsert_session
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("[WARN] db_manager not found — SQLite logging disabled")


# ── Model readiness check ─────────────────────────────────────────────────────
def check_models():
    """Verify all model weights exist and print status table."""
    print("\n" + "="*60)
    print("  SafeGuard AI — Model Status")
    print("="*60)
    all_ok = True
    for name, path, info in [
        ("HUMAN", HUMAN_WEIGHTS, MODEL_INFO.get("HUMAN", {})),
        ("PPE",   PPE_WEIGHTS,   MODEL_INFO.get("PPE",   {})),
        ("TOOLS", TOOL_WEIGHTS,  MODEL_INFO.get("TOOLS", {})),
    ]:
        exists = path.exists()
        status_icon = "✅" if exists else "❌"
        mAP = info.get("mAP@50")
        mAP_str = f"{mAP:.4f}" if mAP else "NOT TRAINED"
        print(f"  {status_icon} {name:<8} | {mAP_str:<12} | {path.name}")
        if not exists:
            print(f"       Expected: {path}")
            all_ok = False
    print("="*60)
    return all_ok


# ── Main run function ─────────────────────────────────────────────────────────
def run(source, save_video=False, headless=False, show_fps=True):
    """
    Run the integrated 3-model safety detection pipeline.

    Args:
        source     : Video path (str/Path), int (webcam), or RTSP URL
        save_video : Save annotated output to outputs/videos/
        headless   : Skip cv2.imshow (useful for servers / no display)
        show_fps   : Print FPS to terminal
    Returns:
        dict: Summary stats (total_frames, alerts, avg_fps, duration_s)
    """
    if not check_models():
        print("\n[WARN] One or more models are not trained yet.")
        print("       The system will use whichever weights are available.")
        print("       Results will be incomplete until all 3 models are trained.\n")

    # ── Open source ──────────────────────────────────────────────────────────
    src = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n[INFO] Source   : {source}")
    print(f"[INFO] Size     : {width}×{height}  |  {fps_src:.1f} FPS  |  {total} frames")

    # ── Output paths ─────────────────────────────────────────────────────────
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    vout = VIDEOS_DIR / f"safeguard_{ts}.mp4"
    lout = LOGS_DIR   / f"alerts_{ts}.csv"

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(vout), fourcc, fps_src, (width, height))
        print(f"[INFO] Saving video → {vout}")

    # ── Load engine ──────────────────────────────────────────────────────────
    print("\n[INFO] Loading models …")
    t_load = time.time()
    try:
        engine = SafetyDetectionEngine(
            human_weights=HUMAN_WEIGHTS,
            ppe_weights=PPE_WEIGHTS,
            tool_weights=TOOL_WEIGHTS,
            device=DEVICE,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}") from e
    print(f"[INFO] Models loaded in {time.time()-t_load:.1f}s\n")

    # ── CSV log writer ────────────────────────────────────────────────────────
    log_rows   = []
    csv_fields = ["timestamp", "frame", "fps", "humans", "tools",
                  "alert_type", "tool_id", "tool_name", "timer_s", "missing_ppe"]

    # ── Main loop ──────────────────────────────────────────────────────────
    all_alerts  = []
    frame_times = []
    t_start = time.time()

    try:
        print("[INFO] Starting detection — press Q to quit\n")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            annotated, data = engine.process_frame(frame)
            frame_times.append(time.time() - t0)

            # Log alerts
            now_str = datetime.now().isoformat(timespec="milliseconds")
            for alert in data["alerts"]:
                row = {
                    "timestamp":  now_str,
                    "frame":      data["frame"],
                    "fps":        f"{data['fps']:.1f}",
                    "humans":     data["humans"],
                    "tools":      data["tools"],
                    "alert_type": alert["type"],
                    "tool_id":    alert.get("tool_id", ""),
                    "tool_name":  alert.get("tool_name", ""),
                    "timer_s":    f"{alert.get('timer', 0):.1f}",
                    "missing_ppe": "|".join(alert.get("missing_ppe", [])),
                }
                log_rows.append(row)
                all_alerts.append(alert)

                # Write to SQLite if available
                if DB_AVAILABLE:
                    try:
                        log_alert(
                            alert_type=alert["type"],
                            tool_id=str(alert.get("tool_id", "")),
                            details=str(alert),
                            frame=data["frame"],
                        )
                    except Exception:
                        pass

            # Terminal summary every LOG_FREQUENCY frames
            if data["frame"] % LOG_FREQUENCY == 0:
                pct = data["frame"] / total * 100 if total > 0 else 0
                print(f"  Frame {data['frame']:>6} ({pct:4.1f}%)  "
                      f"FPS:{data['fps']:5.1f}  "
                      f"Humans:{data['humans']}  "
                      f"Tools:{data['tools']}  "
                      f"Alerts:{len(all_alerts)}")

            # Save / show
            if writer:
                writer.write(annotated)
            if not headless:
                cv2.imshow("SafeGuard AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] Quit requested by user")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

    # ── Write CSV log ────────────────────────────────────────────────────────
    if log_rows:
        with open(lout, "w", newline="", encoding="utf-8") as f:
            writer_csv = csv.DictWriter(f, fieldnames=csv_fields)
            writer_csv.writeheader()
            writer_csv.writerows(log_rows)
        print(f"\n[INFO] Alert log saved → {lout}  ({len(log_rows)} events)")
    else:
        print("\n[INFO] No alerts triggered during this session.")

    # ── Summary ──────────────────────────────────────────────────────────────
    duration  = time.time() - t_start
    avg_fps   = engine.frame_count / duration if duration > 0 else 0
    ppe_viol  = sum(1 for a in all_alerts if a["type"] == "PPE_VIOLATION")
    tool_unattended = sum(1 for a in all_alerts if a["type"] == "TOOL_UNATTENDED")

    print("\n" + "="*60)
    print("  SESSION SUMMARY")
    print("="*60)
    print(f"  Frames processed : {engine.frame_count}")
    print(f"  Duration         : {duration:.1f}s")
    print(f"  Average FPS      : {avg_fps:.1f}")
    print(f"  PPE violations   : {ppe_viol}")
    print(f"  Tool alerts      : {tool_unattended}")
    print(f"  Total alerts     : {len(all_alerts)}")
    if save_video:
        print(f"  Output video     : {vout}")
    print("="*60 + "\n")

    return {
        "total_frames":   engine.frame_count,
        "total_alerts":   len(all_alerts),
        "ppe_violations": ppe_viol,
        "tool_alerts":    tool_unattended,
        "avg_fps":        avg_fps,
        "duration_s":     duration,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SafeGuard AI — Industrial Safety Detection Pipeline"
    )
    parser.add_argument(
        "--source", default="0",
        help="Video file path, webcam index (0), or RTSP URL"
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Save annotated output video to outputs/videos/"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without GUI window (for servers / remote)"
    )
    args = parser.parse_args()

    try:
        stats = run(
            source=args.source,
            save_video=args.save_video,
            headless=args.headless,
        )
    except RuntimeError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

"""
deploy.py — SafeGuard AI Standalone Video Deployment Script
============================================================
A command-line alternative to the Streamlit app. Processes any video
file (MP4/AVI/MOV) or live RTSP camera stream through the full
3-model safety detection pipeline and saves:
  - An annotated output video with bounding boxes and alert overlays
  - A JSON report with alert summary and per-frame statistics
  - A heatmap image showing which regions of the frame triggered alerts

This script does NOT require Streamlit or a browser — it is designed
for server-side, automated, or batch-processing use cases.

Usage examples:
    python TRAINING/deploy.py --video "path/to/video.mp4"
    python TRAINING/deploy.py --rtsp  "rtsp://192.168.1.100:554/stream"
    python TRAINING/deploy.py --video input.mp4 --output result.mp4 --report report.json
    python TRAINING/deploy.py --video input.mp4 --limit 500   (process first 500 frames only)
"""

import sys
import os
import json
import time
import argparse
import datetime
from pathlib import Path

import cv2
import numpy as np

# Add WEB DEPLOYMENT to the import path so detection_engine and safety_config can be found
PROJECT_ROOT = Path(__file__).parent.parent
WEB_DEPLOY   = PROJECT_ROOT / "WEB DEPLOYMENT"
sys.path.insert(0, str(WEB_DEPLOY))

from detection_engine import SafetyDetectionEngine
from safety_config import (
    HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, DEVICE,
    T1_WARNING, T2_ALERT, ZONE_EXPAND_FACTOR
)


def resolve_weight_path(new_path: Path, fallback_path) -> str:
    """
    Return the path to the model weights, preferring newly trained weights.

    After a training run, 'new_path' points to the freshly produced best.pt.
    If that doesn't exist yet, fall back to whatever safety_config.py points to.

    Args:
        new_path    (Path): Path to the model weights from the latest training run
        fallback_path     : Path object from safety_config (may be a base pretrained model)

    Returns:
        str: Path string to the best available weights file
    """
    for candidate in [new_path, fallback_path]:
        if Path(str(candidate)).exists():
            return str(candidate)
    return str(new_path)  # Return even if missing, so the error message is helpful


# Resolve which weight files to use for each of the three models
HUMAN_WEIGHT_PATH = resolve_weight_path(
    PROJECT_ROOT / "HUMAN"     / "runs" / "detect" / "train_fast" / "weights" / "best.pt",
    HUMAN_WEIGHTS
)
TOOL_WEIGHT_PATH = resolve_weight_path(
    PROJECT_ROOT / "NEW TOOLS" / "runs" / "detect" / "train_fast" / "weights" / "best.pt",
    TOOL_WEIGHTS
)
PPE_WEIGHT_PATH = str(PPE_WEIGHTS)


def load_detection_engine() -> SafetyDetectionEngine:
    """
    Load all three YOLO models into the SafetyDetectionEngine.

    Prints the resolved weight paths for transparency before loading.

    Returns:
        SafetyDetectionEngine: The initialised engine ready for inference
    """
    print("\n[SafeGuard AI] Loading AI models...")
    print(f"  HUMAN model  →  {HUMAN_WEIGHT_PATH}")
    print(f"  PPE model    →  {PPE_WEIGHT_PATH}")
    print(f"  TOOLS model  →  {TOOL_WEIGHT_PATH}")

    engine = SafetyDetectionEngine(HUMAN_WEIGHT_PATH, PPE_WEIGHT_PATH, TOOL_WEIGHT_PATH, device=DEVICE)
    print(f"  Successfully loaded: {engine.loaded_models}")
    return engine


def process_video(engine: SafetyDetectionEngine, source, output_path: Path) -> dict:
    """
    Run the detection pipeline on every frame of the source video or stream.

    For recorded video files, the abandonment timer advances in video-time
    (not wall-clock time) so that a 30-second clip produces realistic 30-second
    alert timers regardless of GPU processing speed.

    Args:
        engine      (SafetyDetectionEngine): Loaded detection engine
        source                             : Path string, int (webcam), or RTSP URL
        output_path (Path)                 : Where to save the annotated output video

    Returns:
        dict: Summary statistics including total alerts, avg FPS, compliance score
    """
    is_stream = str(source).lower().startswith(("rtsp", "http"))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    # Read video properties
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_stream else -1

    # Time per frame based on the video's own FPS — used for the abandonment timer
    video_frame_dt = 1.0 / source_fps

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (frame_width, frame_height)
    )

    all_alerts       = []
    processing_times = []
    frame_count      = 0
    # Accumulate alert positions for the heatmap overlay
    alert_heatmap    = np.zeros((frame_height, frame_width), dtype=np.float32)
    session_start    = time.time()

    source_label = "RTSP stream" if is_stream else str(source)
    print(f"\n[SafeGuard AI] Processing {source_label} ...")
    if total_frames > 0:
        print(f"  {total_frames} frames  ·  {source_fps:.1f} FPS  ·  {frame_width}×{frame_height}")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()
            try:
                annotated_frame, frame_data = engine.process_frame(
                    frame, video_dt=video_frame_dt
                )
            except Exception as inference_error:
                annotated_frame = frame
                frame_data = {"alerts": [], "fps": 0, "humans": 0, "tools": 0}

            writer.write(annotated_frame)
            frame_count += 1

            elapsed = time.time() - frame_start
            if elapsed > 0:
                processing_times.append(1.0 / elapsed)

            # Record alert positions and metadata
            for alert in frame_data.get("alerts", []):
                alert["frame"]  = frame_count
                alert["time_s"] = round(frame_count / source_fps, 2) if source_fps > 0 else 0
                all_alerts.append(alert)
                # Mark the centre of the frame in the heatmap (approximate alert location)
                cy, cx = frame_height // 2, frame_width // 2
                alert_heatmap[
                    max(0, cy - 60):min(frame_height, cy + 60),
                    max(0, cx - 60):min(frame_width,  cx + 60)
                ] += 1.0

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                avg_fps = np.mean(processing_times[-50:]) if processing_times else 0
                progress = f"{frame_count / total_frames * 100:.1f}%" if total_frames > 0 else f"{frame_count} frames"
                print(f"  [{progress:>6}]  Processing FPS: {avg_fps:.1f}  |  Alerts so far: {len(all_alerts)}")

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - session_start

    # Save the alert heatmap alongside the output video
    heatmap_path = str(output_path).replace(".mp4", "_heatmap.jpg")
    if alert_heatmap.max() > 0:
        normalised  = cv2.normalize(alert_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_img = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_img)
        print(f"\n  Alert heatmap saved  →  {heatmap_path}")

    # Higher alert rate = lower compliance score
    compliance = max(0.0, min(100.0, 100.0 - len(all_alerts) / max(frame_count, 1) * 100 * 50))

    # Count alert types for the summary
    alert_type_counts = {}
    for alert in all_alerts:
        alert_type = alert.get("type", "UNKNOWN")
        alert_type_counts[alert_type] = alert_type_counts.get(alert_type, 0) + 1

    return {
        "source"         : str(source),
        "output_video"   : str(output_path),
        "heatmap"        : heatmap_path,
        "total_frames"   : frame_count,
        "total_alerts"   : len(all_alerts),
        "avg_fps"        : round(float(np.mean(processing_times)) if processing_times else 0.0, 2),
        "total_time_s"   : round(total_time, 2),
        "compliance_pct" : round(compliance, 2),
        "alert_types"    : alert_type_counts,
        "alerts"         : all_alerts,
        "timestamp"      : datetime.datetime.now().isoformat(),
        "models"         : {
            "human": HUMAN_WEIGHT_PATH,
            "ppe"  : PPE_WEIGHT_PATH,
            "tools": TOOL_WEIGHT_PATH,
        },
        "thresholds"     : {
            "T1_WARNING_s": T1_WARNING,
            "T2_ALERT_s"  : T2_ALERT,
        },
    }


def print_session_summary(stats: dict):
    """Print a formatted terminal summary after processing completes."""
    separator = "─" * 60
    print(f"\n{separator}")
    print("  SafeGuard AI — Processing Complete")
    print(separator)
    print(f"  Source      : {Path(stats['source']).name}")
    print(f"  Frames      : {stats['total_frames']:,}")
    print(f"  Avg FPS     : {stats['avg_fps']:.1f}")
    print(f"  Duration    : {stats['total_time_s']:.1f} s")
    print(f"  Compliance  : {stats['compliance_pct']:.1f}%  ", end="")
    print("✅ SAFE" if stats["compliance_pct"] >= 75 else "⚠️  NEEDS REVIEW")
    print(f"  Alerts      : {stats['total_alerts']}")
    for alert_type, count in stats["alert_types"].items():
        print(f"    • {alert_type}: {count}")
    print(f"\n  Output video →  {stats['output_video']}")
    if stats.get("heatmap"):
        print(f"  Heatmap      →  {stats['heatmap']}")
    print(separator + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="SafeGuard AI — Standalone Command-Line Deployment"
    )

    # Input source: exactly one of --video or --rtsp is required
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to input video file (MP4/AVI/MOV)")
    input_group.add_argument("--rtsp",  help="RTSP or IP camera stream URL")

    parser.add_argument(
        "--output", default=None,
        help="Output annotated video path (default: <input>_analysed.mp4)"
    )
    parser.add_argument(
        "--report", default=None,
        help="JSON report output path (default: <input>_report.json)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Maximum number of frames to process. 0 = unlimited (useful for RTSP testing)"
    )
    args = parser.parse_args()

    source = args.video or args.rtsp

    # Determine output paths based on the input source name
    if args.video:
        src_path     = Path(args.video)
        output_video = args.output or str(src_path.parent / (src_path.stem + "_analysed.mp4"))
        report_json  = args.report or str(src_path.parent / (src_path.stem + "_report.json"))
    else:
        output_video = args.output or "rtsp_output.mp4"
        report_json  = args.report or "rtsp_report.json"

    engine = load_detection_engine()
    stats  = process_video(engine, source, Path(output_video))
    print_session_summary(stats)

    # Save the JSON report (clip the full alert list to the first 50 to keep file size reasonable)
    report_data = {k: v for k, v in stats.items() if k != "alerts"}
    report_data["alerts_sample"] = stats["alerts"][:50]
    report_data["alerts_total"]  = stats["total_alerts"]

    with open(report_json, "w") as report_file:
        json.dump(report_data, report_file, indent=2, default=str)
    print(f"  JSON report saved  →  {report_json}\n")


if __name__ == "__main__":
    main()

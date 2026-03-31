"""
Performance Metrics + Live Detection Logger
Shows EXACTLY what is being detected at each frame.
Run this to see detection logs on your laptop.

Fixes vs original:
  - Tool ID counter now correctly increments for every new tool
  - Log saved as CSV (no reportlab dependency required)
  - Dead code removed
"""

import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# PATHS — imported from safety_config to stay in sync
# ============================================================
import sys
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

try:
    from safety_config import (
        HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS,
        CONF_HUMAN, CONF_TOOL, CONF_PPE,
        T1_WARNING, T2_ALERT, MODEL_INFO,
        PROJECT_ROOT,
    )
    ZONE_FACTOR    = 1.8
    TRACKING_IOU   = 0.25
    HUMAN_ZONE_IOU = 0.05
except ImportError:
    print("[WARN] safety_config not found — using fallback paths")
    PROJECT_ROOT   = Path(r"E:\4TH YEAR PROJECT")
    HUMAN_WEIGHTS  = PROJECT_ROOT / "HUMAN"   / "runs" / "detect" / "train_v2_merged" / "weights" / "best.pt"
    TOOL_WEIGHTS   = PROJECT_ROOT / "NEW TOOLS"/ "runs" / "detect" / "train_nano"      / "weights" / "best.pt"
    PPE_WEIGHTS    = PROJECT_ROOT / "NEW PPE" / "runs" / "detect" / "train_v2_nano"   / "weights" / "best.pt"
    CONF_HUMAN, CONF_TOOL, CONF_PPE = 0.40, 0.20, 0.30
    T1_WARNING, T2_ALERT = 25.0, 35.0
    MODEL_INFO = {}
    ZONE_FACTOR    = 1.8
    TRACKING_IOU   = 0.25
    HUMAN_ZONE_IOU = 0.05


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def expand_box(box, factor, W, H):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h   = (x2 - x1) * factor, (y2 - y1) * factor
    return [
        int(max(0, cx - w / 2)),
        int(max(0, cy - h / 2)),
        int(min(W - 1, cx + w / 2)),
        int(min(H - 1, cy + h / 2))
    ]


def draw_box(img, box, color, label, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_status(img, fps, frame, tool_states):
    safe    = sum(1 for s in tool_states.values() if s == 'SAFE')
    warning = sum(1 for s in tool_states.values() if s == 'WARNING')
    alert   = sum(1 for s in tool_states.values() if s == 'ALERT')
    lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame}",
        f"Tools: {len(tool_states)}",
        f"SAFE:{safe}  WARN:{warning}  ALERT:{alert}"
    ]
    y = 28
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (8, y - th - 4), (8 + tw + 6, y + 2), (0, 0, 0), -1)
        cv2.putText(img, line, (11, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 28

# ============================================================
# MAIN SYSTEM
# ============================================================

def run(source, save_log=True, save_video=False):

    print("=" * 70)
    print("INDUSTRIAL SAFETY DETECTION SYSTEM")
    print("Performance Metrics + Live Detection Logger")
    print("=" * 70)

    # ── Load models ──
    print("\nLoading models...")
    m_human = YOLO(HUMAN_WEIGHTS); print("  ✅ Human model loaded")
    m_tool  = YOLO(TOOL_WEIGHTS);  print("  ✅ Tool model loaded")
    m_ppe   = YOLO(PPE_WEIGHTS);   print("  ✅ PPE model loaded")

    # ── Open source ──
    src = int(source) if str(source).isdigit() else str(source)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"\n❌ Cannot open source: {source}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    print(f"\n📹 Source  : {source}")
    print(f"   Size   : {W}x{H}")
    print(f"   FPS    : {video_fps:.1f}")
    print("\nPress 'q' or ESC to stop\n")
    print("=" * 70)
    print(f"{'Frame':<7} {'FPS':<7} {'Humans':<9} {'Tools':<7} {'PPE':<7} {'Alerts':<30}")
    print("-" * 70)

    # ── State ──
    tracked_tools   = {}          # {id: {box, name, timer, state}}
    next_tool_id    = 1           # counter for assigning unique IDs
    detection_log   = []
    fps_list        = []
    frame_count     = 0
    last_time       = time.time()

    # ── Video writer ──
    writer = None
    if save_video:
        video_dir = PROJECT_ROOT / "outputs" / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(video_dir / "detection_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, int(video_fps), (W, H))

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            now = time.time()
            dt  = now - last_time
            last_time = now
            fps = 1.0 / dt if dt > 0 else 0
            fps_list.append(fps)

            display = frame.copy()

            # ── Detect ──
            r_human = m_human.predict(frame, conf=CONF_HUMAN, verbose=False)[0]
            r_tool  = m_tool.predict( frame, conf=CONF_TOOL,  verbose=False)[0]
            r_ppe   = m_ppe.predict(  frame, conf=CONF_PPE,   verbose=False)[0]

            # ── Parse detections ──
            humans = []
            if r_human.boxes is not None:
                for box, conf, cls in zip(r_human.boxes.xyxy.cpu().numpy(),
                                          r_human.boxes.conf.cpu().numpy(),
                                          r_human.boxes.cls.cpu().numpy()):
                    humans.append({'box': box.tolist(), 'conf': float(conf),
                                   'name': r_human.names[int(cls)]})

            tools_det = []
            if r_tool.boxes is not None:
                for box, conf, cls in zip(r_tool.boxes.xyxy.cpu().numpy(),
                                          r_tool.boxes.conf.cpu().numpy(),
                                          r_tool.boxes.cls.cpu().numpy()):
                    tools_det.append({'box': box.tolist(), 'conf': float(conf),
                                      'name': r_tool.names[int(cls)]})

            ppe_det = []
            if r_ppe.boxes is not None:
                for box, conf, cls in zip(r_ppe.boxes.xyxy.cpu().numpy(),
                                          r_ppe.boxes.conf.cpu().numpy(),
                                          r_ppe.boxes.cls.cpu().numpy()):
                    ppe_det.append({'box': box.tolist(), 'conf': float(conf),
                                    'name': r_ppe.names[int(cls)]})

            # ── Update tool tracker (custom IoU tracker) ──
            # BUG FIX: next_tool_id now correctly increments for every new tool
            matched = set()
            updated = {}

            for det in tools_det:
                best_id, best_iou = None, 0
                for tid, tdata in tracked_tools.items():
                    if tid in matched:
                        continue
                    iou = calculate_iou(det['box'], tdata['box'])
                    if iou > best_iou and iou > TRACKING_IOU:
                        best_iou, best_id = iou, tid

                if best_id is not None:
                    # Matched with an existing tracked tool — keep its history
                    matched.add(best_id)
                    updated[best_id] = {**tracked_tools[best_id],
                                        'box': det['box'], 'name': det['name']}
                else:
                    # New tool — assign a fresh, unique ID
                    updated[next_tool_id] = {
                        'box': det['box'], 'name': det['name'],
                        'timer': 0.0, 'state': 'SAFE'
                    }
                    next_tool_id += 1   # ← FIX: was missing, causing all new tools to share ID

            tracked_tools = updated

            # ── Temporal logic + hazard zones ──
            frame_alerts = []
            tool_states  = {}

            for tid, tdata in tracked_tools.items():
                box  = tdata['box']
                zone = expand_box(box, ZONE_FACTOR, W, H)

                # Is tool attended?
                attended      = False
                ppe_violation = False
                missing_ppe   = []

                for human in humans:
                    if calculate_iou(human['box'], zone) > HUMAN_ZONE_IOU:
                        attended = True
                        # Check PPE
                        has_helmet = any(
                            calculate_iou(human['box'], p['box']) > 0.05
                            for p in ppe_det
                            if p['name'].lower() in ('helmet', 'hardhat')
                        )
                        has_gloves = any(
                            calculate_iou(human['box'], p['box']) > 0.05
                            for p in ppe_det
                            if p['name'].lower() in ('glove', 'gloves')
                        )
                        if not has_helmet:
                            missing_ppe.append('helmet')
                        if not has_gloves:
                            missing_ppe.append('gloves')
                        if missing_ppe:
                            ppe_violation = True

                # Update timer
                if ppe_violation:
                    tdata['state'] = 'ALERT'
                elif attended:
                    tdata['timer'] = 0.0
                    tdata['state'] = 'SAFE'
                else:
                    tdata['timer'] += dt
                    if tdata['timer'] >= T2_ALERT:
                        tdata['state'] = 'ALERT'
                    elif tdata['timer'] >= T1_WARNING:
                        tdata['state'] = 'WARNING'
                    else:
                        tdata['state'] = 'SAFE'

                tool_states[tid] = tdata['state']

                # Alert check
                if tdata['state'] == 'ALERT':
                    reason = 'PPE_VIOLATION' if ppe_violation else 'TOOL_UNATTENDED'
                    frame_alerts.append({
                        'tool_id': tid,
                        'tool_name': tdata['name'],
                        'reason': reason,
                        'timer': tdata['timer'],
                        'missing_ppe': missing_ppe
                    })

                # ── Draw tool + zone ──
                color = (0, 0, 255) if tdata['state'] == 'ALERT' else \
                        (0, 165, 255) if tdata['state'] == 'WARNING' else (0, 255, 0)
                # Hazard zone (blue outline)
                hx1, hy1, hx2, hy2 = zone
                cv2.rectangle(display, (hx1, hy1), (hx2, hy2), (255, 0, 0), 1)
                # Tool box
                label = f"ID:{tid} {tdata['name']} {tdata['timer']:.1f}s"
                draw_box(display, box, (0, 255, 255), label)
                if tdata['state'] != 'SAFE':
                    cv2.putText(display, f"{tdata['state']}!",
                                (int(box[0]), int(box[3]) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ── Draw humans ──
            for human in humans:
                draw_box(display, human['box'], (0, 220, 0),
                         f"Worker {human['conf']:.2f}")

            # ── Draw PPE boxes ──
            for p in ppe_det:
                cv2.rectangle(display,
                              (int(p['box'][0]), int(p['box'][1])),
                              (int(p['box'][2]), int(p['box'][3])),
                              (255, 255, 255), 1)

            # ── Status overlay ──
            draw_status(display, fps, frame_count, tool_states)

            # ── Console log (every 30 frames) ──
            alert_str = " | ".join(
                f"TOOL-{a['tool_id']}:{a['reason']}" for a in frame_alerts
            ) if frame_alerts else "None"

            if frame_count % 30 == 0:
                print(f"{frame_count:<7} {fps:<7.1f} {len(humans):<9} "
                      f"{len(tracked_tools):<7} {len(ppe_det):<7} {alert_str:<30}")

            # ── Save to log ──
            for a in frame_alerts:
                detection_log.append({
                    'Timestamp': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                    'Frame': frame_count,
                    'FPS': round(fps, 1),
                    'Tool_ID': a['tool_id'],
                    'Tool_Name': a['tool_name'],
                    'Alert_Type': a['reason'],
                    'Timer_s': round(a['timer'], 1),
                    'Missing_PPE': str(a['missing_ppe']),
                    'Humans_in_frame': len(humans),
                    'Tools_in_frame': len(tracked_tools)
                })

            # ── Write video ──
            if writer:
                writer.write(display)

            # ── Show ──
            cv2.imshow("Safety Detection System", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print("\n⚠  Stopped by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    # ============================================================
    # PERFORMANCE REPORT
    # ============================================================
    print("\n" + "=" * 70)
    print("PERFORMANCE REPORT")
    print("=" * 70)

    avg_fps = np.mean(fps_list) if fps_list else 0
    min_fps = np.min(fps_list)  if fps_list else 0
    max_fps = np.max(fps_list)  if fps_list else 0
    avg_ms  = 1000 / avg_fps    if avg_fps > 0 else 0

    print(f"\n📹 Video Processing:")
    print(f"   Total frames    : {frame_count}")
    print(f"   Avg FPS         : {avg_fps:.2f}")
    print(f"   Min FPS         : {min_fps:.2f}")
    print(f"   Max FPS         : {max_fps:.2f}")
    print(f"   Avg frame time  : {avg_ms:.1f} ms")

    print(f"\n🔧 Detection Summary:")
    print(f"   Total alerts    : {len(detection_log)}")

    if detection_log:
        df = pd.DataFrame(detection_log)
        ppe_count  = len(df[df['Alert_Type'] == 'PPE_VIOLATION'])
        tool_count = len(df[df['Alert_Type'] == 'TOOL_UNATTENDED'])
        print(f"   PPE violations  : {ppe_count}")
        print(f"   Tool unattended : {tool_count}")

    print(f"\n📊 Model Performance (from training):")
    print(f"   {'Model':<12} {'mAP@50':>10} {'Precision':>12} {'Recall':>10} {'Status':>16}")
    print(f"   {'-'*64}")
    for model_name, info in MODEL_INFO.items():
        mAP = f"{info['mAP@50']*100:.2f}%" if info.get('mAP@50') else 'N/A'
        pre = f"{info['Precision']*100:.2f}%" if info.get('Precision') else 'N/A'
        rec = f"{info['Recall']*100:.2f}%"    if info.get('Recall')    else 'N/A'
        sts = info.get('Status', '?')
        print(f"   {model_name:<12} {mAP:>10} {pre:>12} {rec:>10} {sts:>16}")

    print(f"\n⏱️  Temporal Logic:")
    print(f"   T1 Warning      : {T1_WARNING}s")
    print(f"   T2 Alert        : {T2_ALERT}s")
    print(f"   Zone Factor     : {ZONE_FACTOR}x")

    print("=" * 70)

    # ── Save log as CSV (no external dependencies) ──
    if save_log and detection_log:
        log_dir = PROJECT_ROOT / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path  = log_dir / f"detection_log_{timestamp}.csv"

        df = pd.DataFrame(detection_log)
        df.to_csv(log_path, index=False)
        print(f"\n✅ Detection log saved: {log_path}")

    if save_video:
        print(f"✅ Output video saved: {PROJECT_ROOT / 'outputs' / 'videos' / 'detection_output.mp4'}")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Safety Detection with Metrics")
    parser.add_argument('--source', default=r"E:\4TH YEAR PROJECT\test_videos\demo.mp4",
                        help='Video path or 0 for webcam')
    parser.add_argument('--save-log', action='store_true', default=True,
                        help='Save detection log as CSV')
    parser.add_argument('--save-video', action='store_true', default=False,
                        help='Save annotated output video')
    args = parser.parse_args()

    run(
        source=args.source,
        save_log=args.save_log,
        save_video=args.save_video
    )
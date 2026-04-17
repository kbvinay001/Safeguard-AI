"""
detection_engine.py — SafeGuard AI Core Detection Engine
=========================================================
Implements the three-model inference pipeline that processes each
video frame through human detection, PPE compliance checking, and
tool detection, then fuses the results using IoU-based spatial matching.

Key components:
  SafetyDetectionEngine   — loads all three YOLO models and runs per-frame inference
  ToolTracker             — IoU-based tracker that assigns persistent IDs to tools
  PPEChecker              — checks whether a detected worker is wearing required PPE
  Helper functions        — IoU calculation, box expansion, drawing utilities

This module is imported by:
  streamlit_app.py, api_server.py, integrated_safety_system.py,
  run_with_metrics.py, and deploy.py

It relies on safety_config.py for all threshold and class-name constants.
"""

import time
import gc
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import pandas as pd

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Import all configuration constants from safety_config.
# Using wildcard import here is intentional — the config file is the
# single source of truth and all its names are used throughout this module.
try:
    from safety_config import *
except ImportError:
    print("Warning: Could not import safety_config.py — check file location.")


# ==============================================================================
# SafetyDetectionEngine
# ==============================================================================

class SafetyDetectionEngine:
    """
    The core engine that coordinates all three YOLOv11n models for each frame.

    Models load independently — if one weight file is missing the engine
    continues with graceful degradation rather than crashing entirely.
    This allows the app to be launched and tested even while models are
    still being trained.

    Workflow per frame (process_frame):
      1. Run Human model  → detect all workers
      2. Run Tool model   → detect all tools, assign persistent IDs via ToolTracker
      3. Run PPE model    → detect all PPE items
      4. Spatial fusion   → for each tool, expand a hazard zone and check whether
                            any worker is inside it (IoU matching)
      5. Per-worker PPE check → for each worker near a tool, check PPE compliance
      6. Timer update     → advance FSM timers to trigger WARNING / ALERT events
      7. Annotation       → draw all boxes, labels, and overlays onto the frame
    """

    def __init__(self, human_weights, ppe_weights, tool_weights, device=None):
        """
        Load the three detection models.

        Args:
            human_weights : Path or string to the HUMAN model's best.pt
            ppe_weights   : Path or string to the PPE model's best.pt
            tool_weights  : Path or string to the TOOLS model's best.pt
            device        : Inference device — 0 for first GPU, "cpu" for CPU.
                            If None, reads DEVICE from safety_config (which
                            itself auto-detects CUDA availability at import time).
        """
        # Resolve device: use safety_config auto-detection if not explicitly set
        if device is None:
            try:
                from safety_config import DEVICE as _cfg_device
                device = _cfg_device
            except Exception:
                device = "cpu"
        self.device = device
        self.loaded_models = []

        print(f"Loading SafeGuard AI models... (device={self.device})")

        self.model_human = self._load_model(human_weights, "HUMAN")
        self.model_ppe   = self._load_model(ppe_weights,   "PPE")
        self.model_tool  = self._load_model(tool_weights,  "TOOLS")

        loaded_count = len(self.loaded_models)
        if loaded_count == 3:
            print("All 3 models loaded successfully.")
        elif loaded_count == 0:
            print("ERROR: No models could be loaded — check weight paths in safety_config.py")
        else:
            print(f"Warning: {loaded_count}/3 models loaded: {', '.join(self.loaded_models)}")

        # Tracking and compliance sub-systems
        self.tool_tracker = ToolTracker()
        self.ppe_checker  = PPEChecker(REQUIRED_PPE)

        # FPS tracking
        self.frame_count = 0
        self.last_time   = time.time()
        self.fps         = 0

        # Adaptive frame-skipping state
        # Tracks the last frame's processing time (ms) so we can decide
        # whether to skip inference on the next frame to stay responsive.
        self._last_frame_ms   = 0.0      # Wall-clock ms spent on previous frame
        self._skip_next_frame = False    # If True, reuse previous detections
        self._prev_detections = {        # Cache of last good detection outputs
            "humans": [], "tools": [], "ppe": []
        }
        self._prev_active_tools = {}     # Cache of last good tracker state

    def _load_model(self, weight_path, model_name: str):
        """
        Attempt to load one YOLO model; return None if loading fails.

        Args:
            weight_path : File path to the model weights (.pt file)
            model_name  : Human-readable name for log messages

        Returns:
            YOLO model instance, or None on failure
        """
        try:
            if weight_path is None:
                raise ValueError("weight_path is None")
            model = YOLO(str(weight_path))
            print(f"  {model_name:8} loaded  ←  {Path(str(weight_path)).name}")
            self.loaded_models.append(model_name)
            return model
        except Exception as load_error:
            print(f"  {model_name:8} FAILED  ({load_error})")
            return None

    def _should_skip_frame(self) -> bool:
        """
        Return True if the previous frame took longer than FRAME_SKIP_THRESHOLD_MS
        to process, indicating the GPU pipeline is under load.

        IMPORTANT: On CPU, every frame takes well over 66ms, so skipping there
        would permanently reuse the very first (empty) detection cache, causing
        zero detections for the entire video.  We therefore only enable skipping
        when running on a real CUDA GPU (device != "cpu").
        """
        try:
            # Never skip on CPU — inference is slow by design, not by overload
            if str(self.device) == "cpu":
                return False
            from safety_config import FRAME_SKIP_THRESHOLD_MS
            return self._last_frame_ms > FRAME_SKIP_THRESHOLD_MS
        except Exception:
            return False  # If config unavailable, never skip

    def process_frame(self, frame, video_dt=None):
        """
        Run the full detection and analysis pipeline on a single video frame.

        Args:
            frame     (np.ndarray): BGR image as loaded by cv2
            video_dt  (float | None): Seconds per frame based on the VIDEO's own FPS.
                        Pass this when processing a recorded file so that the
                        abandonment timer advances in video-time, not wall-clock time.
                        Leave None when processing a live RTSP stream.

        Returns:
            tuple:
                annotated (np.ndarray) — the frame with all bounding boxes drawn
                data (dict) — {frame, fps, humans, tools, alerts, tracked_tools}
        """
        frame_start_time = time.time()
        self.frame_count += 1
        current_time = time.time()
        wall_dt      = current_time - self.last_time
        self.last_time = current_time
        if wall_dt > 0:
            self.fps = 1.0 / wall_dt

        # --- Periodic memory management ---
        try:
            from safety_config import GC_INTERVAL_FRAMES
            _gc_interval = GC_INTERVAL_FRAMES
        except Exception:
            _gc_interval = 50

        if self.frame_count % _gc_interval == 0:
            gc.collect()

        if self.frame_count % 100 == 0 and _TORCH_AVAILABLE:
            try:
                _torch.cuda.empty_cache()
            except Exception:
                pass  # Non-critical if CUDA not available

        # Use video-time for the abandonment timer when processing recordings.
        # This prevents a 60 fps video being processed at 5 fps from showing
        # 12× inflated timers.
        timer_dt = video_dt if video_dt is not None else wall_dt

        frame_height, frame_width = frame.shape[:2]
        annotated = frame.copy()

        # --- Top-level crash protection ---
        # If any stage of the pipeline fails on this frame, return the raw
        # frame with an error overlay rather than crashing the entire app.
        try:
            def run_inference(model, confidence_threshold, allowed_classes, input_size=IMAGE_SIZE):
                """Run model.predict on the frame and return a list of detection dicts."""
                if model is None:
                    return []
                # Use FP16 only on a real CUDA GPU — never on CPU (causes crash)
                use_half = (
                    _TORCH_AVAILABLE
                    and str(self.device) != "cpu"
                    and _torch.cuda.is_available()
                )
                try:
                    result = model.predict(
                        frame,
                        conf   = confidence_threshold,
                        iou    = IOU_THRESHOLD,
                        imgsz  = input_size,
                        verbose= False,
                        device = self.device,
                        half   = use_half,
                    )[0]
                    return extract_detections(result, allowed_classes)
                except Exception as inference_error:
                    print(f"  Inference error: {inference_error}")
                    return []

            # --- Adaptive frame-skipping ---
            # If the previous frame was slow, reuse its detection results on
            # this frame (still annotate the new image) to keep FPS up.
            if self._should_skip_frame():
                detected_humans    = self._prev_detections["humans"]
                detected_tools     = self._prev_detections["tools"]
                detected_ppe_items = self._prev_detections["ppe"]
                active_tools       = self._prev_active_tools
            else:
                # Run all three models on the current frame
                detected_humans    = run_inference(self.model_human, CONF_HUMAN, {HUMAN_CLASS}, 480)
                detected_tools     = run_inference(self.model_tool,  CONF_TOOL,  TOOL_CLASSES,  640)
                detected_ppe_items = run_inference(self.model_ppe,   CONF_PPE,   PPE_POSITIVE | PPE_NEGATIVE, 640)

                # Update the tool tracker to assign persistent IDs across frames
                active_tools = self.tool_tracker.update(detected_tools, timer_dt)

                # Cache this frame's results for potential reuse next frame
                self._prev_detections = {
                    "humans": detected_humans,
                    "tools" : detected_tools,
                    "ppe"   : detected_ppe_items,
                }
                self._prev_active_tools = active_tools

            alerts = []
            # Track which workers are inside at least one hazard zone so
            # draw_human can show PPE warnings ONLY for those workers.
            workers_in_hazard_zone = set()   # indices into detected_humans

            for tool_id, tool_data in active_tools.items():
                tool_box    = tool_data["box"]
                hazard_zone = expand_box(tool_box, ZONE_EXPAND_FACTOR, frame_width, frame_height)

                tool_attended   = False
                has_ppe_violation = False

                # Check whether each detected worker is inside this tool's hazard zone
                for w_idx, worker in enumerate(detected_humans):
                    if calculate_iou(worker["box"], hazard_zone) > HUMAN_ZONE_IOU:
                        tool_attended = True
                        workers_in_hazard_zone.add(w_idx)   # mark as zone-worker
                        is_compliant, missing_ppe, _ = self.ppe_checker.check_compliance(
                            worker["box"], detected_ppe_items
                        )
                        if not is_compliant:
                            has_ppe_violation = True
                            alerts.append({
                                "type"       : "PPE_VIOLATION",
                                "tool_id"    : tool_id,
                                "tool_name"  : tool_data["name"],
                                "missing_ppe": missing_ppe,
                            })

                self.tool_tracker.update_timer(tool_id, tool_attended, timer_dt, has_ppe_violation)

                # If the FSM has escalated to ALERT, record the tool abandonment event
                if tool_data["state"] == "ALERT":
                    alerts.append({
                        "type"     : "TOOL_UNATTENDED",
                        "tool_id"  : tool_id,
                        "tool_name": tool_data["name"],
                        "timer"    : tool_data["timer"],
                    })

                annotated = draw_tool(annotated, tool_id, tool_data, hazard_zone)

            for w_idx, worker in enumerate(detected_humans):
                in_zone = w_idx in workers_in_hazard_zone
                annotated = draw_human(annotated, worker, detected_ppe_items,
                                       self.ppe_checker, in_hazard_zone=in_zone)

            annotated = draw_status(annotated, self.fps, self.frame_count, active_tools)

        except Exception as pipeline_error:
            # Single-frame failure — log and return the raw frame with an overlay
            print(f"  Pipeline error on frame {self.frame_count}: {pipeline_error}")
            draw_text_with_background(
                annotated,
                f"Frame error: {type(pipeline_error).__name__}",
                (10, 30), (0, 0, 180), (255, 255, 255)
            )
            alerts      = []
            active_tools = self._prev_active_tools

        # --- Record how long this frame took (for next frame's skip decision) ---
        self._last_frame_ms = (time.time() - frame_start_time) * 1000.0

        return annotated, {
            "frame"        : self.frame_count,
            "fps"          : self.fps,
            "humans"       : len(detected_humans) if 'detected_humans' in dir() else 0,
            "tools"        : len(active_tools),
            "alerts"       : alerts,
            "tracked_tools": active_tools,
        }


# ==============================================================================
# ToolTracker
# ==============================================================================

class ToolTracker:
    """
    Cross-frame IoU-based tracker for detected tools.

    Each new detection is matched to the closest existing tracked tool by
    bounding-box IoU. Matched tools retain their ID and accumulated timer;
    unmatched detections become new tracked tools with fresh IDs.
    Tools that disappear from the frame are simply removed from the active set.
    """

    def __init__(self):
        self.next_id    = 1           # Monotonically increasing ID counter
        self.active_tools = {}        # {tool_id: {box, name, timer, state}}

    def update(self, detections: list, dt: float) -> dict:
        """
        Match new detections to existing tracked tools and return the updated set.

        Args:
            detections (list[dict]): Output of extract_detections for the TOOLS model
            dt         (float):     Seconds since the last frame (unused here, used in update_timer)

        Returns:
            dict: Updated {tool_id: tool_data} mapping
        """
        matched_ids  = set()
        updated_tools = {}

        for detection in detections:
            current_box = detection["box"]
            best_match_id  = None
            best_match_iou = 0

            # Find the existing tracked tool with the highest IoU overlap
            for tid, tool_data in self.active_tools.items():
                if tid in matched_ids:
                    continue
                iou = calculate_iou(current_box, tool_data["box"])
                if iou > best_match_iou and iou > TRACKING_IOU_THRESHOLD:
                    best_match_iou = iou
                    best_match_id  = tid

            if best_match_id is not None:
                # Re-associate with the existing tracked instance — preserve history
                matched_ids.add(best_match_id)
                updated_tools[best_match_id] = {
                    **self.active_tools[best_match_id],
                    "box" : current_box,
                    "name": detection["class_name"],
                }
            else:
                # New tool detected — assign a fresh unique ID
                updated_tools[self.next_id] = {
                    "box"  : current_box,
                    "name" : detection["class_name"],
                    "timer": 0.0,
                    "state": "SAFE",
                }
                self.next_id += 1

        self.active_tools = updated_tools
        return self.active_tools

    def update_timer(self, tool_id: int, is_attended: bool, dt: float, ppe_violation: bool = False):
        """
        Advance the abandonment timer for one tracked tool and update its FSM state.

        State transitions:
          SAFE    — tool is attended by a compliant worker (timer resets to 0)
          WARNING — tool unattended for T1_WARNING seconds
          ALERT   — tool unattended for T2_ALERT seconds, OR PPE violation detected

        Args:
            tool_id       (int):   ID of the tool to update
            is_attended   (bool):  True if a worker is in the hazard zone this frame
            dt            (float): Elapsed seconds since the last frame
            ppe_violation (bool):  True if a nearby worker is missing required PPE
        """
        if tool_id not in self.active_tools:
            return

        tool = self.active_tools[tool_id]

        # PPE violation immediately escalates to ALERT regardless of timer
        if ppe_violation:
            tool["state"] = "ALERT"
            return

        if is_attended:
            # Worker is present and compliant — reset the abandonment timer
            tool["timer"] = 0.0
            tool["state"] = "SAFE"
        else:
            # Tool is unattended — advance the FSM timer
            tool["timer"] += dt
            if tool["timer"] >= T2_ALERT:
                tool["state"] = "ALERT"
            elif tool["timer"] >= T1_WARNING:
                tool["state"] = "WARNING"


# ==============================================================================
# PPEChecker
# ==============================================================================

class PPEChecker:
    """
    Checks whether a detected worker has the required Personal Protective Equipment.

    Matching is done by spatial IoU overlap between the worker's bounding box
    and each PPE item's bounding box. All class name comparisons are
    case-insensitive to handle inconsistent labelling across dataset versions.
    """

    def __init__(self, required_ppe: list):
        """
        Args:
            required_ppe (list[str]): List of PPE item names that workers must wear.
                                      Sourced from REQUIRED_PPE in safety_config.py.
        """
        # Store required items in lowercase for consistent comparison
        self.required = {item.lower() for item in required_ppe}

        # Pre-compute lowercase lookup sets for the positive / negative class lists
        # so check_compliance doesn't repeat this work on every call
        self._positive_classes = {name.lower() for name in PPE_POSITIVE}
        self._negative_classes = {name.lower() for name in PPE_NEGATIVE}

    def check_compliance(self, worker_box: list, ppe_detections: list):
        """
        Determine whether a worker has all required PPE items.

        Association strategy (two-pass, primary wins):
          1. CENTROID CONTAINMENT (primary): the PPE item's centre point lies
             inside the worker's bounding box.  This is the strongest spatial
             signal — a helmet centre inside a person box can only mean that
             the person is wearing it.
          2. IoU OVERLAP (secondary): for PPE that sits at the edge of the box
             (e.g. boots at the bottom, gloves at the sides), a bounding-box
             IoU > PPE_HUMAN_IOU is accepted as a fallback.

        Using CENTROID CONTAINMENT as primary eliminates the false 'helmet
        missing' alert that occurred when a real helmet was present but its
        bounding box overlapped a different (adjacent) worker's box more than
        the correct worker's box at low IoU values.

        The positive-class-wins rule is kept: if both 'helmet' AND 'no_helmet'
        fire on the same worker, the positive detection overrides the negative.

        Args:
            worker_box      (list): [x1, y1, x2, y2] bounding box of the worker
            ppe_detections  (list): List of PPE detection dicts from the PPE model

        Returns:
            tuple:
                is_compliant (bool)     — True if all required PPE is detected
                missing      (list[str]) — Names of required items not found
                detected     (list[str]) — Names of PPE items found near this worker
        """
        found_ppe = set()
        violated  = set()

        wx1, wy1, wx2, wy2 = worker_box

        for ppe_item in ppe_detections:
            pb = ppe_item["box"]          # [px1, py1, px2, py2]
            item_name = ppe_item["class_name"].lower()

            # --- Primary: centroid containment ---
            # The centre of the PPE bounding box must lie inside the worker box.
            px_centre = (pb[0] + pb[2]) / 2
            py_centre = (pb[1] + pb[3]) / 2
            centroid_inside = (
                wx1 <= px_centre <= wx2 and
                wy1 <= py_centre <= wy2
            )

            # --- Secondary: IoU overlap fallback ---
            iou_match = calculate_iou(worker_box, pb) > PPE_HUMAN_IOU

            if centroid_inside or iou_match:
                if item_name in self._positive_classes:
                    found_ppe.add(item_name)
                elif item_name in self._negative_classes:
                    # "no_helmet" → strip prefix to get the violated item name
                    violated_item = (
                        item_name
                        .replace("no_", "")
                        .replace("no-", "")
                        .replace("no ", "")
                    )
                    violated.add(violated_item)

        # Positive-class WINS rule:
        # If both 'helmet' and 'no_helmet' fired on the same worker,
        # the helmet IS present — discard the negative detection.
        real_violations = violated - found_ppe

        # A worker is non-compliant only if a required item is:
        #   (a) not found at all, OR (b) explicitly violated with no positive override
        missing = (self.required - found_ppe) | real_violations
        return len(missing) == 0, list(missing), list(found_ppe)


# ==============================================================================
# Helper functions
# ==============================================================================

def extract_detections(yolo_result, allowed_classes) -> list:
    """
    Convert YOLO inference output into a list of detection dictionaries.

    Class matching is done CASE-INSENSITIVELY so that training datasets
    with inconsistent capitalisation (e.g. 'Wrench' vs 'wrench') never
    silently drop real detections.

    Args:
        yolo_result    : The first element of model.predict() output
        allowed_classes: Set of class name strings to keep; None = keep all

    Returns:
        list[dict]: Each dict has keys 'box', 'confidence', 'class_name'
    """
    detections = []
    if yolo_result is None or yolo_result.boxes is None:
        return detections

    boxes       = yolo_result.boxes.xyxy.cpu().numpy()
    confidences = yolo_result.boxes.conf.cpu().numpy()
    class_ids   = yolo_result.boxes.cls.cpu().numpy().astype(int)
    class_names = yolo_result.names

    # Build a lowercase lookup set so matching is always case-insensitive.
    # This covers 'Drill' == 'drill', 'Wrench' == 'wrench', etc.
    allowed_lower = (
        {name.lower() for name in allowed_classes}
        if allowed_classes is not None else None
    )

    for i, box in enumerate(boxes):
        raw_name   = class_names[class_ids[i]] if class_names else str(class_ids[i])
        name_lower = raw_name.lower()
        if allowed_lower is None or name_lower in allowed_lower:
            detections.append({
                "box"        : box.tolist(),
                "confidence" : float(confidences[i]),
                # Normalise to lowercase so all downstream comparisons
                # (PPEChecker, ToolTracker, alerts) are consistent.
                "class_name" : name_lower,
            })

    return detections


def calculate_iou(box1: list, box2: list) -> float:
    """
    Compute the Intersection over Union (IoU) of two axis-aligned bounding boxes.

    Args:
        box1 (list): [x1, y1, x2, y2]
        box2 (list): [x1, y1, x2, y2]

    Returns:
        float: IoU in [0, 1]; returns 0 if the union area is zero
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection rectangle
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def expand_box(box: list, factor: float, frame_width: int, frame_height: int) -> list:
    """
    Expand a bounding box outward from its centre by a scale factor, clamped
    to the frame boundary.  Used to define the "hazard zone" around each tool.

    Args:
        box          (list): [x1, y1, x2, y2] original bounding box
        factor       (float): Expansion factor (e.g. 1.8 = box is 80% wider/taller)
        frame_width  (int):  Frame width in pixels (used to clamp x coordinates)
        frame_height (int):  Frame height in pixels (used to clamp y coordinates)

    Returns:
        list: [x1, y1, x2, y2] expanded box
    """
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * factor / 2
    half_h = (y2 - y1) * factor / 2

    return [
        int(max(0,                center_x - half_w)),
        int(max(0,                center_y - half_h)),
        int(min(frame_width  - 1, center_x + half_w)),
        int(min(frame_height - 1, center_y + half_h)),
    ]


def draw_text_with_background(img, text: str, position: tuple, bg_color: tuple, text_color=(0, 0, 0)):
    """
    Draw a text label with a solid coloured background rectangle.

    Args:
        img        (np.ndarray): Image to draw on (modified in place)
        text       (str):       Label text
        position   (tuple):     (x, y) bottom-left corner for the text
        bg_color   (tuple):     BGR background fill colour
        text_color (tuple):     BGR text colour (default black)

    Returns:
        np.ndarray: The annotated image
    """
    (text_width, text_height), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    x, y = position
    cv2.rectangle(img, (x, y - text_height - 6), (x + text_width + 6, y), bg_color, -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, FONT_SCALE, text_color, FONT_THICKNESS)
    return img


def draw_tool(img, tool_id: int, tool_data: dict, hazard_zone: list):
    """
    Draw a tool's bounding box, hazard zone outline, label, and alert state.

    The tool box colour is cyan regardless of state; the label and alert
    badge use the FSM state colour (green/orange/red).

    Args:
        img         (np.ndarray): Frame to annotate
        tool_id     (int):        Unique tracker ID for this tool
        tool_data   (dict):       {box, name, timer, state}
        hazard_zone (list):       Expanded hazard zone box [x1,y1,x2,y2]

    Returns:
        np.ndarray: Annotated frame
    """
    box   = tool_data["box"]
    state = tool_data["state"]
    timer = tool_data["timer"]

    state_color = (
        COLOR_ALERT   if state == "ALERT"   else
        COLOR_WARNING if state == "WARNING" else
        COLOR_SAFE
    )

    # Tool bounding box (always cyan)
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_TOOL, 2)

    # ── Hazard zone: drawn around the TOOL, not the worker ──────────────────
    # The zone is an expanded version of the tool's own bounding box.
    # Any worker whose box overlaps this zone is flagged for PPE checks.
    hx1, hy1, hx2, hy2 = map(int, hazard_zone)
    cv2.rectangle(img, (hx1, hy1), (hx2, hy2), COLOR_HAZARD_ZONE, 2)
    # Small label in the top-left corner of the zone so it is unambiguous
    zone_label = f"TOOL ZONE [{tool_data['name'].upper()}]"
    draw_text_with_background(img, zone_label, (hx1 + 4, hy1 + 18),
                              COLOR_HAZARD_ZONE, (255, 255, 255))

    # Tool label — ID, class name, and abandonment timer
    label = f"ID:{tool_id} {tool_data['name']} {timer:.1f}s"
    draw_text_with_background(img, label, (x1, y1 - 5), state_color)

    # Alert badge below the box if not in the safe state
    if state != "SAFE":
        draw_text_with_background(img, f"{state}!", (x1, y2 + 20), state_color, (255, 255, 255))

    return img


def draw_human(img, worker: dict, ppe_items: list, ppe_checker: PPEChecker,
               in_hazard_zone: bool = False):
    """
    Draw a worker's bounding box and PPE status label.

    PPE compliance is ONLY evaluated and displayed for workers who are inside
    a tool's hazard zone (in_hazard_zone=True).  Workers outside any hazard
    zone are shown with a neutral 'Worker (Monitoring)' label in white — this
    avoids false red alerts for people simply walking through the scene.

    Args:
        img            (np.ndarray): Frame to annotate
        worker         (dict):       Detection dict for the worker
        ppe_items      (list):       All PPE detection dicts in the current frame
        ppe_checker    (PPEChecker): Checker instance to assess compliance
        in_hazard_zone (bool):       True if this worker is inside a tool hazard zone

    Returns:
        np.ndarray: Annotated frame
    """
    box = worker["box"]
    x1, y1, x2, y2 = map(int, box)

    if in_hazard_zone:
        # Worker is near a tool — evaluate PPE compliance
        is_compliant, missing_items, detected_items = ppe_checker.check_compliance(box, ppe_items)

        if is_compliant:
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_SAFE, 2)
            label = "Worker  (PPE OK ✓)"
            label_color = COLOR_SAFE
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_ALERT, 2)
            # Show up to 2 missing items to keep the label concise
            label = f"Worker  (Missing: {', '.join(missing_items[:2])})"
            label_color = COLOR_ALERT
    else:
        # Worker is outside all hazard zones — no PPE check, just show presence
        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_HUMAN, 2)
        label = "Worker  (Monitoring)"
        label_color = COLOR_INFO

    draw_text_with_background(img, label, (x1, y1 - 5), label_color)
    return img


def draw_status(img, fps: float, frame_count: int, active_tools: dict):
    """
    Draw the live status overlay in the top-left corner of the frame.

    Shows current FPS, frame number, total tool count, and a breakdown
    of how many tools are SAFE / WARNING / ALERT.

    Args:
        img           (np.ndarray): Frame to annotate
        fps           (float):      Current processing speed in frames per second
        frame_count   (int):        Number of frames processed since engine start
        active_tools  (dict):       Current tool tracker state

    Returns:
        np.ndarray: Annotated frame
    """
    safe_count    = sum(1 for t in active_tools.values() if t["state"] == "SAFE")
    warning_count = sum(1 for t in active_tools.values() if t["state"] == "WARNING")
    alert_count   = sum(1 for t in active_tools.values() if t["state"] == "ALERT")

    overlay_lines = [
        f"FPS: {fps:.1f}",
        f"Frame: {frame_count}",
        f"Tools: {len(active_tools)}",
        f"Safe:{safe_count}  Warn:{warning_count}  Alert:{alert_count}",
    ]

    y_position = 30
    for line in overlay_lines:
        draw_text_with_background(img, line, (10, y_position), (0, 0, 0), (255, 255, 255))
        y_position += 30

    return img

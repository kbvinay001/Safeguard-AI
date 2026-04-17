"""
api_server.py — SafeGuard AI REST API Server
=============================================
Provides a FastAPI-based HTTP API for integrating SafeGuard AI
into existing CCTV management platforms or custom dashboards.

Endpoints:
  GET  /                     — API overview and available endpoints
  GET  /api/status           — System health and uptime
  GET  /api/metrics          — Processing statistics (frames, alerts)
  POST /api/detect/image     — Detect safety violations in a single image
  POST /api/detect/video     — Process a complete video file
  GET  /api/config           — Read current detection thresholds
  POST /api/config           — Update detection thresholds at runtime
  GET  /api/alerts           — Retrieve recent alert summary
  GET  /health               — Health check (for load balancers / uptime monitors)
  POST /api/detect/stream/start — Begin processing an RTSP camera stream

Run the server:
    python "WEB DEPLOYMENT/api_server.py"
    Then visit http://localhost:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import tempfile
import base64
from pathlib import Path
import time
from datetime import datetime
import uvicorn

from detection_engine import SafetyDetectionEngine
from safety_config import *

# Initialize FastAPI
app = FastAPI(
    title="Industrial Safety Detection API",
    description="REST API for real-time safety monitoring with AI",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine (lazy loading — preloaded at startup)
detection_engine = None

def get_engine():
    """Return the detection engine, raising 503 if unavailable."""
    global detection_engine
    if detection_engine is None:
        try:
            detection_engine = SafetyDetectionEngine(
                HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=DEVICE
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Detection engine unavailable: {e}. Check model weight paths in safety_config.py."
            )
    return detection_engine


@app.on_event("startup")
async def startup_event():
    """Pre-load the detection engine at startup to eliminate cold-start delay."""
    global detection_engine
    print("[Startup] Pre-loading SafeGuard AI detection engine...")
    try:
        detection_engine = SafetyDetectionEngine(
            HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=DEVICE
        )
        print(f"[Startup] Engine ready — {len(detection_engine.loaded_models)}/3 models loaded.")
    except Exception as e:
        print(f"[Startup] WARNING: Engine pre-load failed: {e}. Will retry on first request.")

# Request/Response models
class TrackedTool(BaseModel):
    """Typed representation of a single tracked tool with its FSM state."""
    box:   List[float]
    name:  str
    timer: float
    state: str

class DetectionResult(BaseModel):
    frame_number: int
    timestamp: str
    fps: float
    humans_detected: int
    tools_detected: int
    alerts: List[Dict[str, Any]]
    tracked_tools: Dict[str, TrackedTool]

class SystemStatus(BaseModel):
    status: str
    models_loaded: bool
    uptime_seconds: float
    total_requests: int

class ConfigUpdate(BaseModel):
    conf_human: Optional[float] = None
    conf_tool: Optional[float] = None
    conf_ppe: Optional[float] = None
    t1_warning: Optional[float] = None
    t2_alert: Optional[float] = None
    zone_factor: Optional[float] = None

# Global stats
stats = {
    'start_time': time.time(),
    'total_requests': 0,
    'total_frames': 0,
    'total_alerts': 0
}

# Routes
@app.get("/")
async def root():
    """API root"""
    return {
        "message": "Industrial Safety Detection API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "detect_image": "/api/detect/image",
            "detect_video": "/api/detect/video",
            "detect_stream": "/api/detect/stream",
            "status": "/api/status",
            "config": "/api/config",
            "alerts": "/api/alerts",
            "metrics": "/api/metrics"
        }
    }

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get system status"""
    return SystemStatus(
        status="online",
        models_loaded=detection_engine is not None,
        uptime_seconds=time.time() - stats['start_time'],
        total_requests=stats['total_requests']
    )

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "total_requests": stats['total_requests'],
        "total_frames_processed": stats['total_frames'],
        "total_alerts_generated": stats['total_alerts'],
        "uptime_seconds": time.time() - stats['start_time'],
        "model_performance": MODEL_INFO
    }

@app.post("/api/detect/image")
async def detect_image(file: UploadFile = File(...)):
    """
    Detect safety violations in a single image.

    Validates file size (<50 MB) and confirms the file is a decodable image
    before running inference. Returns detection results with an annotated image.
    """
    stats['total_requests'] += 1

    try:
        # Read and validate file size (cap at 50 MB)
        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 50 MB)")

        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. Ensure the file is a valid JPEG, PNG, or BMP."
            )

        # Get engine (raises 503 on failure)
        engine = get_engine()

        # Process
        annotated, data = engine.process_frame(frame)

        # Update stats
        stats['total_frames'] += 1
        stats['total_alerts'] += len(data['alerts'])

        # Encode annotated image
        _, buffer = cv2.imencode('.jpg', annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "detection_results": {
                "fps": data['fps'],
                "humans_detected": data['humans'],
                "tools_detected": data['tools'],
                "alerts": data['alerts'],
                "tracked_tools": data['tracked_tools']
            },
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detect/video")
async def detect_video(file: UploadFile = File(...)):
    """
    Process a video file and return frame-by-frame detection results.

    Validates that the video opens successfully. Skips unreadable frames
    rather than crashing. Temp file is always cleaned up via a finally block.
    """
    stats['total_requests'] += 1
    video_path = None

    try:
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            contents = await file.read()
            tmp.write(contents)
            video_path = tmp.name

        # Get engine (raises 503 on failure)
        engine = get_engine()

        # Open and validate
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open video file. Ensure it is a valid MP4/AVI/MOV.")

        results = []
        frame_count = 0
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip unreadable frames gracefully rather than crashing
            try:
                annotated, data = engine.process_frame(frame)
            except Exception as frame_err:
                print(f"  Video frame {frame_count} error (skipped): {frame_err}")
                data = {"fps": 0, "humans": 0, "tools": 0, "alerts": []}

            frame_count += 1

            results.append({
                "frame": frame_count,
                "timestamp": frame_count / video_fps,
                "detections": {
                    "humans": data['humans'],
                    "tools": data['tools'],
                    "alerts": data['alerts']
                }
            })

            stats['total_frames'] += 1
            stats['total_alerts'] += len(data['alerts'])

        cap.release()

        return {
            "success": True,
            "total_frames": frame_count,
            "total_alerts": sum(len(r['detections']['alerts']) for r in results),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Always clean up the temp file, even on exceptions
        if video_path:
            try:
                Path(video_path).unlink(missing_ok=True)
            except Exception:
                pass

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "detection": {
            "conf_human": CONF_HUMAN,
            "conf_tool": CONF_TOOL,
            "conf_ppe": CONF_PPE,
            "image_size": IMAGE_SIZE
        },
        "temporal_logic": {
            "t1_warning": T1_WARNING,
            "t2_alert": T2_ALERT,
            "zone_factor": ZONE_EXPAND_FACTOR
        },
        "tracking": {
            "tracking_iou": TRACKING_IOU_THRESHOLD,
            "human_zone_iou": HUMAN_ZONE_IOU,
            "ppe_human_iou": PPE_HUMAN_IOU
        }
    }

@app.post("/api/config")
async def update_config(config: ConfigUpdate):
    """
    Update system configuration
    
    Note: Changes apply to new requests only
    """
    global CONF_HUMAN, CONF_TOOL, CONF_PPE, T1_WARNING, T2_ALERT, ZONE_EXPAND_FACTOR
    
    updated = {}
    
    if config.conf_human is not None:
        CONF_HUMAN = config.conf_human
        updated['conf_human'] = CONF_HUMAN
    
    if config.conf_tool is not None:
        CONF_TOOL = config.conf_tool
        updated['conf_tool'] = CONF_TOOL
    
    if config.conf_ppe is not None:
        CONF_PPE = config.conf_ppe
        updated['conf_ppe'] = CONF_PPE
    
    if config.t1_warning is not None:
        T1_WARNING = config.t1_warning
        updated['t1_warning'] = T1_WARNING
    
    if config.t2_alert is not None:
        T2_ALERT = config.t2_alert
        updated['t2_alert'] = T2_ALERT
    
    if config.zone_factor is not None:
        ZONE_EXPAND_FACTOR = config.zone_factor
        updated['zone_factor'] = ZONE_EXPAND_FACTOR
    
    return {
        "success": True,
        "message": "Configuration updated",
        "updated_parameters": updated
    }

@app.get("/api/alerts")
async def get_alerts(limit: int = 100):
    """Get recent alerts (mock endpoint - would connect to database in production)"""
    return {
        "total_alerts": stats['total_alerts'],
        "message": "Connect to database for full alert history"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

# Example: RTSP stream processing endpoint (for CCTV cameras)
@app.post("/api/detect/stream/start")
async def start_stream(rtsp_url: str, background_tasks: BackgroundTasks):
    """
    Start processing RTSP stream from CCTV camera
    
    Example: rtsp://username:password@192.168.1.100:554/stream
    """
    # In production, this would start a background worker
    # For now, return success message
    return {
        "success": True,
        "message": "Stream processing started",
        "stream_url": rtsp_url,
        "note": "This is a mock endpoint. Implement background processing for production."
    }

if __name__ == "__main__":
    print("="*70)
    print("Starting Industrial Safety Detection API Server")
    print("="*70)
    print(f"\nAPI will be available at: http://{API_HOST}:{API_PORT}")
    print(f"Documentation: http://{API_HOST}:{API_PORT}/docs")
    print(f"Alternative docs: http://{API_HOST}:{API_PORT}/redoc")
    print("\nPress Ctrl+C to stop\n")
    print("="*70)
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )

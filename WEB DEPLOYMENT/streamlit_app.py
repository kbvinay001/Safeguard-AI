"""
streamlit_app.py — SafeGuard AI Cyberpunk Dashboard
====================================================
Complete UI overhaul: military-HUD / cyberpunk aesthetic with
corner-bracket cards, animated cyan accents, and monospace fonts.

Deployment: Railway (Procfile) + Supabase Postgres
"""

import streamlit as st
import cv2, numpy as np, tempfile, time, os, threading, queue
import datetime, uuid, base64
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SafeGuard AI — Industrial Safety Monitor",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Local imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from detection_engine import SafetyDetectionEngine
    from safety_config    import *
    from db_manager       import (log_alert, upsert_session,
                                  get_all_alerts, get_all_sessions,
                                  get_alert_counts_by_hour)
    from pdf_report       import generate_pdf
    from ui_styles        import get_css, BG_B64, BAN_B64
except Exception as _e:
    st.error(f"Import error: {_e}")
    st.stop()

# ── Inject CSS ─────────────────────────────────────────────────────────────
st.markdown(get_css(BG_B64), unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = dict(
        engine=None, alert_log=[], total_frames=0, total_alerts=0,
        processed_video_path=None, processing_stats=None,
        system_log=[], session_id=str(uuid.uuid4())[:8].upper(),
        heatmap_acc=None, live_active=False,
        live_queue=None, live_stop=None,
        compliance=100.0,
        # Two-level navigation
        top_page="dashboard",   # "dashboard" | "deployment"
        sub_page="home",        # "home"|"analyse"|"analytics"|"history"|"logs"|"train"
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


def add_log(msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.system_log.append({"time": ts, "level": level, "msg": msg})
    if len(st.session_state.system_log) > 300:
        st.session_state.system_log = st.session_state.system_log[-300:]


def cmd_card(label: str, value: str, sub: str = "", color: str = "cy") -> str:
    """Return HTML for one Command Center metric card."""
    return (
        f'<div class="cmd-card {color}">'
        f'<div class="cmd-label">{label}</div>'
        f'<div class="cmd-value {color}">{value}</div>'
        f'<div class="cmd-sub">{sub}</div>'
        f'</div>'
    )


def section(title: str) -> str:
    return f'<div class="sg-section">{title}</div>'


# ════════════════════════════════════════════════════════════════════════════
# LOAD ENGINE
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_engine():
    try:
        e = SafetyDetectionEngine(HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=DEVICE)
        return e, None
    except Exception as ex:
        return None, str(ex)

if st.session_state.engine is None:
    with st.spinner("INITIALISING MODELS — YOLOV11 HUMAN / TOOL / PPE..."):
        _eng, _err = load_engine()
        if _err:
            add_log(f"Model load warning: {_err}", "WARN")
        if _eng:
            st.session_state.engine = _eng
            _loaded = ", ".join(_eng.loaded_models) if _eng.loaded_models else "none"
            add_log(f"Models loaded: {_loaded} | Human 99.4% | PPE 79.9% | Tools 67.9%", "SUCCESS")
            add_log(f"Session {st.session_state.session_id} started", "INFO")
        else:
            st.error("CRITICAL: Could not load any model. Check weight paths in safety_config.py.")
            st.stop()


# ════════════════════════════════════════════════════════════════════════════
# TOP NAVIGATION BAR
# ════════════════════════════════════════════════════════════════════════════
engine = st.session_state.engine
models_active = len(engine.loaded_models) if engine else 0

_top  = st.session_state.top_page
_sub  = st.session_state.sub_page

st.markdown(f"""
<div class="topnav">
  <div class="topnav-brand">
    <div class="topnav-dots">
      <div class="topnav-dot"></div>
      <div class="topnav-dot"></div>
      <div class="topnav-dot"></div>
    </div>
    <div>
      <div class="topnav-logo-text">&#9632;&nbsp;SAFE<em>GUARD</em></div>
      <div class="topnav-subtitle">AI SAFETY MONITORING SYSTEM v2.0</div>
    </div>
  </div>
  <div class="topnav-tabs">
    <div class="topnav-tab {'active' if _top=='dashboard' else ''}">
      <div class="tab-dot"></div>
      ◆ DASHBOARD
    </div>
    <div class="topnav-tab {'active' if _top=='deployment' else ''}">
      <div class="tab-radio"><div class="tab-radio-inner"></div></div>
      ◎ DEPLOYMENT
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Actual Streamlit navigation buttons (invisible, positioned over HTML tabs)
t1, t2, _g = st.columns([1, 1, 6])
with t1:
    if st.button("▸ DASHBOARD", key="nav_dashboard", use_container_width=True):
        st.session_state.top_page = "dashboard"
        st.session_state.sub_page = "home"
        st.rerun()
with t2:
    if st.button("⊙ DEPLOYMENT", key="nav_deployment", use_container_width=True):
        st.session_state.top_page = "deployment"
        st.rerun()

top_page = st.session_state.top_page


# ════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT MODE — Railway / Supabase deployment info page
# ════════════════════════════════════════════════════════════════════════════
if top_page == "deployment":
    st.markdown(section("// DEPLOYMENT CONFIGURATION"), unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown("""
<div class="sg-card">
  <div class="qa-title">🛤 RAILWAY (PROCFILE)</div>
  <div class="feature-desc" style="margin-top:.6rem;">
    <strong style="color:#a0c4b0;">Procfile</strong><br>
    <code style="font-family:var(--mono);color:#00ffcc;font-size:.78rem;">
    web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
    </code><br><br>
    Railway auto-detects the Procfile and deploys the app publicly.<br>
    No localhost required — Railway assigns a public HTTPS URL.<br><br>
    <strong style="color:#a0c4b0;">requirements.txt</strong> includes all Python deps.<br>
    Model weights are loaded from environment-variable paths.
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="sg-card" style="margin-top:.9rem;">
  <div class="qa-title">🗄 SUPABASE POSTGRES</div>
  <div class="feature-desc" style="margin-top:.6rem;">
    Set these environment variables in Railway:<br><br>
    <code style="font-family:var(--mono);color:#00ffcc;font-size:.76rem;">
    SUPABASE_URL=https://&lt;project&gt;.supabase.co<br>
    SUPABASE_KEY=&lt;service_role_key&gt;<br>
    DATABASE_URL=postgresql://postgres:...@db.supabase.co:5432/postgres
    </code><br><br>
    The db_manager.py automatically uses Postgres when DATABASE_URL is set,
    falling back to SQLite for local development.
  </div>
</div>
""", unsafe_allow_html=True)

    with d2:
        st.markdown("""
<div class="sg-card">
  <div class="qa-title">⚡ QUICK DEPLOY STEPS</div>
  <div class="feature-desc" style="margin-top:.8rem;font-family:var(--mono);font-size:.76rem;line-height:2;">
    <span style="color:#00ffcc;">01</span>&nbsp; Push code to GitHub<br>
    <span style="color:#00ffcc;">02</span>&nbsp; New project → Deploy from GitHub<br>
    <span style="color:#00ffcc;">03</span>&nbsp; Add Supabase plugin in Railway<br>
    <span style="color:#00ffcc;">04</span>&nbsp; Set YOLO weight paths as env vars<br>
    <span style="color:#00ffcc;">05</span>&nbsp; Railway auto-builds via requirements.txt<br>
    <span style="color:#00ffcc;">06</span>&nbsp; Public HTTPS URL generated automatically<br>
    <span style="color:#00ffcc;">07</span>&nbsp; SafeGuard AI live — no localhost needed
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div class="cmd-grid" style="margin-top:.9rem;">
  {cmd_card("PLATFORM", "Railway", "PaaS, auto HTTPS", "cy")}
  {cmd_card("DATABASE", "Supabase", "Postgres in prod", "gn")}
  {cmd_card("PORT", "$PORT", "Auto-assigned", "bl")}
  {cmd_card("MODELS", f"{models_active}/3", "YOLOv11n loaded", "or")}
</div>
""", unsafe_allow_html=True)

    st.markdown(section("// PROCFILE CONTENT"), unsafe_allow_html=True)
    st.code(
        "web: streamlit run streamlit_app.py "
        "--server.port=$PORT --server.address=0.0.0.0 "
        "--server.headless=true --browser.gatherUsageStats=false",
        language="bash"
    )

    st.markdown(section("// ENV VARIABLES (set in Railway dashboard)"), unsafe_allow_html=True)
    st.code("""DATABASE_URL=postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<service_role_key>
HUMAN_WEIGHTS_PATH=/app/models/human_best.pt
PPE_WEIGHTS_PATH=/app/models/ppe_best.pt
TOOL_WEIGHTS_PATH=/app/models/tools_best.pt""", language="bash")

    st.markdown("""
<div class="sg-footer">
  <span>SafeGuard AI</span> &nbsp;·&nbsp; Deploy via Railway + Supabase &nbsp;·&nbsp;
  No localhost required &nbsp;·&nbsp; HTTPS auto-provisioned
</div>
""", unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD MODE — sub navigation
# ════════════════════════════════════════════════════════════════════════════
SUB_PAGES = [
    ("home",      "♦",  "HOME"),
    ("analyse",   "▶",  "ANALYSE"),
    ("analytics", "◉", "ANALYTICS"),
    ("history",   "≋",  "HISTORY"),
    ("logs",      "≡",  "LOGS"),
    ("train",     "⚡", "TRAIN"),
]

# Render HTML sub-nav (for display)
sub_html = '<div class="subnav">'
for key, icon, label in SUB_PAGES:
    active_cls = "active" if _sub == key else ""
    sub_html += f'<div class="subnav-btn {active_cls}"><span class="subnav-icon">{icon}</span>{label}</div>'
sub_html += "</div>"
st.markdown(sub_html, unsafe_allow_html=True)

# Actual Streamlit sub-nav buttons (small, rendered above the visual sub-nav)
sub_cols = st.columns(len(SUB_PAGES))
for i, (key, icon, label) in enumerate(SUB_PAGES):
    with sub_cols[i]:
        if st.button(f"{icon} {label}", key=f"sub_{key}", use_container_width=True):
            st.session_state.sub_page = key
            st.rerun()

sub_page = st.session_state.sub_page


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def calc_compliance(frames: int, alerts: int) -> float:
    if frames == 0:
        return 100.0
    return max(0.0, min(100.0, 100.0 - (alerts / max(frames, 1)) * 100 * 50))


def compliance_gauge(score: float):
    color = "#00ff88" if score >= 75 else ("#ffe033" if score >= 50 else "#ff3333")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 30, "color": "#e0fff0", "family": "Share Tech Mono"}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#0d2b1e",
                      tickfont=dict(color="#3a6050", size=9, family="Share Tech Mono")),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#040806", bordercolor="#0d2b1e", borderwidth=1,
            steps=[
                dict(range=[0,  50], color="rgba(255,51,51,.1)"),
                dict(range=[50, 75], color="rgba(255,224,51,.08)"),
                dict(range=[75,100], color="rgba(0,255,136,.08)"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=.75, value=score),
        ),
        title={"text": "SAFETY COMPLIANCE", "font": {"size": 10, "color": "#3a6050", "family": "Share Tech Mono"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#040806", plot_bgcolor="#040806",
        font=dict(family="Share Tech Mono"),
    )
    return fig


def alerts_bar(alerts: list):
    if not alerts:
        return None
    df = pd.DataFrame(alerts)
    if "Type" not in df.columns:
        return None
    vc = df["Type"].value_counts().reset_index()
    vc.columns = ["Type", "Count"]
    fig = px.bar(vc, x="Type", y="Count", color="Type",
                 color_discrete_map={"PPE_VIOLATION": "#ff3333", "TOOL_UNATTENDED": "#ff6a00"})
    fig.update_layout(
        height=200, margin=dict(t=20, b=30, l=20, r=10),
        paper_bgcolor="#040806", plot_bgcolor="#040806",
        showlegend=False,
        font=dict(family="Share Tech Mono", color="#a0c4b0"),
        xaxis=dict(gridcolor="#0d2b1e"),
        yaxis=dict(gridcolor="#0d2b1e"),
    )
    return fig


def fps_line(readings: list):
    if len(readings) < 2:
        return None
    fig = go.Figure(go.Scatter(
        y=readings, mode="lines",
        line=dict(color="#00ffcc", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,255,204,.06)",
    ))
    fig.update_layout(
        height=160, margin=dict(t=10, b=20, l=30, r=10),
        paper_bgcolor="#040806", plot_bgcolor="#040806",
        title=dict(text="PROCESSING SPEED (FPS)", font=dict(size=9, color="#3a6050", family="Share Tech Mono"), x=0),
        xaxis=dict(showticklabels=False, gridcolor="#0d2b1e"),
        yaxis=dict(gridcolor="#0d2b1e", tickfont=dict(color="#3a6050")),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING FUNCTION
# ════════════════════════════════════════════════════════════════════════════
def process_video(input_path: str, output_path: str, progress_cb=None):
    eng = st.session_state.engine
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, "Cannot open video"

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    all_alerts, fps_readings, frame_count = [], [], 0
    heatmap_acc = np.zeros((h, w), dtype=np.float32)
    t0 = time.time()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            ft = time.time()
            try:
                annotated, data = eng.process_frame(frame)
            except Exception:
                annotated = frame
                data = {"fps": 0, "humans": 0, "tools": 0, "alerts": [], "tracked_tools": {}}
            writer.write(annotated)
            frame_count += 1
            dt = time.time() - ft
            if dt > 0:
                fps_readings.append(1.0 / dt)
            for a in data.get("alerts", []):
                a["frame"] = frame_count
                all_alerts.append(a)
                cx, cy = w // 2, h // 2
                heatmap_acc[max(0,cy-60):min(h,cy+60), max(0,cx-60):min(w,cx+60)] += 1.0
            if progress_cb and frame_count % 8 == 0:
                progress_cb(frame_count / max(total, 1), frame_count, total,
                            fps_readings[-1] if fps_readings else 0.0)
    finally:
        cap.release()
        writer.release()

    comp = calc_compliance(frame_count, len(all_alerts))
    st.session_state.heatmap_acc = heatmap_acc

    return {
        "total_frames": frame_count,
        "total_alerts": len(all_alerts),
        "avg_fps":      float(np.mean(fps_readings)) if fps_readings else 0.0,
        "total_time":   time.time() - t0,
        "alerts":       all_alerts,
        "fps_series":   fps_readings[::10],
        "compliance":   comp,
    }, None


# ════════════════════════════════════════════════════════════════════════════
# LIVE STREAM THREAD
# ════════════════════════════════════════════════════════════════════════════
def _live_worker(url, fq: queue.Queue, stop: threading.Event, eng):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        fq.put(("ERR", f"Cannot open: {url}"))
        return
    fq.put(("LOG", f"Stream connected: {url}"))
    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            fq.put(("ERR", "Stream ended."))
            break
        try:
            ann, data = eng.process_frame(frame)
        except Exception:
            ann = frame
            data = {"fps": 0, "humans": 0, "tools": 0, "alerts": []}
        try:
            fq.get_nowait()
        except Exception:
            pass
        fq.put(("FRAME", ann, data))
    cap.release()
    fq.put(("LOG", "Stream stopped."))


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ════════════════════════════════════════════════════════════════════════════
if sub_page == "home":
    st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">// INDUSTRIAL SAFETY INTELLIGENCE · YOLOv11 · RTX 4060 · CUDA FP16</div>
  <h1>SafeGuard AI</h1>
  <p class="hero-desc">
    Autonomous real-time PPE compliance and tool abandonment detection
    for industrial CCTV infrastructure.
  </p>
</div>
""", unsafe_allow_html=True)

    # Command Center metrics
    comp = st.session_state.compliance
    sid  = st.session_state.session_id
    st.markdown(section("COMMAND CENTER"), unsafe_allow_html=True)
    st.markdown(f"""
<div class="cmd-grid">
  {cmd_card("SAFETY COMPLIANCE", f"{comp:.1f}%", "Current session", "cy" if comp >= 75 else "rd")}
  {cmd_card("FRAMES ANALYSED",  str(st.session_state.total_frames), "This session", "gn")}
  {cmd_card("ALERTS RAISED",    str(st.session_state.total_alerts), "Safety violations", "rd" if st.session_state.total_alerts > 0 else "gn")}
  {cmd_card("SESSION ID",       sid, "Active token", "wh")}
  {cmd_card("MODELS ACTIVE",    f"{models_active}/3", "YOLOv11n loaded", "or")}
</div>
""", unsafe_allow_html=True)

    # Quick Actions
    st.markdown(section("QUICK ACTIONS"), unsafe_allow_html=True)
    st.markdown("""
<div class="qa-grid">
  <div class="qa-card">
    <div class="qa-title">▶ VIDEO ANALYSIS</div>
    <div class="qa-desc">Upload CCTV recordings for batch PPE + tool detection with heatmap and PDF export.</div>
  </div>
  <div class="qa-card">
    <div class="qa-title">◉ LIVE RTSP</div>
    <div class="qa-desc">Connect IP cameras via RTSP URL for real-time detection with background threading.</div>
  </div>
  <div class="qa-card">
    <div class="qa-title">◉ ANALYTICS</div>
    <div class="qa-desc">Compliance gauge, hourly alert charts, session history and audit export.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # CTA
    cta1, cta2, cta3 = st.columns(3)
    with cta1:
        if st.button("▶ ANALYSE VIDEO", use_container_width=True, type="primary"):
            st.session_state.sub_page = "analyse"; st.rerun()
    with cta2:
        if st.button("◉ LIVE STREAM",   use_container_width=True):
            st.session_state.sub_page = "live"; st.rerun()
    with cta3:
        if st.button("◈ VIEW ANALYTICS", use_container_width=True):
            st.session_state.sub_page = "analytics"; st.rerun()

    # Core capabilities
    st.markdown(section("CORE CAPABILITIES"), unsafe_allow_html=True)
    st.markdown("""
<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">👁</div>
    <div class="feature-title">HUMAN DETECTION</div>
    <div class="feature-desc">YOLOv11n fine-tuned on 15,357 industrial images. Tracks every worker with persistent ID.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🦺</div>
    <div class="feature-title">PPE COMPLIANCE</div>
    <div class="feature-desc">Helmet · Vest · Gloves · Goggles per worker every frame. Alert within 1 second of violation.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">🔧</div>
    <div class="feature-title">TOOL TRACKING</div>
    <div class="feature-desc">Drill · Hammer · Pliers · Wrench. Temporal FSM: 25s warning + 35s critical alert.</div>
  </div>
  <div class="feature-card">
    <div class="feature-icon">📡</div>
    <div class="feature-title">RTSP STREAMING</div>
    <div class="feature-desc">Connect IP cameras via RTSP/RTMP. Live detection with background threading.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # System architecture
    st.markdown(section("SYSTEM ARCHITECTURE"), unsafe_allow_html=True)
    st.markdown(f"""
<div class="sg-card">
  <pre style="font-family:var(--mono);font-size:.76rem;color:#3a6050;margin:0;line-height:1.9;">
 CCTV / RTSP INPUT
   │
   ├── Human Model (YOLOv11n · 99.44% mAP) ──→ Worker boxes, persistent IDs
   ├── Tool Model  (YOLOv11n · 67.87% mAP) ──→ Abandoned-tool detection + FSM timer
   └── PPE Model   (YOLOv11n · 79.90% mAP) ──→ Helmet / Vest / Gloves compliance
         │         │         │
         └─────────┴─────────┘
                   │
           IoU Spatial Fusion
                   │
         ┌─────────▼──────────┐
         │  ALERT SYSTEM       │  WARNING @25s · CRITICAL @35s · PPE instant
         │  SQLite / Postgres  │  Persistent alert history
         │  Streamlit UI       │  Live dashboard + PDF reports
         └────────────────────┘
  Session: <span style="color:#00ffcc;">{st.session_state.session_id}</span> · Models: <span style="color:#00ffcc;">{models_active}/3</span> · Device: <span style="color:#00ffcc;">CUDA FP16</span>
  </pre>
</div>
""", unsafe_allow_html=True)

    # Settings expander
    with st.expander("⚙ DETECTION SETTINGS"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown('<div class="sg-section">THRESHOLDS</div>', unsafe_allow_html=True)
            conf_tool  = st.slider("Tool Confidence",  0.10, 0.80, float(CONF_TOOL),  0.05)
            conf_human = st.slider("Human Confidence", 0.15, 0.80, float(CONF_HUMAN), 0.05)
            conf_ppe   = st.slider("PPE Confidence",   0.10, 0.80, float(CONF_PPE),   0.05)
        with sc2:
            st.markdown('<div class="sg-section">TEMPORAL LOGIC</div>', unsafe_allow_html=True)
            t1   = st.slider("T1 Warning (s)", 5,  60, int(T1_WARNING),        5)
            t2   = st.slider("T2 Alert (s)",   10, 90, int(T2_ALERT),          5)
            zone = st.slider("Zone Factor",    1.0, 3.0, float(ZONE_EXPAND_FACTOR), 0.1)
        with sc3:
            st.markdown('<div class="sg-section">SESSION</div>', unsafe_allow_html=True)
            st.metric("Frames", st.session_state.total_frames)
            st.metric("Alerts", st.session_state.total_alerts)
            st.metric("Compliance", f"{st.session_state.compliance:.1f}%")
            if st.button("↺ RESET SESSION", use_container_width=True):
                for k in ["alert_log", "system_log"]:
                    st.session_state[k] = []
                for k in ["total_frames", "total_alerts"]:
                    st.session_state[k] = 0
                st.session_state.compliance = 100.0
                st.session_state.processed_video_path = None
                st.session_state.processing_stats = None
                st.session_state.heatmap_acc = None
                st.session_state.session_id = str(uuid.uuid4())[:8].upper()
                add_log("Session reset.", "INFO"); st.rerun()
        CONF_TOOL = conf_tool; CONF_HUMAN = conf_human; CONF_PPE = conf_ppe
        T1_WARNING = t1; T2_ALERT = t2; ZONE_EXPAND_FACTOR = zone


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYSE (Video Analysis)
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "analyse":
    st.markdown(section("// VIDEO ANALYSIS"), unsafe_allow_html=True)

    cu, ch = st.columns([2, 1])
    with cu:
        uf = st.file_uploader("UPLOAD CCTV / FACTORY VIDEO (MP4 · AVI · MOV)",
                              type=["mp4", "avi", "mov", "mkv"])
    with ch:
        st.markdown("""
<div class="ic in" style="font-family:var(--mono);font-size:.76rem;line-height:2;">
  <strong style="color:#00ffcc;">WORKFLOW</strong><br>
  01 &nbsp;Upload CCTV video<br>
  02 &nbsp;GPU processes all frames<br>
  03 &nbsp;Review heatmap + charts<br>
  04 &nbsp;Download video / CSV / PDF
</div>""", unsafe_allow_html=True)

    if uf:
        fsz = len(uf.getbuffer()) / 1048576
        st.markdown(f'<div class="sg-section">FILE: {uf.name} ({fsz:.1f} MB)</div>', unsafe_allow_html=True)
        if st.button("▶ PROCESS VIDEO", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ti:
                ti.write(uf.getbuffer()); inp = ti.name
            out = inp.replace(".mp4", "_out.mp4")
            add_log(f"Processing: {uf.name} ({fsz:.1f} MB)", "INFO")

            pb  = st.progress(0.0)
            pt  = st.empty()
            fpc = st.empty()

            def upd(p, f, tot, cfps):
                pb.progress(float(p))
                pt.markdown(f'<div class="cmd-sub" style="color:#00ffcc;font-family:var(--mono);">FRAME {f}/{tot} — {p*100:.1f}% — {cfps:.1f} FPS</div>', unsafe_allow_html=True)

            t0 = time.time()
            stats, err = process_video(inp, out, upd)
            elapsed = time.time() - t0

            if err:
                st.error(err); add_log(f"FAILED: {err}", "ALERT")
            elif stats:
                st.session_state.processing_stats = stats
                st.session_state.processed_video_path = out
                st.session_state.total_frames  += stats["total_frames"]
                st.session_state.total_alerts  += stats["total_alerts"]
                st.session_state.compliance     = stats["compliance"]
                for a in stats["alerts"]:
                    st.session_state.alert_log.append({
                        "Frame": a.get("frame", 0),
                        "Type":  a.get("type", "UNKNOWN"),
                        "Tool":  a.get("tool_name", "N/A"),
                        "Timer (s)": round(float(a.get("timer", 0)), 2),
                        "Missing PPE": str(a.get("missing_ppe", [])),
                    })
                    log_alert(st.session_state.session_id, "video_upload",
                              a.get("frame", 0), a.get("type", "?"),
                              a.get("tool_name", ""), float(a.get("timer", 0)),
                              a.get("missing_ppe", []))
                upsert_session(st.session_state.session_id, "video_upload",
                               st.session_state.total_frames,
                               st.session_state.total_alerts,
                               stats["compliance"])
                pb.progress(1.0)
                add_log(f"Done — {stats['total_frames']} frames | {stats['total_alerts']} alerts | {stats['avg_fps']:.1f} FPS", "SUCCESS")
            try:
                os.unlink(inp)
            except Exception:
                pass

    # Results display
    if st.session_state.processed_video_path and Path(st.session_state.processed_video_path).exists():
        stats = st.session_state.processing_stats
        if stats:
            st.markdown(section("RESULTS"), unsafe_allow_html=True)
            co, cg = st.columns([2, 1])
            with co:
                st.markdown(f"""
<div class="cmd-grid">
  {cmd_card("FRAMES",     str(stats['total_frames']),          "Total analysed",   "cy")}
  {cmd_card("AVG FPS",    f"{stats['avg_fps']:.1f}",           "Processing speed", "gn")}
  {cmd_card("ALERTS",     str(stats['total_alerts']),           "Safety violations","rd" if stats['total_alerts']>0 else "gn")}
  {cmd_card("TIME",       f"{stats['total_time']:.1f}s",        "Wall-clock",       "or")}
  {cmd_card("COMPLIANCE", f"{stats['compliance']:.1f}%",        "Safety score",     "cy" if stats['compliance']>=75 else "rd")}
</div>""", unsafe_allow_html=True)
                if stats.get("fps_series"):
                    st.plotly_chart(fps_line(stats["fps_series"]), use_container_width=True, config={"displayModeBar": False})
            with cg:
                st.plotly_chart(compliance_gauge(stats["compliance"]), use_container_width=True, config={"displayModeBar": False})

        cv, ch2 = st.columns(2)
        with cv:
            st.markdown(section("ANNOTATED OUTPUT"), unsafe_allow_html=True)
            try:
                with open(st.session_state.processed_video_path, "rb") as vf:
                    st.video(vf.read())
            except Exception as ex:
                st.warning(f"Preview unavailable: {ex}")
        with ch2:
            st.markdown(section("VIOLATION HEATMAP"), unsafe_allow_html=True)
            if st.session_state.heatmap_acc is not None and st.session_state.heatmap_acc.max() > 0:
                acc  = st.session_state.heatmap_acc
                norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hm   = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB),
                         caption="Red zones = frequent violation locations", use_column_width=True)
            else:
                st.markdown('<div class="ic in" style="text-align:center;padding:2rem;">No violation data — heatmap appears after alerts are detected.</div>', unsafe_allow_html=True)

        # Downloads
        st.markdown(section("EXPORT"), unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            try:
                with open(st.session_state.processed_video_path, "rb") as f:
                    st.download_button("⬇ VIDEO", f.read(), "safety_output.mp4", "video/mp4", use_container_width=True)
            except Exception:
                pass
        with d2:
            if st.session_state.alert_log:
                df_dl = pd.DataFrame(st.session_state.alert_log)
                st.download_button("⬇ CSV", df_dl.to_csv(index=False), "alerts.csv", "text/csv", use_container_width=True)
        with d3:
            if stats:
                alts_for_pdf = [{"ts": datetime.datetime.now().isoformat(),
                                  "type": a.get("type", "?"),
                                  "tool": a.get("tool_name", ""),
                                  "timer_s": float(a.get("timer", 0)),
                                  "missing_ppe": str(a.get("missing_ppe", []))}
                                 for a in stats.get("alerts", [])]
                try:
                    pdf_bytes = generate_pdf({**stats, "source": "video_upload", "compliance": stats["compliance"]},
                                             alts_for_pdf, st.session_state.session_id)
                    st.download_button("⬇ PDF REPORT", pdf_bytes, "safety_report.pdf", "application/pdf", use_container_width=True)
                except Exception:
                    pass
        with d4:
            if st.button("✕ CLEAR", use_container_width=True):
                st.session_state.processed_video_path = None
                st.session_state.processing_stats = None
                add_log("Results cleared.", "INFO"); st.rerun()
    else:
        st.markdown('<div class="ic in" style="margin-top:1rem;">Upload a CCTV or factory video file above to begin analysis.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE STREAM (sub_page == "live" accessed via analyse tab)
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "live":
    st.markdown(section("// LIVE RTSP / IP CAMERA"), unsafe_allow_html=True)
    st.markdown("""
<div class="ic in" style="font-family:var(--mono);font-size:.76rem;line-height:2.1;">
  <strong style="color:#00ffcc;">SUPPORTED PROTOCOLS</strong><br>
  RTSP &nbsp; rtsp://user:pass@ip:554/stream1<br>
  HTTP &nbsp; http://ip:port/video<br>
  Local &nbsp;0 (USB / built-in webcam)
</div>""", unsafe_allow_html=True)

    ci1, ci2 = st.columns([3, 1])
    with ci1:
        ip_in = st.text_input("CAMERA IP / STREAM URL", "", placeholder="rtsp://admin:admin@192.168.1.100:554/stream1")
    with ci2:
        proto = st.selectbox("PROTOCOL", ["RTSP", "HTTP MJPEG", "Local Device (0)"])

    cp1, cp2, cp3 = st.columns(3)
    port     = cp1.text_input("PORT", "554")
    chan     = cp2.selectbox("CHANNEL", ["stream1", "stream2", "ch01", "ch02", "live"])
    use_auth = cp3.checkbox("REQUIRES AUTH")
    if use_auth:
        au, ap = st.columns(2)
        cam_u = au.text_input("USERNAME", "admin")
        cam_p = ap.text_input("PASSWORD", "", type="password")
    else:
        cam_u = cam_p = ""

    def build_url(ip, proto, port, chan, u, p):
        r = ip.strip()
        if r.startswith(("rtsp://", "http://", "https://")):
            return r
        try:
            return int(r)
        except Exception:
            pass
        if proto == "RTSP":
            return f"rtsp://{u}:{p}@{r}:{port}/{chan}" if u else f"rtsp://{r}:{port}/{chan}"
        elif proto == "HTTP MJPEG":
            return f"http://{r}:{port}/video"
        return 0

    surl = build_url(ip_in, proto, port, chan, cam_u, cam_p)
    if ip_in.strip():
        st.markdown(f'<div class="ic in" style="font-family:var(--mono);font-size:.76rem;color:#00ffcc;">URL: {surl}</div>', unsafe_allow_html=True)

    sb1, sb2 = st.columns(2)
    if sb1.button("▶ CONNECT", type="primary", use_container_width=True) and ip_in.strip():
        if st.session_state.live_stop:
            st.session_state.live_stop.set()
            time.sleep(0.3)
        fq   = queue.Queue(maxsize=2)
        stop = threading.Event()
        threading.Thread(target=_live_worker, args=(surl, fq, stop, st.session_state.engine), daemon=True).start()
        st.session_state.live_queue  = fq
        st.session_state.live_stop   = stop
        st.session_state.live_active = True
        add_log(f"Live stream started: {surl}", "INFO")
    if sb2.button("⏹ DISCONNECT", use_container_width=True) and st.session_state.live_stop:
        st.session_state.live_stop.set()
        st.session_state.live_active = False
        add_log("Live stream disconnected.", "WARN"); st.rerun()

    if st.session_state.live_active and st.session_state.live_queue:
        st.markdown(section("LIVE FEED"), unsafe_allow_html=True)
        cs, cm = st.columns([3, 1])
        with cs:
            st.markdown('<span class="pill pill-on">● STREAM ACTIVE</span>', unsafe_allow_html=True)
            sp = st.empty()
        with cm:
            hp  = st.empty()
            tp  = st.empty()
            alp = st.empty()
            fp  = st.empty()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                msg = st.session_state.live_queue.get(timeout=0.15)
                if msg[0] == "FRAME":
                    _, ann, data = msg
                    sp.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    hp.metric("HUMANS",  data.get("humans", 0))
                    tp.metric("TOOLS",   data.get("tools",  0))
                    alp.metric("ALERTS", len(data.get("alerts", [])))
                    fp.metric("FPS",     f"{data.get('fps',0):.1f}")
                    for a in data.get("alerts", []):
                        log_alert(st.session_state.session_id, "live_stream", 0,
                                  a.get("type", "?"), a.get("tool_name", ""),
                                  float(a.get("timer", 0)), a.get("missing_ppe", []))
                elif msg[0] == "ERR":
                    st.error(msg[1])
                    st.session_state.live_active = False
                    break
                elif msg[0] == "LOG":
                    add_log(msg[1], "INFO")
            except queue.Empty:
                break
        st.rerun()
    else:
        st.markdown('<div class="ic in" style="text-align:center;padding:2rem;margin-top:1rem;">NO ACTIVE STREAM — Enter camera URL above and click CONNECT.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "analytics":
    st.markdown(section("// SESSION ANALYTICS"), unsafe_allow_html=True)

    comp = st.session_state.compliance
    status = ("EXCELLENT" if comp >= 90 else "GOOD" if comp >= 75
              else "NEEDS IMPROVEMENT" if comp >= 60 else "CRITICAL")
    sc = "cy" if comp >= 75 else ("ye" if comp >= 60 else "rd")

    a1, a2 = st.columns([1, 2])
    with a1:
        st.plotly_chart(compliance_gauge(comp), use_container_width=True, config={"displayModeBar": False})
    with a2:
        st.markdown(f"""
<div class="cmd-grid">
  {cmd_card("COMPLIANCE SCORE", f"{comp:.1f}%", "Violation rate based", sc)}
  {cmd_card("TOTAL FRAMES",     str(st.session_state.total_frames), "This session", "gn")}
  {cmd_card("TOTAL ALERTS",     str(st.session_state.total_alerts), "All violations", "rd" if st.session_state.total_alerts > 0 else "gn")}
  {cmd_card("STATUS",           status, "Safety assessment", sc)}
</div>""", unsafe_allow_html=True)
        if st.session_state.alert_log:
            fig = alerts_bar(st.session_state.alert_log)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(section("HISTORICAL ALERT VOLUME BY HOUR"), unsafe_allow_html=True)
    hourly = get_alert_counts_by_hour()
    if hourly:
        hdf  = pd.DataFrame(hourly, columns=["Hour", "Count"])
        fig2 = px.bar(hdf, x="Hour", y="Count", color_discrete_sequence=["#00ffcc"])
        fig2.update_layout(
            height=200, margin=dict(t=20, b=30, l=20, r=10),
            paper_bgcolor="#040806", plot_bgcolor="#040806",
            font=dict(family="Share Tech Mono", color="#a0c4b0"),
            xaxis=dict(gridcolor="#0d2b1e", title="HOUR", titlefont=dict(color="#3a6050")),
            yaxis=dict(gridcolor="#0d2b1e", title="ALERTS", titlefont=dict(color="#3a6050")),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="ic in" style="text-align:center;">No historical data yet — process a video to populate charts.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "history":
    st.markdown(section("// SESSION HISTORY"), unsafe_allow_html=True)
    sessions = get_all_sessions()
    if sessions:
        sdf = pd.DataFrame(sessions)
        st.dataframe(sdf, use_container_width=True, height=200)
    else:
        st.markdown('<div class="ic in">No previous sessions recorded yet.</div>', unsafe_allow_html=True)

    st.markdown(section("FULL ALERT HISTORY — ALL SESSIONS"), unsafe_allow_html=True)
    all_al = get_all_alerts()
    if all_al:
        adf = pd.DataFrame(all_al)
        st.dataframe(adf, use_container_width=True, height=360)
        st.download_button("⬇ DOWNLOAD HISTORY CSV", adf.to_csv(index=False),
                           "full_alert_history.csv", "text/csv")
    else:
        st.markdown('<div class="ic in">No alerts recorded yet across any session.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE: LOGS
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "logs":
    st.markdown(section("// SYSTEM EVENT LOG"), unsafe_allow_html=True)
    lf1, lf2 = st.columns([3, 1])
    with lf1:
        lvl = st.selectbox("FILTER", ["ALL", "INFO", "SUCCESS", "WARN", "ALERT"])
    with lf2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕ CLEAR", use_container_width=True):
            st.session_state.system_log = []; st.rerun()

    logs = st.session_state.system_log
    if lvl != "ALL":
        logs = [l for l in logs if l["level"] == lvl]

    if logs:
        cls_map  = {"INFO": "li", "SUCCESS": "ls", "WARN": "lw", "ALERT": "la"}
        log_html = ""
        for e in reversed(logs[-120:]):
            c   = cls_map.get(e["level"], "")
            log_html += (f'<span class="lt">[{e["time"]}]</span> '
                         f'<span class="{c}">[{e["level"].ljust(7)}]</span> '
                         f'{e["msg"]}<br>')
        st.markdown(f'<div class="logc">{log_html}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cmd-sub" style="margin-top:.4rem;">Showing {min(len(logs),120)} of {len(logs)} entries — newest first</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="ic in" style="text-align:center;padding:1.5rem;">No log entries to display.</div>', unsafe_allow_html=True)

    if st.session_state.system_log:
        st.download_button("⬇ DOWNLOAD LOG CSV",
                           pd.DataFrame(st.session_state.system_log).to_csv(index=False),
                           "system_log.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN monitor
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "train":
    st.markdown(section("// TRAINING MONITOR"), unsafe_allow_html=True)
    TOOLS_CSV = Path(r"E:\4TH YEAR PROJECT\NEW TOOLS\runs\detect\train_fast\results.csv")
    PPE_CSV   = Path(r"E:\4TH YEAR PROJECT\NEW PPE\runs\detect\train_v2_nano\results.csv")
    HUMAN_CSV = Path(r"E:\4TH YEAR PROJECT\HUMAN\runs\detect\train_fast\results.csv")

    def load_results(csv_path: Path):
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception:
            return None

    def training_panel(df, model_name: str, target: int, baseline: float):
        if df is None or df.empty:
            st.markdown(f'<div class="ic in">No training data for {model_name} yet.</div>', unsafe_allow_html=True)
            return
        total_ep  = len(df)
        best_map  = df.get("metrics/mAP50(B)", pd.Series([0])).max()
        last      = df.iloc[-1]
        box_loss  = float(last.get("train/box_loss", 0))
        cls_loss  = float(last.get("train/cls_loss", 0))
        pct       = total_ep / target

        st.markdown(f"""
<div class="cmd-grid">
  {cmd_card("EPOCH",       f"{total_ep}/{target}", f"{pct*100:.0f}% complete", "cy")}
  {cmd_card("BEST mAP@50", f"{best_map*100:.2f}%", "Best checkpoint",          "gn" if best_map>baseline else "ye")}
  {cmd_card("BOX LOSS",    f"{box_loss:.4f}",       "Lower is better",          "or")}
  {cmd_card("CLS LOSS",    f"{cls_loss:.4f}",       "Lower is better",          "or")}
</div>""", unsafe_allow_html=True)

        st.progress(min(pct, 1.0))

        lc1, lc2 = st.columns(2)
        with lc1:
            if "train/box_loss" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df["train/box_loss"], name="Box", line=dict(color="#ff6a00", width=1.5)))
                fig.add_trace(go.Scatter(y=df["train/cls_loss"], name="Cls", line=dict(color="#ff3333", width=1.5)))
                fig.update_layout(height=200, margin=dict(t=20,b=20,l=20,r=10),
                    paper_bgcolor="#040806", plot_bgcolor="#040806",
                    title=dict(text="TRAINING LOSSES", font=dict(size=9, color="#3a6050", family="Share Tech Mono")),
                    legend=dict(font=dict(color="#a0c4b0", size=9), bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#0d2b1e", tickfont=dict(color="#3a6050")),
                    yaxis=dict(gridcolor="#0d2b1e", tickfont=dict(color="#3a6050")))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with lc2:
            if "metrics/mAP50(B)" in df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=df["metrics/mAP50(B)"], name="mAP@50",
                    line=dict(color="#00ffcc", width=2), fill="tozeroy", fillcolor="rgba(0,255,204,.06)"))
                fig2.add_hline(y=baseline, line_dash="dash", line_color="#ff3333",
                    annotation_text=f"Baseline {baseline*100:.1f}%",
                    annotation_font_color="#ff3333")
                fig2.update_layout(height=200, margin=dict(t=20,b=20,l=20,r=10),
                    paper_bgcolor="#040806", plot_bgcolor="#040806",
                    title=dict(text="mAP@50 PROGRESS", font=dict(size=9, color="#3a6050", family="Share Tech Mono")),
                    legend=dict(font=dict(color="#a0c4b0", size=9), bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(gridcolor="#0d2b1e", tickfont=dict(color="#3a6050")),
                    yaxis=dict(gridcolor="#0d2b1e", tickfont=dict(color="#3a6050"), range=[0,1]))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    for csv_path, model_name, target, baseline in [
        (HUMAN_CSV, "HUMAN MODEL — 99.44% target",  75,  0.9944),
        (TOOLS_CSV, "TOOLS MODEL — 67.87% achieved",60,  0.6787),
        (PPE_CSV,   "PPE MODEL — 79.90% achieved",  200, 0.7990),
    ]:
        st.markdown(section(model_name), unsafe_allow_html=True)
        training_panel(load_results(csv_path), model_name, target, baseline)

    rc1, rc2 = st.columns([4, 1])
    with rc1:
        st.markdown(f'<div class="cmd-sub">Last refreshed: {datetime.datetime.now().strftime("%H:%M:%S")} — auto-refreshes every 30s</div>', unsafe_allow_html=True)
    with rc2:
        if st.button("↺ REFRESH", use_container_width=True):
            st.rerun()
    time.sleep(30); st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="sg-footer">
  <span>SafeGuard AI</span> &nbsp;·&nbsp; INDUSTRIAL SAFETY DETECTION SYSTEM &nbsp;·&nbsp;
  SESSION <span>{st.session_state.session_id}</span> &nbsp;·&nbsp;
  YOLOv11 &nbsp;·&nbsp; TOOL TRACKER &nbsp;·&nbsp; TEMPORAL FSM
</div>
""", unsafe_allow_html=True)
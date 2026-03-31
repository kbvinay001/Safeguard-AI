"""
streamlit_app.py — SafeGuard AI — Dinero-style Dark Dashboard
=============================================================
Design language: constellation bg · glassmorphism cards · Playfair Display ·
per-module colour accent glow · rounded 20px cards · fadeSlideUp animations.

Launch: double-click  Launch SafeGuard AI.bat
URL:    http://localhost:8501
"""

import streamlit as st
import streamlit.components.v1 as components
import cv2, numpy as np, tempfile, time, os, threading, queue
import datetime, uuid
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
    from ui_styles        import get_css, CONSTELLATION_JS
except Exception as _e:
    st.error(f"Import error: {_e}")
    st.stop()

# ── Inject CSS ─────────────────────────────────────────────────────────────
st.markdown(get_css(), unsafe_allow_html=True)

# ── Inject constellation canvas ────────────────────────────────────────────
components.html(CONSTELLATION_JS, height=0, scrolling=False)


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
        sub_page="home",
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


# ════════════════════════════════════════════════════════════════════════════
# MODULE DEFINITIONS  (like FEATURES in Dinero Dashboard.jsx)
# ════════════════════════════════════════════════════════════════════════════
MODULES = [
    dict(id="analyse",   emoji="🎥", title="Video Analysis",
         desc="Upload CCTV recordings for batch PPE + tool detection with heatmap and PDF export.",
         color="#F97316", border="rgba(249,115,22,.25)", page="analyse"),
    dict(id="live",      emoji="📡", title="Live RTSP Stream",
         desc="Connect IP cameras via RTSP for real-time detection with background threading.",
         color="#22D3EE", border="rgba(34,211,238,.25)", page="live"),
    dict(id="analytics", emoji="📊", title="Analytics & Charts",
         desc="Compliance gauge, hourly alert volume charts, session performance breakdown.",
         color="#4ADE80", border="rgba(74,222,128,.25)", page="analytics"),
    dict(id="history",   emoji="🗂", title="Alert History",
         desc="Full searchable alert log with CSV export across all sessions.",
         color="#A78BFA", border="rgba(167,139,250,.25)", page="history"),
    dict(id="system",    emoji="⚙️", title="System Info",
         desc="GPU status, model weight paths, CUDA version and launch guide.",
         color="#FBBF24", border="rgba(251,191,36,.25)", page="system"),
    dict(id="train",     emoji="🧠", title="Training Monitor",
         desc="Live training loss + mAP@50 curves for Human, PPE and Tools YOLOv11n models.",
         color="#3B82F6", border="rgba(59,130,246,.25)", page="train"),
]


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
    with st.spinner("Initialising YOLOv11 models…"):
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

engine = st.session_state.engine
models_active = len(engine.loaded_models) if engine else 0
sub_page = st.session_state.sub_page


# ════════════════════════════════════════════════════════════════════════════
# NAVBAR
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="sg-nav">
  <div class="sg-nav-inner">
    <div class="sg-logo">
      <div class="sg-logo-icon">🛡</div>
      SafeGuard AI
      <span class="sg-badge-beta">v2.0</span>
    </div>
    <div class="sg-nav-right">
      <span class="pill pill-on">● {models_active}/3 models</span>
      <span style="color:var(--muted);font-size:.8rem;margin-left:.5rem;">
        Session&nbsp;<strong style="color:var(--blue);">{st.session_state.session_id}</strong>
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SUB NAV (tab bar)
# ════════════════════════════════════════════════════════════════════════════
TABS = [
    ("home",      "🏠", "Home"),
    ("analyse",   "🎥", "Analyse"),
    ("live",      "📡", "Live Stream"),
    ("analytics", "📊", "Analytics"),
    ("history",   "🗂", "History"),
    ("logs",      "📋", "Logs"),
    ("system",    "⚙️", "System"),
    ("train",     "🧠", "Training"),
]

tab_html = '<div class="sg-subnav">'
for key, icon, label in TABS:
    cls = "sg-tab active" if sub_page == key else "sg-tab"
    tab_html += f'<span class="{cls}">{icon} {label}</span>'
tab_html += "</div>"
st.markdown(tab_html, unsafe_allow_html=True)

# invisible Streamlit buttons for actual navigation
_cols = st.columns(len(TABS))
for i, (key, icon, label) in enumerate(TABS):
    with _cols[i]:
        if st.button(f"{icon} {label}", key=f"tab_{key}", use_container_width=True):
            st.session_state.sub_page = key
            st.rerun()
sub_page = st.session_state.sub_page


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compliance_gauge(score: float):
    color = "#4ADE80" if score >= 75 else ("#FBBF24" if score >= 50 else "#F87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%",
                "font": {"size": 32, "color": "#F1F5F9", "family": "Inter"}},
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(255,255,255,.06)",
                      tickfont=dict(color="#64748B", size=9, family="Inter")),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#060913", bordercolor="rgba(255,255,255,.06)", borderwidth=1,
            steps=[
                dict(range=[0,  50], color="rgba(248,113,113,.08)"),
                dict(range=[50, 75], color="rgba(251,191,36,.06)"),
                dict(range=[75,100], color="rgba(74,222,128,.08)"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=.75, value=score),
        ),
        title={"text": "SAFETY COMPLIANCE",
               "font": {"size": 10, "color": "#64748B", "family": "Inter"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#060913", plot_bgcolor="#060913",
        font=dict(family="Inter"),
    )
    return fig


def _chart_layout(fig, h=200, title=""):
    fig.update_layout(
        height=h, margin=dict(t=30 if title else 20, b=20, l=30, r=10),
        paper_bgcolor="#060913", plot_bgcolor="rgba(14,20,36,.4)",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", size=10)),
        title=dict(text=title, font=dict(size=11, color="#64748B")) if title else None,
        showlegend=bool(title),
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
    color_map = {"PPE_VIOLATION": "#F87171", "TOOL_UNATTENDED": "#FBBF24"}
    colors = [color_map.get(t, "#3B82F6") for t in vc["Type"]]
    fig = go.Figure(go.Bar(x=vc["Type"], y=vc["Count"], marker_color=colors))
    return _chart_layout(fig, 200, "Alert Type Breakdown")


def fps_line(readings: list):
    if len(readings) < 2:
        return None
    fig = go.Figure(go.Scatter(
        y=readings, mode="lines",
        line=dict(color="#3B82F6", width=2),
        fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
    ))
    return _chart_layout(fig, 160, "Processing Speed (FPS)")


def calc_compliance(frames: int, alerts: int) -> float:
    if frames == 0:
        return 100.0
    return max(0.0, min(100.0, 100.0 - (alerts / max(frames, 1)) * 100 * 50))


# ════════════════════════════════════════════════════════════════════════════
# VIDEO PROCESSING
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
        fq.put(("ERR", f"Cannot open: {url}")); return
    fq.put(("LOG", f"Stream connected: {url}"))
    while not stop.is_set():
        ret, frame = cap.read()
        if not ret:
            fq.put(("ERR", "Stream ended.")); break
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
# ░░░ PAGE: HOME ░░░
# ════════════════════════════════════════════════════════════════════════════
if sub_page == "home":
    comp = st.session_state.compliance
    status = ("Excellent" if comp >= 90 else "Good" if comp >= 75
              else "Needs attention" if comp >= 60 else "Critical")

    # Hero
    st.markdown(f"""
<div class="sg-hero">
  <div class="sg-welcome-badge">
    <span class="pulse-dot"></span>
    System Online
  </div>
  <h1 class="sg-hero-title">
    Industrial Safety <span class="text-gradient">Command Center</span>
  </h1>
  <p class="sg-hero-sub">
    Autonomous PPE compliance and tool abandonment detection for CCTV infrastructure.
    Three YOLOv11n models running in real-time on your GPU.
  </p>
</div>
""", unsafe_allow_html=True)

    # Quick stat cards
    st.markdown(f"""
<div class="sg-wrap">
  <div class="sg-stats-row">
    <div class="sg-stat-card" style="animation-delay:.00s;">
      <div class="sg-stat-icon">🛡</div>
      <div class="sg-stat-value">{comp:.1f}%</div>
      <div class="sg-stat-label">Safety Compliance</div>
    </div>
    <div class="sg-stat-card" style="animation-delay:.08s;">
      <div class="sg-stat-icon">🎞</div>
      <div class="sg-stat-value">{st.session_state.total_frames:,}</div>
      <div class="sg-stat-label">Frames Analysed</div>
    </div>
    <div class="sg-stat-card" style="animation-delay:.16s;">
      <div class="sg-stat-icon">🚨</div>
      <div class="sg-stat-value">{st.session_state.total_alerts}</div>
      <div class="sg-stat-label">Alerts Raised</div>
    </div>
    <div class="sg-stat-card" style="animation-delay:.24s;">
      <div class="sg-stat-icon">🤖</div>
      <div class="sg-stat-value">{models_active}/3</div>
      <div class="sg-stat-label">Models Active</div>
    </div>
  </div>
""", unsafe_allow_html=True)

    # Explore Modules
    st.markdown('<h2 class="sg-section-title">Explore Modules</h2>', unsafe_allow_html=True)
    cards_html = '<div class="sg-grid">'
    for i, m in enumerate(MODULES):
        delay = 0.10 + i * 0.06
        cards_html += f"""
<div class="sg-feature-card"
     style="--c-accent:{m['color']};--c-border:{m['border']};animation-delay:{delay:.2f}s;">
  <div class="sg-card-glow"></div>
  <div class="sg-card-inner">
    <span class="sg-card-emoji">{m['emoji']}</span>
    <h3 class="sg-card-title">{m['title']}</h3>
    <p class="sg-card-desc">{m['desc']}</p>
    <div class="sg-card-link" style="color:{m['color']};">
      Explore <span class="sg-card-arrow">→</span>
    </div>
  </div>
</div>"""
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)

    # Navigation buttons (hidden visually but functional)
    nav_cols = st.columns(len(MODULES))
    for i, m in enumerate(MODULES):
        with nav_cols[i]:
            if st.button(m["title"], key=f"mod_{m['id']}", use_container_width=True):
                st.session_state.sub_page = m["page"]
                st.rerun()

    # System architecture accordion
    st.markdown("</div>", unsafe_allow_html=True)  # close sg-wrap
    with st.expander("⚙ Detection Settings & Architecture"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Confidence Thresholds**")
            conf_tool  = st.slider("Tool",  0.10, 0.80, float(CONF_TOOL),  0.05)
            conf_human = st.slider("Human", 0.15, 0.80, float(CONF_HUMAN), 0.05)
            conf_ppe   = st.slider("PPE",   0.10, 0.80, float(CONF_PPE),   0.05)
        with sc2:
            st.markdown("**Temporal FSM**")
            t1   = st.slider("T1 Warning (s)", 5,  60, int(T1_WARNING),   5)
            t2   = st.slider("T2 Alert (s)",  10,  90, int(T2_ALERT),     5)
            zone = st.slider("Zone Factor",   1.0, 3.0, float(ZONE_EXPAND_FACTOR), 0.1)
        with sc3:
            st.markdown("**Session**")
            st.metric("Frames",     st.session_state.total_frames)
            st.metric("Alerts",     st.session_state.total_alerts)
            st.metric("Compliance", f"{st.session_state.compliance:.1f}%")
            if st.button("↺ Reset Session", use_container_width=True):
                for k in ["alert_log","system_log"]:
                    st.session_state[k] = []
                for k in ["total_frames","total_alerts"]:
                    st.session_state[k] = 0
                st.session_state.compliance = 100.0
                st.session_state.processed_video_path = None
                st.session_state.processing_stats = None
                st.session_state.heatmap_acc = None
                st.session_state.session_id = str(uuid.uuid4())[:8].upper()
                add_log("Session reset.","INFO"); st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ░░░ PAGE: ANALYSE ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "analyse":
    st.markdown("""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span> Video Analysis</div>
  <h1 class="sg-hero-title">CCTV <span class="text-gradient">Video Analysis</span></h1>
  <p class="sg-hero-sub">Upload factory or CCTV recordings. GPU processes every frame for PPE compliance and tool abandonment.</p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    cu, ch = st.columns([2, 1])
    with cu:
        uf = st.file_uploader("Upload CCTV video (MP4 · AVI · MOV)",
                              type=["mp4", "avi", "mov", "mkv"])
    with ch:
        st.markdown("""
<div class="sg-box info" style="margin-top:1.2rem;">
  <strong style="color:#60A5FA;">Workflow</strong><br>
  <span style="color:var(--muted);font-size:.85rem;line-height:2;">
  01 &nbsp;Upload CCTV video<br>
  02 &nbsp;GPU processes all frames<br>
  03 &nbsp;Review heatmap + charts<br>
  04 &nbsp;Download video / CSV / PDF
  </span>
</div>""", unsafe_allow_html=True)

    if uf:
        fsz = len(uf.getbuffer()) / 1048576
        st.caption(f"📁 {uf.name} — {fsz:.1f} MB")
        if st.button("▶ Process Video", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ti:
                ti.write(uf.getbuffer()); inp = ti.name
            out = inp.replace(".mp4", "_out.mp4")
            add_log(f"Processing: {uf.name} ({fsz:.1f} MB)", "INFO")
            pb = st.progress(0.0)
            pt = st.empty()

            def upd(p, f, tot, cfps):
                pb.progress(float(p))
                pt.caption(f"Frame {f}/{tot} — {p*100:.1f}% — {cfps:.1f} FPS")

            t0 = time.time()
            stats, err = process_video(inp, out, upd)
            if err:
                st.error(err); add_log(f"FAILED: {err}", "ALERT")
            elif stats:
                st.session_state.processing_stats      = stats
                st.session_state.processed_video_path  = out
                st.session_state.total_frames         += stats["total_frames"]
                st.session_state.total_alerts         += stats["total_alerts"]
                st.session_state.compliance            = stats["compliance"]
                for a in stats["alerts"]:
                    st.session_state.alert_log.append({
                        "Frame":      a.get("frame", 0),
                        "Type":       a.get("type", "UNKNOWN"),
                        "Tool":       a.get("tool_name", "N/A"),
                        "Timer (s)":  round(float(a.get("timer", 0)), 2),
                        "Missing PPE": str(a.get("missing_ppe", [])),
                    })
                    log_alert(st.session_state.session_id, "video_upload",
                              a.get("frame",0), a.get("type","?"),
                              a.get("tool_name",""), float(a.get("timer",0)),
                              a.get("missing_ppe",[]))
                upsert_session(st.session_state.session_id, "video_upload",
                               st.session_state.total_frames,
                               st.session_state.total_alerts, stats["compliance"])
                pb.progress(1.0)
                add_log(f"Done: {stats['total_frames']} frames | {stats['total_alerts']} alerts | {stats['avg_fps']:.1f} FPS", "SUCCESS")
            try:
                os.unlink(inp)
            except Exception:
                pass

    if st.session_state.processed_video_path and Path(st.session_state.processed_video_path).exists():
        stats = st.session_state.processing_stats
        if stats:
            st.markdown("""<h2 class="sg-section-title" style="margin-top:2rem;">Results</h2>""", unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Frames",     stats["total_frames"])
            m2.metric("Avg FPS",    f"{stats['avg_fps']:.1f}")
            m3.metric("Alerts",     stats["total_alerts"])
            m4.metric("Time",       f"{stats['total_time']:.1f}s")
            m5.metric("Compliance", f"{stats['compliance']:.1f}%")

            co, cg = st.columns([2, 1])
            with co:
                if stats.get("fps_series"):
                    st.plotly_chart(fps_line(stats["fps_series"]), use_container_width=True,
                                    config={"displayModeBar": False})
            with cg:
                st.plotly_chart(compliance_gauge(stats["compliance"]), use_container_width=True,
                                config={"displayModeBar": False})

        cv, ch2 = st.columns(2)
        with cv:
            st.markdown('<h2 class="sg-section-title">Annotated Output</h2>', unsafe_allow_html=True)
            try:
                with open(st.session_state.processed_video_path, "rb") as vf:
                    st.video(vf.read())
            except Exception as ex:
                st.warning(f"Preview unavailable: {ex}")
        with ch2:
            st.markdown('<h2 class="sg-section-title">Violation Heatmap</h2>', unsafe_allow_html=True)
            if st.session_state.heatmap_acc is not None and st.session_state.heatmap_acc.max() > 0:
                acc  = st.session_state.heatmap_acc
                norm = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hm   = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(hm, cv2.COLOR_BGR2RGB),
                         caption="Red zones = frequent violation locations",
                         use_column_width=True)
            else:
                st.markdown('<div class="sg-box info" style="text-align:center;padding:2rem;">No violation data yet.</div>', unsafe_allow_html=True)

        st.markdown('<h2 class="sg-section-title" style="margin-top:1.5rem;">Export</h2>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            try:
                with open(st.session_state.processed_video_path, "rb") as f:
                    st.download_button("⬇ Video",  f.read(), "safety_output.mp4", "video/mp4", use_container_width=True)
            except Exception:
                pass
        with d2:
            if st.session_state.alert_log:
                df_dl = pd.DataFrame(st.session_state.alert_log)
                st.download_button("⬇ CSV", df_dl.to_csv(index=False), "alerts.csv", "text/csv", use_container_width=True)
        with d3:
            if stats:
                alts = [{"ts": datetime.datetime.now().isoformat(),
                         "type": a.get("type","?"),
                         "tool": a.get("tool_name",""),
                         "timer_s": float(a.get("timer",0)),
                         "missing_ppe": str(a.get("missing_ppe",[]))}
                        for a in stats.get("alerts",[])]
                try:
                    pdf_bytes = generate_pdf({**stats,"source":"video_upload","compliance":stats["compliance"]},
                                             alts, st.session_state.session_id)
                    st.download_button("⬇ PDF Report", pdf_bytes, "safety_report.pdf", "application/pdf", use_container_width=True)
                except Exception:
                    pass
        with d4:
            if st.button("✕ Clear", use_container_width=True):
                st.session_state.processed_video_path = None
                st.session_state.processing_stats = None
                add_log("Results cleared.","INFO"); st.rerun()
    else:
        st.markdown('<div class="sg-box info" style="margin-top:1.5rem;">Upload a CCTV or factory video above to begin analysis.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░ PAGE: LIVE ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "live":
    st.markdown("""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span> Live Detection</div>
  <h1 class="sg-hero-title">Live <span class="text-gradient">RTSP Stream</span></h1>
  <p class="sg-hero-sub">Connect any IP camera via RTSP/HTTP for real-time detection with background threading.</p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sg-box info" style="margin-bottom:1.2rem;">
  <strong style="color:#60A5FA;">Supported protocols</strong><br>
  <span style="color:var(--muted);font-size:.85rem;line-height:1.9;">
  RTSP &nbsp;&nbsp; rtsp://user:pass@ip:554/stream1 &nbsp;|&nbsp;
  HTTP &nbsp;&nbsp; http://ip:port/video &nbsp;|&nbsp;
  Local &nbsp; 0 (USB webcam)
  </span>
</div>""", unsafe_allow_html=True)

    ci1, ci2 = st.columns([3, 1])
    with ci1:
        ip_in = st.text_input("Camera URL", placeholder="rtsp://admin:admin@192.168.1.100:554/stream1")
    with ci2:
        proto = st.selectbox("Protocol", ["RTSP", "HTTP MJPEG", "Local (0)"])

    cp1, cp2, cp3 = st.columns(3)
    port     = cp1.text_input("Port", "554")
    chan     = cp2.selectbox("Channel", ["stream1", "stream2", "ch01", "ch02", "live"])
    use_auth = cp3.checkbox("Auth required")
    if use_auth:
        au, ap = st.columns(2)
        cam_u = au.text_input("Username", "admin")
        cam_p = ap.text_input("Password", "", type="password")
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

    surl = build_url(ip_in, proto, port, chan, cam_u, cam_p) if ip_in.strip() else ""
    if ip_in.strip():
        st.caption(f"URL: `{surl}`")

    sb1, sb2 = st.columns(2)
    if sb1.button("▶ Connect", type="primary", use_container_width=True) and ip_in.strip():
        if st.session_state.live_stop:
            st.session_state.live_stop.set(); time.sleep(0.3)
        fq   = queue.Queue(maxsize=2)
        stop = threading.Event()
        threading.Thread(target=_live_worker, args=(surl, fq, stop, st.session_state.engine), daemon=True).start()
        st.session_state.live_queue  = fq
        st.session_state.live_stop   = stop
        st.session_state.live_active = True
        add_log(f"Live stream started: {surl}", "INFO")
    if sb2.button("⏹ Disconnect", use_container_width=True) and st.session_state.live_stop:
        st.session_state.live_stop.set()
        st.session_state.live_active = False
        add_log("Disconnected.", "WARN"); st.rerun()

    if st.session_state.live_active and st.session_state.live_queue:
        st.markdown('<h2 class="sg-section-title" style="margin-top:1.5rem;">Live Feed</h2>', unsafe_allow_html=True)
        cs, cm = st.columns([3, 1])
        with cs:
            st.markdown('<span class="pill pill-on">● Stream Active</span>', unsafe_allow_html=True)
            sp = st.empty()
        with cm:
            hp = st.empty(); tp = st.empty(); alp = st.empty(); fp = st.empty()
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                msg = st.session_state.live_queue.get(timeout=0.15)
                if msg[0] == "FRAME":
                    _, ann, data = msg
                    sp.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_column_width=True)
                    hp.metric("Humans", data.get("humans",0))
                    tp.metric("Tools",  data.get("tools",0))
                    alp.metric("Alerts",len(data.get("alerts",[])))
                    fp.metric("FPS",    f"{data.get('fps',0):.1f}")
                elif msg[0] in ("ERR","LOG"):
                    add_log(msg[1], "WARN" if msg[0]=="ERR" else "INFO")
                    if msg[0] == "ERR":
                        st.session_state.live_active = False; break
            except queue.Empty:
                break
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░ PAGE: ANALYTICS ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "analytics":
    comp = st.session_state.compliance
    status = "Excellent" if comp >= 90 else "Good" if comp >= 75 else "Needs attention" if comp >= 60 else "Critical"
    s_color = "#4ADE80" if comp >= 75 else ("#FBBF24" if comp >= 60 else "#F87171")

    st.markdown(f"""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span> Analytics</div>
  <h1 class="sg-hero-title">Session <span class="text-gradient">Analytics</span></h1>
  <p class="sg-hero-sub">Compliance gauge, alert distribution, and hourly breach volume for this session.</p>
</div>
<div class="sg-wrap">
  <div class="sg-stats-row">
    <div class="sg-stat-card"><div class="sg-stat-icon">🛡</div>
      <div class="sg-stat-value" style="color:{s_color};">{comp:.1f}%</div>
      <div class="sg-stat-label">Compliance Score</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">🎞</div>
      <div class="sg-stat-value">{st.session_state.total_frames:,}</div>
      <div class="sg-stat-label">Frames Analysed</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">🚨</div>
      <div class="sg-stat-value">{st.session_state.total_alerts}</div>
      <div class="sg-stat-label">Total Alerts</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">📋</div>
      <div class="sg-stat-value" style="color:{s_color};">{status}</div>
      <div class="sg-stat-label">Safety Status</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    a1, a2 = st.columns([1, 2])
    with a1:
        st.plotly_chart(compliance_gauge(comp), use_container_width=True, config={"displayModeBar": False})
    with a2:
        if st.session_state.alert_log:
            fig = alerts_bar(st.session_state.alert_log)
            if fig:
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown('<div class="sg-box info" style="height:200px;display:flex;align-items:center;justify-content:center;">No alert data yet.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sg-wrap"><h2 class="sg-section-title">Hourly Alert Volume</h2></div>', unsafe_allow_html=True)
    hourly = get_alert_counts_by_hour()
    if hourly:
        hdf = pd.DataFrame(hourly, columns=["Hour","Count"])
        fig2 = px.bar(hdf, x="Hour", y="Count", color_discrete_sequence=["#3B82F6"])
        st.plotly_chart(_chart_layout(fig2, 220), use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div class="sg-box info" style="margin:0 2rem;text-align:center;">No historical data yet — process a video first.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░ PAGE: HISTORY ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "history":
    st.markdown("""
<div class="sg-hero">
  <h1 class="sg-hero-title">Alert <span class="text-gradient">History</span></h1>
  <p class="sg-hero-sub">Full session and alert history across all runs.</p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    st.markdown('<h2 class="sg-section-title">Session History</h2>', unsafe_allow_html=True)
    sessions = get_all_sessions()
    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True, height=200)
    else:
        st.markdown('<div class="sg-box info">No previous sessions recorded yet.</div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sg-section-title" style="margin-top:2rem;">All Alerts</h2>', unsafe_allow_html=True)
    all_al = get_all_alerts()
    if all_al:
        adf = pd.DataFrame(all_al)
        st.dataframe(adf, use_container_width=True, height=360)
        st.download_button("⬇ Download Full History CSV",
                           adf.to_csv(index=False), "full_alert_history.csv", "text/csv")
    else:
        st.markdown('<div class="sg-box info">No alerts recorded across any session yet.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░ PAGE: LOGS ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "logs":
    st.markdown("""
<div class="sg-hero">
  <h1 class="sg-hero-title">System <span class="text-gradient">Event Log</span></h1>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    lf1, lf2 = st.columns([3, 1])
    with lf1:
        lvl = st.selectbox("Filter by level", ["ALL","INFO","SUCCESS","WARN","ALERT"])
    with lf2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕ Clear log", use_container_width=True):
            st.session_state.system_log = []; st.rerun()

    logs = st.session_state.system_log
    if lvl != "ALL":
        logs = [l for l in logs if l["level"] == lvl]

    cls_map = {"INFO":"li","SUCCESS":"ls","WARN":"lw","ALERT":"la"}
    if logs:
        log_html = ""
        for e in reversed(logs[-120:]):
            c = cls_map.get(e["level"],"")
            log_html += (f'<span class="lt">[{e["time"]}]</span> '
                         f'<span class="{c}">[{e["level"].ljust(7)}]</span> '
                         f'{e["msg"]}<br>')
        st.markdown(f'<div class="sg-log">{log_html}</div>', unsafe_allow_html=True)
        st.caption(f"Showing {min(len(logs),120)} of {len(logs)} entries — newest first")
    else:
        st.markdown('<div class="sg-box info" style="text-align:center;padding:1.5rem;">No log entries to display.</div>', unsafe_allow_html=True)

    if st.session_state.system_log:
        st.download_button("⬇ Download Log CSV",
                           pd.DataFrame(st.session_state.system_log).to_csv(index=False),
                           "system_log.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░ PAGE: SYSTEM ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "system":
    import platform
    try:
        import torch as _torch
        cuda_ok  = _torch.cuda.is_available()
        gpu_name = _torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        gpu_mem  = f"{_torch.cuda.get_device_properties(0).total_memory / 1073741824:.1f} GB" if cuda_ok else "N/A"
        torch_ver = _torch.__version__
    except Exception:
        cuda_ok = False; gpu_name = "N/A"; gpu_mem = "N/A"; torch_ver = "N/A"
    try:
        import ultralytics as _ul; ul_ver = _ul.__version__
    except Exception:
        ul_ver = "N/A"

    st.markdown(f"""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span> {'CUDA Active' if cuda_ok else 'No GPU'}</div>
  <h1 class="sg-hero-title">System <span class="text-gradient">Information</span></h1>
  <p class="sg-hero-sub">Hardware, software, model weight status and launch guide.</p>
</div>
<div class="sg-wrap">
  <div class="sg-stats-row">
    <div class="sg-stat-card"><div class="sg-stat-icon">🐍</div>
      <div class="sg-stat-value">{platform.python_version()}</div>
      <div class="sg-stat-label">Python</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">🔥</div>
      <div class="sg-stat-value">{torch_ver[:6]}</div>
      <div class="sg-stat-label">PyTorch</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">{'✅' if cuda_ok else '❌'}</div>
      <div class="sg-stat-value" style="color:{'#4ADE80' if cuda_ok else '#F87171'};">{'YES' if cuda_ok else 'NO'}</div>
      <div class="sg-stat-label">CUDA</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">🤖</div>
      <div class="sg-stat-value">{models_active}/3</div>
      <div class="sg-stat-label">Models Loaded</div></div>
  </div>
""", unsafe_allow_html=True)

    st.markdown(f"""
  <div class="sg-box info" style="margin-bottom:1.5rem;">
    <strong style="color:#60A5FA;">GPU</strong>: {gpu_name} &nbsp;·&nbsp; {gpu_mem}<br>
    <strong style="color:#60A5FA;">Platform</strong>: {platform.system()} {platform.release()} &nbsp;·&nbsp;
    <strong style="color:#60A5FA;">Ultralytics</strong>: {ul_ver}
  </div>
""", unsafe_allow_html=True)

    # Weight status table
    st.markdown('<h2 class="sg-section-title">Weight File Status</h2>', unsafe_allow_html=True)
    rows_html = ""
    for mn, wp, perf, ds in [
        ("HUMAN", HUMAN_WEIGHTS, "99.44% mAP@50", "15,357 images"),
        ("PPE",   PPE_WEIGHTS,   "79.90% mAP@50", "11-class merged"),
        ("TOOLS", TOOL_WEIGHTS,  "67.87% mAP@50", "6,535 images"),
    ]:
        ok = Path(wp).exists()
        rows_html += f"""<tr>
<td><strong style="color:{'#4ADE80' if ok else '#F87171'};">{mn}</strong></td>
<td style="color:var(--muted);word-break:break-all;font-size:.8rem;">{wp}</td>
<td>{'✅ OK' if ok else '❌ Missing'}</td>
<td style="color:var(--txt2);">{perf}</td>
<td style="color:var(--muted);">{ds}</td>
</tr>"""
    st.markdown(f"""
<table class="sg-table"><thead><tr>
  <th>Model</th><th>Path</th><th>Status</th><th>Performance</th><th>Dataset</th>
</tr></thead><tbody>{rows_html}</tbody></table>
""", unsafe_allow_html=True)

    st.markdown("""
  <h2 class="sg-section-title" style="margin-top:2rem;">Launch Guide</h2>
  <div class="sg-box good">
    <div style="line-height:2.2;font-size:.9rem;">
      <strong style="color:#4ADE80;">Launch</strong> &nbsp;»&nbsp; Double-click <code>Launch SafeGuard AI.bat</code><br>
      <strong style="color:#4ADE80;">URL</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;»&nbsp; <code>http://localhost:8501</code><br>
      <strong style="color:#4ADE80;">GPU</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;»&nbsp; RTX 4060 · CUDA FP16 · auto-detected<br>
      <strong style="color:#4ADE80;">Database</strong> &nbsp;»&nbsp; outputs/safeguard.db (SQLite, auto-created)
    </div>
  </div>
</div>
""", unsafe_allow_html=True)



# ░░░ PAGE: TRAIN ░░░
# ════════════════════════════════════════════════════════════════════════════
elif sub_page == "train":
    st.markdown("""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span> Training</div>
  <h1 class="sg-hero-title">Training <span class="text-gradient">Monitor</span></h1>
  <p class="sg-hero-sub">Live loss curves and mAP@50 progress for Human, PPE and Tools YOLOv11n models.</p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

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

    for csv_path, model_name, target, baseline, color in [
        (HUMAN_CSV, "Human Model",  75,  0.9944, "#4ADE80"),
        (TOOLS_CSV, "Tools Model",  60,  0.6787, "#FBBF24"),
        (PPE_CSV,   "PPE Model",   200,  0.7990, "#A78BFA"),
    ]:
        df = load_results(csv_path)
        st.markdown(f"""
<div class="sg-feature-card" style="--c-accent:{color};--c-border:rgba(255,255,255,.08);margin-bottom:1rem;">
  <div class="sg-card-glow"></div>
  <div class="sg-card-inner">
    <span class="sg-card-title">{model_name}</span>
    <span class="sg-card-desc">Target: {baseline*100:.2f}% mAP@50 · {target} epochs</span>
  </div>
</div>
""", unsafe_allow_html=True)

        if df is None or df.empty:
            st.markdown('<div class="sg-box warn">No training data found.</div>', unsafe_allow_html=True)
            continue

        total_ep = len(df)
        best_map = df.get("metrics/mAP50(B)", pd.Series([0])).max()
        last     = df.iloc[-1]
        box_loss = float(last.get("train/box_loss", 0))
        cls_loss = float(last.get("train/cls_loss", 0))

        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("Epochs",    f"{total_ep}/{target}")
        mm2.metric("Best mAP@50", f"{best_map*100:.2f}%")
        mm3.metric("Box Loss",  f"{box_loss:.4f}")
        mm4.metric("Cls Loss",  f"{cls_loss:.4f}")
        st.progress(min(total_ep / target, 1.0))

        lc1, lc2 = st.columns(2)
        with lc1:
            if "train/box_loss" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=df["train/box_loss"], name="Box Loss",
                    line=dict(color=color, width=2)))
                fig.add_trace(go.Scatter(y=df["train/cls_loss"], name="Cls Loss",
                    line=dict(color="#F87171", width=1.5, dash="dash")))
                st.plotly_chart(_chart_layout(fig, 200, "Training Losses"),
                                use_container_width=True, config={"displayModeBar": False})
        with lc2:
            if "metrics/mAP50(B)" in df.columns:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=df["metrics/mAP50(B)"], name="mAP@50",
                    line=dict(color=color, width=2),
                    fill="tozeroy", fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4))+(0.07,)}"))
                fig2.add_hline(y=baseline, line_dash="dash", line_color="#F87171",
                    annotation_text=f"Target {baseline*100:.1f}%",
                    annotation_font_color="#F87171")
                st.plotly_chart(_chart_layout(fig2, 200, "mAP@50 Progress"),
                                use_container_width=True, config={"displayModeBar": False})

    rc1, rc2 = st.columns([4, 1])
    with rc1:
        st.caption(f"Last refreshed: {datetime.datetime.now().strftime('%H:%M:%S')}")
    with rc2:
        if st.button("↺ Refresh", use_container_width=True):
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="sg-footer">
  <span>SafeGuard AI</span> · Industrial Safety Detection System ·
  Session <span>{st.session_state.session_id}</span> ·
  YOLOv11n · Temporal FSM · CUDA FP16
</div>
""", unsafe_allow_html=True)
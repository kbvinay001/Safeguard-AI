"""
streamlit_app.py — SafeGuard AI — Dinero-style Dark Dashboard v2
=================================================================
Navigation:  Module cards on Home → each card navigates to its page.
             "← Back" button on every inner page returns to Home.
No duplicate button rows. No tab strip. Cards are the nav.

Launch: double-click  Launch SafeGuard AI.bat
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

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from detection_engine import SafetyDetectionEngine
    from safety_config    import *
    from db_manager       import (log_alert, upsert_session,
                                  get_all_alerts, get_all_sessions,
                                  get_alert_counts_by_hour)
    from pdf_report       import generate_pdf
    from ui_styles        import get_css, CONSTELLATION_JS, ICONS, STAT_ICONS
except Exception as _e:
    st.error(f"Import error: {_e}")
    st.stop()

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown(get_css(), unsafe_allow_html=True)

# ── Constellation canvas (injects into parent document via iframe JS) ──────
components.html(CONSTELLATION_JS, height=0, scrolling=False)


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
_defaults = dict(
    engine=None, alert_log=[], total_frames=0, total_alerts=0,
    processed_video_path=None, processing_stats=None,
    system_log=[], session_id=str(uuid.uuid4())[:8].upper(),
    heatmap_acc=None, live_active=False,
    live_queue=None, live_stop=None,
    compliance=100.0,
    page="home",   # "home" | "analyse" | "live" | "analytics" | "history" | "logs" | "system" | "train"
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def add_log(msg: str, level: str = "INFO"):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.system_log.append({"time": ts, "level": level, "msg": msg})
    if len(st.session_state.system_log) > 300:
        st.session_state.system_log = st.session_state.system_log[-300:]

def go_page(p: str):
    st.session_state.page = p
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# MODULE DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════
MODULES = [
    dict(id="analyse",   title="Video Analysis",
         desc="Upload CCTV recordings for batch PPE + tool detection with heatmap and PDF export.",
         color="#F97316", acc="acc-orange"),
    dict(id="live",      title="Live RTSP Stream",
         desc="Connect IP cameras via RTSP for real-time detection with background threading.",
         color="#22D3EE", acc="acc-cyan"),
    dict(id="analytics", title="Analytics & Charts",
         desc="Safety compliance gauge, hourly alert volume charts, session performance breakdown.",
         color="#4ADE80", acc="acc-green"),
    dict(id="history",   title="Alert History",
         desc="Full searchable alert log with CSV export across all sessions.",
         color="#A78BFA", acc="acc-purple"),
    dict(id="system",    title="System Info",
         desc="GPU status, model weight paths, CUDA version and launch guide.",
         color="#FBBF24", acc="acc-yellow"),
    dict(id="train",     title="Training Monitor",
         desc="Live training loss + mAP@50 curves for Human, PPE and Tools YOLOv11n models.",
         color="#3B82F6", acc="acc-blue"),
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

engine       = st.session_state.engine
models_active = len(engine.loaded_models) if engine else 0
page         = st.session_state.page


# ════════════════════════════════════════════════════════════════════════════
# NAVBAR  (always visible)
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="sg-nav">
  <div class="sg-nav-inner">
    <div class="sg-logo">
      <div class="sg-logo-icon">🛡</div>
      SafeGuard AI
      <span class="sg-badge">v2.0</span>
    </div>
    <div class="sg-nav-right">
      <span class="pill-on">● {models_active}/3 models</span>
      <span style="color:var(--muted);font-size:.8rem;">
        Session&nbsp;<strong style="color:#3B82F6;">{st.session_state.session_id}</strong>
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Back button (only on inner pages, not home) ────────────────────────────
if page != "home":
    st.markdown('<div class="sg-backrow">', unsafe_allow_html=True)
    if st.button("← Home", key="back_home"):
        go_page("home")
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════
def compliance_gauge(score: float):
    color = "#4ADE80" if score >= 75 else ("#FBBF24" if score >= 50 else "#F87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 32, "color": "#F1F5F9", "family": "Inter"}},
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="rgba(255,255,255,.05)",
                      tickfont=dict(color="#64748B", size=9)),
            bar=dict(color=color, thickness=0.22),
            bgcolor="#060913", bordercolor="rgba(255,255,255,.05)", borderwidth=1,
            steps=[
                dict(range=[0,  50], color="rgba(248,113,113,.08)"),
                dict(range=[50, 75], color="rgba(251,191,36,.06)"),
                dict(range=[75,100], color="rgba(74,222,128,.08)"),
            ],
            threshold=dict(line=dict(color=color, width=2), thickness=.75, value=score),
        ),
        title={"text":"SAFETY COMPLIANCE","font":{"size":10,"color":"#64748B","family":"Inter"}},
        domain={"x":[0,1],"y":[0,1]},
    ))
    fig.update_layout(height=260, margin=dict(t=40,b=10,l=20,r=20),
                       paper_bgcolor="#060913", plot_bgcolor="#060913")
    return fig


def _chart_layout(fig, h=200, title=""):
    fig.update_layout(
        height=h, margin=dict(t=30 if title else 20,b=20,l=30,r=10),
        paper_bgcolor="#060913", plot_bgcolor="rgba(14,20,36,.4)",
        font=dict(family="Inter", color="#94A3B8"),
        xaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,.04)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", size=10)),
        title=dict(text=title, font=dict(size=11,color="#64748B")) if title else None,
    )
    return fig


def calc_compliance(frames: int, alerts: int) -> float:
    return max(0.0, min(100.0, 100.0 - (alerts / max(frames,1)) * 100 * 50))


def process_video(input_path: str, output_path: str, progress_cb=None):
    eng = st.session_state.engine
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, "Cannot open video"
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
    all_alerts, fps_readings, frame_count = [], [], 0
    heatmap_acc = np.zeros((h,w), dtype=np.float32)
    t0 = time.time()
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            ft = time.time()
            try: annotated, data = eng.process_frame(frame)
            except Exception:
                annotated = frame
                data = {"fps":0,"humans":0,"tools":0,"alerts":[],"tracked_tools":{}}
            writer.write(annotated)
            frame_count += 1
            dt = time.time() - ft
            if dt > 0: fps_readings.append(1.0/dt)
            for a in data.get("alerts",[]):
                a["frame"] = frame_count
                all_alerts.append(a)
                cy, cx = h//2, w//2
                heatmap_acc[max(0,cy-60):min(h,cy+60), max(0,cx-60):min(w,cx+60)] += 1.0
            if progress_cb and frame_count % 8 == 0:
                progress_cb(frame_count/max(total,1), frame_count, total,
                            fps_readings[-1] if fps_readings else 0.0)
    finally:
        cap.release(); writer.release()
    st.session_state.heatmap_acc = heatmap_acc
    return {
        "total_frames": frame_count, "total_alerts": len(all_alerts),
        "avg_fps": float(np.mean(fps_readings)) if fps_readings else 0.0,
        "total_time": time.time()-t0,
        "alerts": all_alerts, "fps_series": fps_readings[::10],
        "compliance": calc_compliance(frame_count, len(all_alerts)),
    }, None


def _live_worker(url, fq, stop, eng):
    """
    Background thread that reads from an RTSP/HTTP/local camera stream,
    runs inference on each frame, and pushes results to the queue.

    Hardened behaviour:
      1. Up to MAX_RECONNECT_ATTEMPTS reconnection attempts with exponential backoff.
      2. If all reconnects fail, tries to play the fallback_demo.mp4 video instead.
      3. If the fallback is also missing, sends a clean "Feed Interrupted" frame
         generated with OpenCV (no Python traceback shown to the user).
    """
    from safety_config import MAX_RECONNECT_ATTEMPTS, FALLBACK_VIDEO

    def make_interrupted_frame(msg="FEED INTERRUPTED"):
        """Generate a clean black error frame instead of showing a traceback."""
        h, w = 480, 854
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, msg, (w//2 - 180, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 2)
        cv2.putText(img, "Check camera connection and URL", (w//2 - 220, h//2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 1)
        return img

    # --- Attempt connection with retries ---
    cap = None
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        fq.put(("LOG", f"Connecting (attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}): {url}"))
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                fq.put(("LOG", f"Stream connected: {url}"))
                break
        except Exception as e:
            fq.put(("LOG", f"Connection attempt {attempt} failed: {e}"))
        cap = None
        if attempt < MAX_RECONNECT_ATTEMPTS:
            time.sleep(2 ** attempt)   # Exponential back-off: 2s, 4s

    # --- Stream loop ---
    if cap and cap.isOpened():
        drop_count = 0
        while not stop.is_set():
            ret, frame = cap.read()
            if not ret:
                drop_count += 1
                if drop_count >= 5:
                    fq.put(("RECONNECTING", "Stream dropped — attempting reconnect…"))
                    cap.release()
                    cap = None
                    # Try to reconnect once
                    try:
                        cap = cv2.VideoCapture(url)
                        if cap.isOpened():
                            drop_count = 0
                            fq.put(("LOG", "Stream reconnected."))
                            continue
                    except Exception:
                        pass
                    break  # Give up on live stream, fall through to fallback
                continue
            drop_count = 0
            try:
                ann, data = eng.process_frame(frame)
            except Exception:
                ann = frame
                data = {"fps": 0, "humans": 0, "tools": 0, "alerts": []}
            # Replace oldest frame in queue (maxsize=2 keeps it fresh)
            try:
                fq.get_nowait()
            except Exception:
                pass
            fq.put(("FRAME", ann, data))
        if cap:
            cap.release()

    # --- Fallback video or interrupted frame ---
    if not stop.is_set():
        if FALLBACK_VIDEO.exists():
            fq.put(("LOG", f"Switching to fallback video: {FALLBACK_VIDEO.name}"))
            fallback_cap = cv2.VideoCapture(str(FALLBACK_VIDEO))
            while not stop.is_set() and fallback_cap.isOpened():
                ret, frame = fallback_cap.read()
                if not ret:
                    fallback_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop fallback
                    continue
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
            fallback_cap.release()
        else:
            # No fallback video — show a clean interrupted frame
            interrupted = make_interrupted_frame()
            fq.put(("FRAME", interrupted, {"fps": 0, "humans": 0, "tools": 0, "alerts": []}))
            fq.put(("ERR", "Camera unavailable. No fallback video found."))

    fq.put(("LOG", "Stream stopped."))


# ════════════════════════════════════════════════════════════════════════════
# ░░░  HOME PAGE  ░░░
# ════════════════════════════════════════════════════════════════════════════
if page == "home":
    comp = st.session_state.compliance

    # ── Hero — big inline heading ─────────────────────────────────────────
    st.markdown(f"""
<div class="sg-hero">
  <div class="sg-welcome-badge">
    <span class="pulse-dot"></span>System Online &nbsp;&mdash;&nbsp; {models_active}/3 models active
  </div>
  <h1 style="
    font-family:'Playfair Display',Georgia,serif;
    font-size:clamp(3.2rem,6.5vw,5.2rem);
    font-weight:800; line-height:1.05;
    letter-spacing:-0.035em; color:#ffffff;
    margin:0 0 1rem 0;
    animation:fadeSlideUp .6s .1s ease both;
  ">
    Industrial Safety<br>
    <span style="background:linear-gradient(135deg,#3B82F6,#22C55E);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
      Command Center
    </span>
  </h1>
  <p class="sg-hero-sub">
    Autonomous PPE compliance and tool abandonment detection for CCTV
    infrastructure &mdash; three YOLOv11n models fused in real-time on GPU.
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Stat row with SVG icons ───────────────────────────────────────────
    st.markdown(f"""
<div class="sg-wrap">
  <div class="sg-stats-row">
    <div class="sg-stat-card" style="animation-delay:.00s">
      <div class="sg-stat-icon">{STAT_ICONS['shield']}</div>
      <div class="sg-stat-value">{comp:.1f}%</div>
      <div class="sg-stat-label">Safety Compliance</div>
    </div>
    <div class="sg-stat-card" style="animation-delay:.08s">
      <div class="sg-stat-icon">🎞️</div>
      <div class="sg-stat-value">{st.session_state.total_frames:,}</div>
      <div class="sg-stat-label">Frames Analysed</div>
    </div>
    <div class="sg-stat-card" style="animation-delay:.16s">
      <div class="sg-stat-icon">🚨</div>
      <div class="sg-stat-value">{st.session_state.total_alerts}</div>
      <div class="sg-stat-label">Alerts Raised</div>
    </div>
    <div class="sg-stat-card" style="animation-delay:.24s">
      <div class="sg-stat-icon">🤖</div>
      <div class="sg-stat-value">{models_active}/3</div>
      <div class="sg-stat-label">Models Active</div>
    </div>
  </div>

  <h2 class="sg-section-title">Explore Modules</h2>
</div>
""", unsafe_allow_html=True)

    # ── Module cards: HTML visual + Explore button ────────────────────────
    def _card_row(mods, row_idx):
        cols = st.columns(3, gap="large")
        for i, mod in enumerate(mods):
            svg   = ICONS.get(mod["id"], "")
            delay = (row_idx * 3 + i) * 0.08
            with cols[i]:
                st.markdown(f"""
<div class="mod-card-visual {mod['acc']}" style="animation-delay:{delay:.2f}s;">
  <div class="mod-icon-wrap">{svg}</div>
  <div class="mod-card-title">{mod['title']}</div>
  <div class="mod-card-desc">{mod['desc']}</div>
</div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="mod-explore {mod["acc"]}">', unsafe_allow_html=True)
                if st.button(f"Explore {mod['title']} \u2192",
                             key=f"mod_{mod['id']}", use_container_width=True):
                    go_page(mod["id"])
                st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="max-width:1240px;margin:0 auto;padding:0 2.5rem">', unsafe_allow_html=True)
    _card_row(MODULES[:3], 0)
    st.markdown("<br>", unsafe_allow_html=True)
    _card_row(MODULES[3:], 1)
    st.markdown("</div>", unsafe_allow_html=True)


    # ── Settings expander ─────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Detection Settings & Session Controls"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Confidence Thresholds**")
            st.slider("Tool",  0.10, 0.80, float(CONF_TOOL),  0.05)
            st.slider("Human", 0.15, 0.80, float(CONF_HUMAN), 0.05)
            st.slider("PPE",   0.10, 0.80, float(CONF_PPE),   0.05)
        with sc2:
            st.markdown("**Temporal FSM**")
            st.slider("T1 Warning (s)", 5,  60, int(T1_WARNING), 5)
            st.slider("T2 Alert (s)",  10,  90, int(T2_ALERT),   5)
        with sc3:
            st.markdown("**Session**")
            st.metric("Frames",     st.session_state.total_frames)
            st.metric("Alerts",     st.session_state.total_alerts)
            st.metric("Compliance", f"{st.session_state.compliance:.1f}%")
            if st.button("\u21ba Reset Session", use_container_width=True):
                for k in ["alert_log","system_log"]: st.session_state[k] = []
                for k in ["total_frames","total_alerts"]: st.session_state[k] = 0
                st.session_state.compliance = 100.0
                st.session_state.processed_video_path = None
                st.session_state.processing_stats = None
                st.session_state.heatmap_acc = None
                st.session_state.session_id = str(uuid.uuid4())[:8].upper()
                add_log("Session reset.", "INFO"); st.rerun()




# ════════════════════════════════════════════════════════════════════════════
# ░░░  ANALYSE  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "analyse":
    st.markdown("""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span>Video Analysis</div>
  <h1 class="sg-hero-title">
    CCTV <span class="text-gradient">Video Analysis</span>
  </h1>
  <p class="sg-hero-sub">
    Upload factory or CCTV recordings. GPU processes every frame for PPE
    compliance and tool abandonment detection.
  </p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    cu, ch = st.columns([2, 1])
    with cu:
        uf = st.file_uploader("Upload CCTV video (MP4 · AVI · MOV)",
                              type=["mp4","avi","mov","mkv"])
    with ch:
        st.markdown("""
<div class="sg-box info" style="margin-top:1.2rem;">
  <strong style="color:#60A5FA;">Workflow</strong><br>
  <span style="color:var(--muted);font-size:.85rem;line-height:2.1;">
  01 &nbsp;Upload CCTV video<br>
  02 &nbsp;GPU processes all frames<br>
  03 &nbsp;Review heatmap + charts<br>
  04 &nbsp;Export video / CSV / PDF
  </span>
</div>""", unsafe_allow_html=True)

    if uf:
        fsz = len(uf.getbuffer()) / 1048576
        st.caption(f"📁 {uf.name} — {fsz:.1f} MB")
        if st.button("▶ Process Video", type="primary", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ti:
                ti.write(uf.getbuffer()); inp = ti.name
            out = inp.replace(".mp4","_out.mp4")
            add_log(f"Processing: {uf.name} ({fsz:.1f} MB)", "INFO")
            pb = st.progress(0.0); pt = st.empty()
            def upd(p,f,tot,cfps): pb.progress(float(p)); pt.caption(f"Frame {f}/{tot} — {p*100:.1f}% — {cfps:.1f} FPS")
            stats, err = process_video(inp, out, upd)
            if err:
                st.error(err); add_log(f"FAILED: {err}", "ALERT")
            elif stats:
                st.session_state.processing_stats     = stats
                st.session_state.processed_video_path = out
                st.session_state.total_frames        += stats["total_frames"]
                st.session_state.total_alerts        += stats["total_alerts"]
                st.session_state.compliance           = stats["compliance"]
                for a in stats["alerts"]:
                    st.session_state.alert_log.append({
                        "Frame": a.get("frame",0), "Type": a.get("type","?"),
                        "Tool": a.get("tool_name","N/A"),
                        "Timer (s)": round(float(a.get("timer",0)),2),
                        "Missing PPE": str(a.get("missing_ppe",[])),
                    })
                    log_alert(st.session_state.session_id,"video_upload",
                              a.get("frame",0),a.get("type","?"),
                              a.get("tool_name",""),float(a.get("timer",0)),
                              a.get("missing_ppe",[]))
                upsert_session(st.session_state.session_id,"video_upload",
                               st.session_state.total_frames,
                               st.session_state.total_alerts,stats["compliance"])
                pb.progress(1.0)
                add_log(f"Done: {stats['total_frames']} frames | {stats['total_alerts']} alerts | {stats['avg_fps']:.1f} FPS","SUCCESS")
            try: os.unlink(inp)
            except Exception: pass

    if (st.session_state.processed_video_path
            and Path(st.session_state.processed_video_path).exists()):
        stats = st.session_state.processing_stats
        if stats:
            st.markdown('<h2 class="sg-section-title" style="margin-top:2rem;">Results</h2>', unsafe_allow_html=True)
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Frames",     stats["total_frames"])
            m2.metric("Avg FPS",    f"{stats['avg_fps']:.1f}")
            m3.metric("Alerts",     stats["total_alerts"])
            m4.metric("Time",       f"{stats['total_time']:.1f}s")
            m5.metric("Compliance", f"{stats['compliance']:.1f}%")

            co,cg = st.columns([2,1])
            with co:
                if stats.get("fps_series"):
                    fig = go.Figure(go.Scatter(y=stats["fps_series"], mode="lines",
                        line=dict(color="#3B82F6",width=2),fill="tozeroy",
                        fillcolor="rgba(59,130,246,.07)"))
                    st.plotly_chart(_chart_layout(fig,160,"Processing Speed (FPS)"),
                                    use_container_width=True, config={"displayModeBar":False})
            with cg:
                st.plotly_chart(compliance_gauge(stats["compliance"]),
                                use_container_width=True, config={"displayModeBar":False})

        cv2c, ch2 = st.columns(2)
        with cv2c:
            st.markdown('<h2 class="sg-section-title">Annotated Output</h2>', unsafe_allow_html=True)
            try:
                with open(st.session_state.processed_video_path,"rb") as vf:
                    st.video(vf.read())
            except Exception as ex:
                st.warning(f"Preview unavailable: {ex}")
        with ch2:
            st.markdown('<h2 class="sg-section-title">Violation Heatmap</h2>', unsafe_allow_html=True)
            if (st.session_state.heatmap_acc is not None
                    and st.session_state.heatmap_acc.max() > 0):
                acc  = st.session_state.heatmap_acc
                norm = cv2.normalize(acc,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                hm   = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                st.image(cv2.cvtColor(hm,cv2.COLOR_BGR2RGB),
                         caption="Red = frequent violation areas", use_column_width=True)
            else:
                st.markdown('<div class="sg-box info" style="text-align:center;padding:2rem;">No violation data yet.</div>', unsafe_allow_html=True)

        st.markdown('<h2 class="sg-section-title" style="margin-top:1.5rem;">Export</h2>', unsafe_allow_html=True)
        d1,d2,d3,d4 = st.columns(4)
        with d1:
            try:
                with open(st.session_state.processed_video_path,"rb") as f:
                    st.download_button("⬇ Video",f.read(),"safety_output.mp4","video/mp4",use_container_width=True)
            except Exception: pass
        with d2:
            if st.session_state.alert_log:
                st.download_button("⬇ CSV",pd.DataFrame(st.session_state.alert_log).to_csv(index=False),
                                   "alerts.csv","text/csv",use_container_width=True)
        with d3:
            if stats:
                try:
                    alts = [{"ts":datetime.datetime.now().isoformat(),
                             "type":a.get("type","?"),"tool":a.get("tool_name",""),
                             "timer_s":float(a.get("timer",0)),
                             "missing_ppe":str(a.get("missing_ppe",[]))}
                            for a in stats.get("alerts",[])]
                    pdf_bytes = generate_pdf({**stats,"source":"video_upload","compliance":stats["compliance"]},
                                             alts, st.session_state.session_id)
                    st.download_button("⬇ PDF Report",pdf_bytes,"safety_report.pdf","application/pdf",use_container_width=True)
                except Exception: pass
        with d4:
            if st.button("✕ Clear",use_container_width=True):
                st.session_state.processed_video_path = None
                st.session_state.processing_stats = None
                add_log("Results cleared.","INFO"); st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░  LIVE RTSP  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "live":
    st.markdown("""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span>Live Detection</div>
  <h1 class="sg-hero-title">Live <span class="text-gradient">RTSP Stream</span></h1>
  <p class="sg-hero-sub">Connect any IP camera via RTSP/HTTP for real-time detection with background threading.</p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sg-box info" style="margin-bottom:1.2rem;">
  <strong style="color:#60A5FA;">Supported URLs</strong><br>
  <span style="color:var(--muted);font-size:.85rem;line-height:2;">
  RTSP &nbsp;→&nbsp; rtsp://user:pass@ip:554/stream1 &nbsp;|&nbsp; HTTP &nbsp;→&nbsp; http://ip:port/video &nbsp;|&nbsp; Local &nbsp;→&nbsp; 0
  </span>
</div>""", unsafe_allow_html=True)

    ci1, ci2 = st.columns([3,1])
    with ci1:
        ip_in = st.text_input("Camera URL", placeholder="rtsp://admin:admin@192.168.1.100:554/stream1")
    with ci2:
        proto = st.selectbox("Protocol", ["RTSP","HTTP MJPEG","Local (0)"])

    use_auth = st.checkbox("Requires authentication")
    if use_auth:
        au, ap = st.columns(2)
        cam_u = au.text_input("Username","admin")
        cam_p = ap.text_input("Password","",type="password")
    else:
        cam_u = cam_p = ""

    def build_url(ip,proto,u,p):
        r = ip.strip()
        if r.startswith(("rtsp://","http://","https://")): return r
        try: return int(r)
        except: pass
        if proto=="RTSP":
            return f"rtsp://{u}:{p}@{r}:554/stream1" if u else f"rtsp://{r}:554/stream1"
        return f"http://{r}:80/video"

    surl = build_url(ip_in,proto,cam_u,cam_p) if ip_in.strip() else ""
    if ip_in.strip(): st.caption(f"URL: `{surl}`")

    sb1, sb2 = st.columns(2)
    if sb1.button("▶ Connect",type="primary",use_container_width=True) and ip_in.strip():
        if st.session_state.live_stop: st.session_state.live_stop.set(); time.sleep(.3)
        fq = queue.Queue(maxsize=2); stop = threading.Event()
        threading.Thread(target=_live_worker,args=(surl,fq,stop,engine),daemon=True).start()
        st.session_state.live_queue=fq; st.session_state.live_stop=stop; st.session_state.live_active=True
        add_log(f"Live stream started: {surl}","INFO")
    if sb2.button("⏹ Disconnect",use_container_width=True) and st.session_state.live_stop:
        st.session_state.live_stop.set(); st.session_state.live_active=False
        add_log("Disconnected.","WARN"); st.rerun()

    if st.session_state.live_active and st.session_state.live_queue:
        # ── HUD-style live feed display ──────────────────────────────────────
        st.markdown('<h2 class="sg-section-title" style="margin-top:1.5rem;">Live Feed</h2>', unsafe_allow_html=True)
        cs, cm = st.columns([3, 1])
        with cs:
            feed_placeholder  = st.empty()
            status_placeholder = st.empty()
        with cm:
            st.markdown("<div style='padding-top:.5rem'>", unsafe_allow_html=True)
            hp  = st.empty()  # humans metric
            tp  = st.empty()  # tools metric
            alp = st.empty()  # alerts metric
            fp  = st.empty()  # fps metric
            st.markdown("</div>", unsafe_allow_html=True)

        deadline = time.time() + 5.0
        last_fsm = "SAFE"
        while time.time() < deadline:
            try:
                msg = st.session_state.live_queue.get(timeout=.15)
                if msg[0] == "FRAME":
                    _, ann, data = msg
                    # Determine FSM state from active tools
                    tool_states = [t.get("state","SAFE") for t in data.get("tracked_tools",{}).values()]
                    if "ALERT"   in tool_states: last_fsm = "ALERT"
                    elif "WARNING" in tool_states: last_fsm = "WARNING"
                    else:                          last_fsm = "SAFE"
                    fsm_color  = {"SAFE":"#4ADE80","WARNING":"#FBBF24","ALERT":"#F87171"}.get(last_fsm,"#4ADE80")
                    fsm_label  = {"SAFE":"● SAFE","WARNING":"⚠ WARNING","ALERT":"🚨 ALERT"}.get(last_fsm,"● SAFE")
                    status_placeholder.markdown(
                        f"""<div class="sg-hud-status">
                            <span class="sg-stream-live">● LIVE</span>
                            <span class="sg-fsm-badge" style="color:{fsm_color};border-color:{fsm_color};">{fsm_label}</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    feed_placeholder.image(
                        cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                        use_column_width=True
                    )
                    hp.metric("👷 Humans",  data.get("humans", 0))
                    tp.metric("🔧 Tools",   data.get("tools",  0))
                    alp.metric("🚨 Alerts", len(data.get("alerts", [])))
                    fp.metric("⚡ FPS",     f"{data.get('fps', 0):.1f}")

                elif msg[0] == "RECONNECTING":
                    status_placeholder.markdown(
                        f"""<div class="sg-hud-status">
                            <span class="sg-stream-reconnecting">◌ RECONNECTING…</span>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    add_log(msg[1], "WARN")

                elif msg[0] in ("ERR", "LOG"):
                    add_log(msg[1], "WARN" if msg[0] == "ERR" else "INFO")
                    if msg[0] == "ERR":
                        st.session_state.live_active = False
                        break
            except queue.Empty:
                break
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░  ANALYTICS  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "analytics":
    comp = st.session_state.compliance
    s_color = "#4ADE80" if comp>=75 else ("#FBBF24" if comp>=60 else "#F87171")
    status = "Excellent" if comp>=90 else "Good" if comp>=75 else "Needs attention" if comp>=60 else "Critical"

    st.markdown(f"""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span>Analytics</div>
  <h1 class="sg-hero-title">Session <span class="text-gradient">Analytics</span></h1>
  <p class="sg-hero-sub">Compliance gauge, alert distribution and hourly breach volume.</p>
</div>
<div class="sg-wrap">
  <div class="sg-stats-row">
    <div class="sg-stat-card"><div class="sg-stat-icon">🛡️</div>
      <div class="sg-stat-value" style="color:{s_color};">{comp:.1f}%</div>
      <div class="sg-stat-label">Compliance Score</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">🎞️</div>
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
        st.plotly_chart(compliance_gauge(comp), use_container_width=True, config={"displayModeBar":False})
    with a2:
        if st.session_state.alert_log:
            df_a = pd.DataFrame(st.session_state.alert_log)
            if "Type" in df_a.columns:
                vc = df_a["Type"].value_counts().reset_index(); vc.columns=["Type","Count"]
                colors = [{"PPE_VIOLATION":"#F87171","TOOL_UNATTENDED":"#FBBF24"}.get(t,"#3B82F6") for t in vc["Type"]]
                fig = go.Figure(go.Bar(x=vc["Type"],y=vc["Count"],marker_color=colors))
                st.plotly_chart(_chart_layout(fig,200,"Alert Type Breakdown"), use_container_width=True, config={"displayModeBar":False})
        else:
            st.markdown('<div class="sg-box info" style="height:180px;display:flex;align-items:center;justify-content:center;">No data yet.</div>', unsafe_allow_html=True)

    hourly = get_alert_counts_by_hour()
    if hourly:
        hdf = pd.DataFrame(hourly, columns=["Hour","Count"])
        fig2 = px.bar(hdf,x="Hour",y="Count",color_discrete_sequence=["#3B82F6"])
        st.plotly_chart(_chart_layout(fig2,220,"Hourly Alert Volume"), use_container_width=True, config={"displayModeBar":False})


# ════════════════════════════════════════════════════════════════════════════
# ░░░  HISTORY  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "history":
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
                           adf.to_csv(index=False),"full_alert_history.csv","text/csv")
    else:
        st.markdown('<div class="sg-box info">No alerts recorded yet.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░  LOGS  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "logs":
    st.markdown("""
<div class="sg-hero">
  <h1 class="sg-hero-title">System <span class="text-gradient">Event Log</span></h1>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    lf1, lf2 = st.columns([3,1])
    with lf1: lvl = st.selectbox("Filter",["ALL","INFO","SUCCESS","WARN","ALERT"])
    with lf2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕ Clear",use_container_width=True): st.session_state.system_log=[]; st.rerun()

    logs = st.session_state.system_log
    if lvl!="ALL": logs=[l for l in logs if l["level"]==lvl]
    cls_map={"INFO":"li","SUCCESS":"ls","WARN":"lw","ALERT":"la"}
    if logs:
        log_html=""
        for e in reversed(logs[-120:]):
            c=cls_map.get(e["level"],"")
            log_html+=(f'<span class="lt">[{e["time"]}]</span> '
                       f'<span class="{c}">[{e["level"].ljust(7)}]</span> '
                       f'{e["msg"]}<br>')
        st.markdown(f'<div class="sg-log">{log_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="sg-box info" style="text-align:center;padding:1.5rem;">No log entries.</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░  SYSTEM  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "system":
    import platform
    try:
        import torch as _torch
        cuda_ok  = _torch.cuda.is_available()
        gpu_name = _torch.cuda.get_device_name(0) if cuda_ok else "N/A"
        gpu_mem  = f"{_torch.cuda.get_device_properties(0).total_memory/1073741824:.1f} GB" if cuda_ok else "N/A"
        torch_ver = _torch.__version__
    except Exception:
        cuda_ok=False; gpu_name=gpu_mem=torch_ver="N/A"
    try:
        import ultralytics as _ul; ul_ver=_ul.__version__
    except Exception: ul_ver="N/A"

    st.markdown(f"""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span>{'CUDA Active' if cuda_ok else 'CPU Mode'}</div>
  <h1 class="sg-hero-title">System <span class="text-gradient">Information</span></h1>
  <p class="sg-hero-sub">Hardware, software versions, model weight status and launch guide.</p>
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
      <div class="sg-stat-label">CUDA GPU</div></div>
    <div class="sg-stat-card"><div class="sg-stat-icon">🤖</div>
      <div class="sg-stat-value">{models_active}/3</div>
      <div class="sg-stat-label">Models Loaded</div></div>
  </div>
  <div class="sg-box info">
    <strong style="color:#60A5FA;">GPU</strong>: {gpu_name} &nbsp;·&nbsp; {gpu_mem} &nbsp;|&nbsp;
    <strong style="color:#60A5FA;">Platform</strong>: {platform.system()} {platform.release()} &nbsp;·&nbsp;
    <strong style="color:#60A5FA;">Ultralytics</strong>: {ul_ver}
  </div>
""", unsafe_allow_html=True)

    st.markdown('<h2 class="sg-section-title" style="margin-top:1.5rem;">Weight File Status</h2>', unsafe_allow_html=True)
    rows_html=""
    for mn,wp,perf,ds in [
        ("HUMAN",HUMAN_WEIGHTS,"99.44% mAP@50","15,357 images"),
        ("PPE",  PPE_WEIGHTS,  "79.90% mAP@50","11-class merged"),
        ("TOOLS",TOOL_WEIGHTS, "67.87% mAP@50","6,535 images"),
    ]:
        ok=Path(wp).exists()
        rows_html+=f"""<tr>
<td><strong style="color:{'#4ADE80' if ok else '#F87171'};">{mn}</strong></td>
<td style="color:var(--muted);font-size:.8rem;word-break:break-all;">{wp}</td>
<td>{'✅ OK' if ok else '❌ Missing'}</td>
<td style="color:var(--txt2);">{perf}</td>
<td style="color:var(--muted);">{ds}</td>
</tr>"""
    st.markdown(f"""
<table class="sg-table"><thead><tr>
  <th>Model</th><th>Path</th><th>Status</th><th>Performance</th><th>Dataset</th>
</tr></thead><tbody>{rows_html}</tbody></table>

<h2 class="sg-section-title" style="margin-top:2rem;">Launch Guide</h2>
<div class="sg-box good">
  <div style="line-height:2.3;font-size:.9rem;">
    <strong style="color:#4ADE80;">Launch</strong> &nbsp;»&nbsp; Double-click <code>Launch SafeGuard AI.bat</code><br>
    <strong style="color:#4ADE80;">URL</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;»&nbsp; <code>http://localhost:8501</code><br>
    <strong style="color:#4ADE80;">GPU</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;»&nbsp; RTX 4060 · CUDA FP16 · auto-detected<br>
    <strong style="color:#4ADE80;">Database</strong> »&nbsp; outputs/safeguard.db (SQLite, auto-created)
  </div>
</div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ░░░  TRAINING MONITOR  ░░░
# ════════════════════════════════════════════════════════════════════════════
elif page == "train":
    st.markdown("""
<div class="sg-hero">
  <div class="sg-welcome-badge"><span class="pulse-dot"></span>Training</div>
  <h1 class="sg-hero-title">Training <span class="text-gradient">Monitor</span></h1>
  <p class="sg-hero-sub">Live loss curves and mAP@50 progress for Human, PPE and Tools YOLOv11n models.</p>
</div>
<div class="sg-wrap">
""", unsafe_allow_html=True)

    TOOLS_CSV = Path(r"E:\4TH YEAR PROJECT\NEW TOOLS\runs\detect\train_fast\results.csv")
    PPE_CSV   = Path(r"E:\4TH YEAR PROJECT\NEW PPE\runs\detect\train_v2_nano\results.csv")
    HUMAN_CSV = Path(r"E:\4TH YEAR PROJECT\HUMAN\runs\detect\train_fast\results.csv")

    def load_results(p):
        if not p.exists(): return None
        try:
            df=pd.read_csv(p); df.columns=[c.strip() for c in df.columns]; return df
        except Exception: return None

    for csv_path,model_name,target,baseline,color in [
        (HUMAN_CSV,"Human Model",75,0.9944,"#4ADE80"),
        (TOOLS_CSV,"Tools Model",60,0.6787,"#FBBF24"),
        (PPE_CSV,  "PPE Model",  200,0.7990,"#A78BFA"),
    ]:
        df=load_results(csv_path)
        st.markdown(f'<h2 class="sg-section-title">{model_name}</h2>', unsafe_allow_html=True)
        if df is None or df.empty:
            st.markdown('<div class="sg-box warn">No training data found.</div>', unsafe_allow_html=True)
            continue
        total_ep=len(df)
        best_map=df.get("metrics/mAP50(B)",pd.Series([0])).max()
        last=df.iloc[-1]
        mm1,mm2,mm3,mm4=st.columns(4)
        mm1.metric("Epochs",f"{total_ep}/{target}")
        mm2.metric("Best mAP@50",f"{best_map*100:.2f}%")
        mm3.metric("Box Loss",f"{float(last.get('train/box_loss',0)):.4f}")
        mm4.metric("Cls Loss",f"{float(last.get('train/cls_loss',0)):.4f}")
        st.progress(min(total_ep/target,1.0))
        lc1,lc2=st.columns(2)
        with lc1:
            if "train/box_loss" in df.columns:
                fig=go.Figure()
                fig.add_trace(go.Scatter(y=df["train/box_loss"],name="Box Loss",line=dict(color=color,width=2)))
                fig.add_trace(go.Scatter(y=df["train/cls_loss"],name="Cls Loss",line=dict(color="#F87171",width=1.5,dash="dash")))
                st.plotly_chart(_chart_layout(fig,200,"Losses"),use_container_width=True,config={"displayModeBar":False})
        with lc2:
            if "metrics/mAP50(B)" in df.columns:
                fig2=go.Figure()
                fig2.add_trace(go.Scatter(y=df["metrics/mAP50(B)"],name="mAP@50",
                    line=dict(color=color,width=2),fill="tozeroy",fillcolor="rgba(59,130,246,.07)"))
                fig2.add_hline(y=baseline,line_dash="dash",line_color="#F87171",
                    annotation_text=f"Target {baseline*100:.1f}%",annotation_font_color="#F87171")
                st.plotly_chart(_chart_layout(fig2,200,"mAP@50"),use_container_width=True,config={"displayModeBar":False})

    rc1,rc2=st.columns([4,1])
    with rc1: st.caption(f"Last refreshed: {datetime.datetime.now().strftime('%H:%M:%S')}")
    with rc2:
        if st.button("↺ Refresh",use_container_width=True): st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="sg-footer">
  <span>SafeGuard AI</span> · Industrial Safety Detection System ·
  YOLOv11n · PPE Compliance · Tool Abandonment FSM · CUDA FP16
</div>
""", unsafe_allow_html=True)
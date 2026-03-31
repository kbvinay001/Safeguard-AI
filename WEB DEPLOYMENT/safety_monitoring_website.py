"""
SafeGuard AI  –  2060-class Command Interface
CSS injected via JS (bypasses Streamlit sanitizer).
Navigation via st.radio (WebSocket – instant, no reload).
"""
import os
os.environ["STREAMLIT_DEPRECATION_WARNINGS"] = "false"

import streamlit as st
import streamlit.components.v1 as components
import cv2, numpy as np, tempfile, time, threading, queue as _queue
import datetime, uuid, base64, json
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="SafeGuard AI", page_icon="⬡",
                   layout="wide", initial_sidebar_state="collapsed")

HERE = Path(__file__).parent
import sys; sys.path.insert(0, str(HERE))

try:
    from detection_engine import SafetyDetectionEngine
    from safety_config import *
    from db_manager import log_alert, upsert_session, get_all_alerts, get_all_sessions, get_alert_counts_by_hour
    from pdf_report import generate_pdf
    IMPORTS_OK = True
except Exception as _e:
    IMPORTS_OK = False; _IMPORT_ERR = str(_e)

def _b64(p):
    try: return base64.b64encode(open(p,"rb").read()).decode()
    except: return ""

ARCH_B64 = _b64(str(HERE / "architecture.jpeg"))
BG_B64   = _b64(str(HERE / "bg_jarvis.png"))

# ── session state ─────────────────────────────────────────────────────────────
for k, v in dict(
    engine=None, alert_log=[], total_frames=0, total_alerts=0,
    processed_video_path=None, processing_stats=None,
    system_log=[], session_id=str(uuid.uuid4())[:8].upper(),
    heatmap_acc=None, compliance=100.0,
    live_active=False, live_queue=None, live_stop=None,
).items():
    if k not in st.session_state: st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# CSS  –  injected via JS so Streamlit sanitizer never sees the raw CSS text
# ══════════════════════════════════════════════════════════════════════════════
_BG = f"url('data:image/png;base64,{BG_B64}') center/cover no-repeat fixed" if BG_B64 else "radial-gradient(ellipse at 20% 50%,#030a1a 0%,#000507 60%,#000 100%)"

# Plain Python string – NO f-string {{ }} escaping needed at all
_CSS = """
:root {
  --c1:#00fff7; --c2:#7c3aed; --c3:#ff00aa;
  --c4:#ffd700; --c5:#00ff85; --cw:#e8f9ff;
  --glow1:rgba(0,255,247,0.5); --glow2:rgba(124,58,237,0.4);
  --glass:rgba(0,255,247,0.032); --border:rgba(0,255,247,0.18);
}
html,body,[class*="css"] {
  font-family:'Rajdhani',sans-serif !important;
  background-color:#000507; color:#e8f9ff;
}
.main::before {
  content:''; position:fixed; top:0; left:0; right:0; bottom:0;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px);
  pointer-events:none; z-index:9999;
}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stSidebar"],.stDeployButton { display:none !important; }
.main .block-container { padding:0 2rem 2rem; max-width:1600px; margin:0 auto; }
.stSpinner > div { border-color:var(--c1) !important; }
h1,h2,h3,h4 { font-family:'Orbitron',sans-serif !important; }

/* NAV */
.nav-shell {
  display:flex; align-items:center; justify-content:space-between;
  padding:.7rem 2rem; margin:-1rem -2rem 0;
  background:rgba(0,3,10,.94); border-bottom:1px solid var(--border);
  backdrop-filter:blur(24px); position:sticky; top:0; z-index:100;
}
.nav-brand {
  font-family:'Orbitron',sans-serif; font-size:1.05rem; font-weight:900;
  color:var(--c1); letter-spacing:.22em; text-shadow:0 0 20px var(--glow1);
  display:flex; align-items:center; gap:.6rem;
}
.nav-brand span { font-size:.52rem; color:rgba(0,255,247,.45); letter-spacing:.12em; display:block; margin-top:.1rem; }
.nav-dot {
  width:8px; height:8px; background:var(--c1); border-radius:50%;
  box-shadow:0 0 12px var(--glow1); animation:npulse 2s infinite;
}
@keyframes npulse { 0%,100%{ opacity:1; } 50%{ opacity:.28; } }

/* RADIO as nav tabs */
[data-testid="stRadio"] > label { display:none !important; }
[data-testid="stRadio"] div[role="radiogroup"] { display:flex; gap:.45rem; flex-wrap:nowrap; }
[data-testid="stRadio"] div[role="radiogroup"] label {
  display:inline-flex !important; align-items:center;
  padding:.42rem 1.3rem; border:1px solid var(--border);
  background:rgba(0,255,247,.025); border-radius:3px; cursor:pointer;
  font-family:'Orbitron',sans-serif !important; font-size:.6rem;
  font-weight:700; letter-spacing:.16em; color:rgba(0,255,247,.45);
  transition:all .22s; position:relative; overflow:hidden;
}
[data-testid="stRadio"] div[role="radiogroup"] label:hover {
  color:var(--c1); border-color:rgba(0,255,247,.5);
  background:rgba(0,255,247,.07); box-shadow:0 0 18px rgba(0,255,247,.18);
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  color:var(--c1); border-color:var(--c1);
  background:rgba(0,255,247,.1);
  box-shadow:0 0 22px rgba(0,255,247,.28), inset 0 0 16px rgba(0,255,247,.04);
  text-shadow:0 0 10px var(--glow1);
}
[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked)::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,var(--c1),transparent);
  animation:scanb 1.6s linear infinite;
}
@keyframes scanb { 0%{ transform:scaleX(.1); opacity:.3; } 50%{ transform:scaleX(1); opacity:1; } 100%{ transform:scaleX(.1); opacity:.3; } }
[data-testid="stRadio"] div[role="radiogroup"] input { display:none !important; }

/* sub-nav purple */
.sub-nav-wrap [data-testid="stRadio"] div[role="radiogroup"] label {
  padding:.32rem .9rem; font-size:.54rem; letter-spacing:.1em;
  border-color:rgba(124,58,237,.28); color:rgba(124,58,237,.55);
  background:rgba(124,58,237,.025);
}
.sub-nav-wrap [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
  border-color:var(--c2); color:#b794f4;
  background:rgba(124,58,237,.1); box-shadow:0 0 18px rgba(124,58,237,.28);
  text-shadow:0 0 10px var(--glow2);
}

/* GLASS PANELS */
.holo-panel {
  background:var(--glass); border:1px solid var(--border);
  border-radius:4px; padding:1.4rem 1.6rem; position:relative;
  backdrop-filter:blur(12px);
}
.holo-panel::before,.holo-panel::after {
  content:''; position:absolute; width:14px; height:14px;
  border-color:var(--c1); border-style:solid;
}
.holo-panel::before { top:-1px; left:-1px; border-width:2px 0 0 2px; }
.holo-panel::after  { bottom:-1px; right:-1px; border-width:0 2px 2px 0; }

/* METRICS */
.metric-grid { display:flex; gap:1rem; flex-wrap:wrap; margin:1rem 0; }
.mcard {
  flex:1; min-width:140px; background:var(--glass);
  border:1px solid var(--border); border-radius:4px; padding:1rem 1.2rem;
  position:relative; overflow:hidden;
}
.mcard::before {
  content:''; position:absolute; top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,transparent,var(--c1),transparent);
  animation:scant 3s linear infinite;
}
@keyframes scant { 0%{ transform:translateX(-100%); } 100%{ transform:translateX(100%); } }
.mcard-lbl { font-family:'Share Tech Mono',monospace; font-size:.6rem; letter-spacing:.14em; color:rgba(0,255,247,.48); text-transform:uppercase; }
.mcard-val { font-family:'Orbitron',sans-serif; font-size:1.9rem; font-weight:900; line-height:1.1; margin:.3rem 0 .1rem; }
.mcard-sub { font-size:.6rem; color:rgba(0,255,247,.32); letter-spacing:.08em; }
.vc { color:var(--c1); text-shadow:0 0 14px var(--glow1); }
.vg { color:var(--c5); text-shadow:0 0 14px rgba(0,255,133,.5); }
.va { color:#ff4444; text-shadow:0 0 14px rgba(255,68,68,.5); }
.vw { color:var(--c4); text-shadow:0 0 14px rgba(255,215,0,.5); }
.vp { color:var(--c3); text-shadow:0 0 14px rgba(255,0,170,.4); }

/* HEADERS */
.sec-hdr {
  font-family:'Orbitron',sans-serif; font-size:.68rem; font-weight:700;
  letter-spacing:.28em; color:rgba(0,255,247,.58); text-transform:uppercase;
  display:flex; align-items:center; gap:1rem; margin:1.8rem 0 1rem;
}
.sec-hdr::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(0,255,247,.28),transparent); }

/* HERO */
.hero {
  text-align:center; padding:3.5rem 1rem 2.5rem;
  background:radial-gradient(ellipse at 50% 0%, rgba(0,255,247,.055) 0%, transparent 70%);
  border-bottom:1px solid var(--border); margin:-1rem -2rem 2rem;
}
.hero-eyebrow { font-family:'Share Tech Mono',monospace; font-size:.63rem; letter-spacing:.38em; color:rgba(0,255,247,.48); margin-bottom:1rem; }
.hero h1 {
  font-family:'Orbitron',sans-serif !important; font-size:3.2rem; font-weight:900; margin:0;
  background:linear-gradient(135deg,#fff 0%,var(--c1) 50%,#fff 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.hero-sub { font-family:'Rajdhani',sans-serif; font-size:1rem; color:rgba(232,249,255,.42); max-width:600px; margin:.8rem auto 0; }

/* BUTTONS */
.stButton > button {
  font-family:'Orbitron',sans-serif !important; font-size:.6rem !important;
  font-weight:700; letter-spacing:.18em; text-transform:uppercase;
  border:1px solid var(--c1) !important; border-radius:3px !important;
  background:rgba(0,255,247,.06) !important; color:var(--c1) !important; transition:all .28s;
}
.stButton > button:hover {
  background:rgba(0,255,247,.14) !important;
  box-shadow:0 0 28px rgba(0,255,247,.32), inset 0 0 18px rgba(0,255,247,.04) !important;
  transform:translateY(-1px);
}
.stButton > button[kind="primary"] { background:rgba(0,255,247,.11) !important; box-shadow:0 0 18px rgba(0,255,247,.2) !important; }

/* MISC */
[data-testid="stDataFrame"] { background:var(--glass); border:1px solid var(--border); }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,var(--c2),var(--c1)) !important; box-shadow:0 0 11px var(--glow1) !important; }
[data-testid="stFileUploaderDropzone"] { border:1px dashed rgba(0,255,247,.28) !important; background:rgba(0,255,247,.018) !important; border-radius:4px !important; }
.logbox { background:#000c18; border:1px solid var(--border); border-radius:4px; padding:1rem; font-family:'Share Tech Mono',monospace; font-size:.7rem; line-height:1.8; max-height:400px; overflow-y:auto; color:rgba(0,255,247,.68); }
[data-testid="stTextInput"] input { background:rgba(0,255,247,.038) !important; border:1px solid var(--border) !important; border-radius:3px !important; color:#e8f9ff !important; font-family:'Share Tech Mono',monospace !important; }
[data-testid="stTextInput"] input:focus { border-color:var(--c1) !important; box-shadow:0 0 14px rgba(0,255,247,.18) !important; }
[data-testid="stSelectbox"] > div > div { background:rgba(0,255,247,.038) !important; border:1px solid var(--border) !important; color:#e8f9ff !important; }
.fcard { background:var(--glass); border:1px solid var(--border); border-radius:4px; padding:1.3rem; transition:all .3s; height:100%; }
.fcard:hover { border-color:rgba(0,255,247,.42); transform:translateY(-3px); box-shadow:0 8px 38px rgba(0,255,247,.1); }
.fcard-icon { font-size:1.6rem; margin-bottom:.7rem; }
.fcard-title { font-family:'Orbitron',sans-serif; font-size:.7rem; font-weight:700; letter-spacing:.12em; color:var(--c1); margin-bottom:.5rem; text-transform:uppercase; }
.fcard-body { font-size:.85rem; color:rgba(232,249,255,.52); line-height:1.6; }
.status-strip { background:rgba(0,255,133,.055); border:1px solid rgba(0,255,133,.28); border-radius:3px; padding:.6rem 1rem; font-family:'Share Tech Mono',monospace; font-size:.72rem; color:rgba(0,255,133,.88); margin:1rem 0; }
.warn-strip { background:rgba(255,215,0,.055); border:1px solid rgba(255,215,0,.28); border-radius:3px; padding:.6rem 1rem; font-family:'Share Tech Mono',monospace; font-size:.72rem; color:rgba(255,215,0,.88); margin:1rem 0; }
"""

# Inject fonts (safe – no curly braces in link tags)
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Inject all CSS + background via JS (completely bypasses Streamlit sanitizer)
_bg_css = f"html,body{{background:{_BG};background-color:#000507}}"
components.html(
    "<script>"
    "(function(){"
    f"  var s=document.createElement('style'); s.textContent={json.dumps(_CSS)}; window.parent.document.head.appendChild(s);"
    f"  var b=document.createElement('style'); b.textContent={json.dumps(_bg_css)}; window.parent.document.head.appendChild(b);"
    "})();"
    "</script>",
    height=0, scrolling=False
)

# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION  —  st.radio = WebSocket (instant, no page reload)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="nav-shell">', unsafe_allow_html=True)
nc1, nc2 = st.columns([1, 2])
with nc1:
    st.markdown('<div class="nav-brand"><div class="nav-dot"></div>⬡ SAFEGUARD<span>AI SAFETY MONITORING SYSTEM v2.0</span></div>', unsafe_allow_html=True)
with nc2:
    section_sel = st.radio("main_nav", ["◈  DASHBOARD", "⬡  DEPLOYMENT"],
                           horizontal=True, key="main_nav_r",
                           label_visibility="collapsed",
                           index=1 if st.session_state.get("_sec_dep", False) else 0)
st.markdown('</div>', unsafe_allow_html=True)

is_deployment = "DEPLOYMENT" in section_sel
st.session_state["_sec_dep"] = is_deployment

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if not is_deployment:
    st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">// INDUSTRIAL SAFETY INTELLIGENCE · YOLOv11 · RTX 4060 · CUDA FP16</div>
  <h1>SafeGuard AI</h1>
  <div class="hero-sub">Autonomous real-time PPE compliance and tool abandonment detection for industrial CCTV infrastructure</div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">— CORE CAPABILITIES</div>', unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    for col,(ic,ti,bd) in zip([f1,f2,f3,f4],[
        ("👁️","HUMAN DETECTION","YOLOv11n fine-tuned on 15,357 industrial images. Tracks every worker with persistent ID."),
        ("🦺","PPE COMPLIANCE","Helmet · Vest · Gloves · Goggles per worker every frame. Alert within 1 second of violation."),
        ("🔧","TOOL TRACKING","Drill · Hammer · Pliers · Wrench. Temporal FSM: 25s warning → 35s critical alert."),
        ("📡","RTSP STREAMING","Connect IP cameras via RTSP/RTMP. Live detection with background threading."),
    ]):
        with col:
            st.markdown(f'<div class="fcard"><div class="fcard-icon">{ic}</div><div class="fcard-title">{ti}</div><div class="fcard-body">{bd}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">— SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)
    if ARCH_B64:
        st.markdown(f'<div class="holo-panel" style="padding:.8rem"><img src="data:image/jpeg;base64,{ARCH_B64}" style="width:100%;border-radius:3px"/></div>', unsafe_allow_html=True)
    else:
        st.info("architecture.jpeg not found.")

    st.markdown('<div class="sec-hdr">— DETECTION PIPELINE</div>', unsafe_allow_html=True)
    for col,(n,t,b) in zip(st.columns(6),[
        ("01","INGEST","RTSP / Video → OpenCV"),
        ("02","DETECT","3× YOLOv11n FP16"),
        ("03","TRACK","IoU multi-object tracking"),
        ("04","ANALYSE","FSM · PPE match · Hazard"),
        ("05","ALERT","Tiered → SQLite log"),
        ("06","REPORT","PDF / CSV export"),
    ]):
        with col:
            st.markdown(f'<div class="holo-panel" style="padding:1rem;text-align:center"><div style="font-family:Orbitron;font-size:1.4rem;color:var(--c1);opacity:.35;font-weight:900">{n}</div><div style="font-family:Orbitron;font-size:.6rem;letter-spacing:.14em;color:var(--c1);margin:.4rem 0">{t}</div><div style="font-size:.7rem;color:rgba(232,249,255,.38);line-height:1.5">{b}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">— MODEL STATUS</div>', unsafe_allow_html=True)
    for col,(name,arch,data,cls,status,vc) in zip(st.columns(3),[
        ("🧍 HUMAN","YOLOv11n","15,357 imgs","1 class","Training 75 ep","vw"),
        ("🦺 PPE","YOLOv11n","11 classes","200 ep","79.9% mAP50","vg"),
        ("🔧 TOOLS","YOLOv11n","6,535 imgs","5 classes","Training 60 ep","vw"),
    ]):
        with col:
            st.markdown(f'<div class="holo-panel"><div style="font-family:Orbitron;font-size:.72rem;font-weight:700;color:var(--c1);margin-bottom:.8rem">{name}</div><div class="mcard-lbl">Architecture</div><div style="margin-bottom:.4rem">{arch}</div><div class="mcard-lbl">Dataset</div><div style="margin-bottom:.4rem">{data}</div><div class="mcard-lbl">Classes</div><div style="margin-bottom:.6rem">{cls}</div><div class="mcard-lbl">Status</div><div class="{vc}" style="font-family:Orbitron;font-size:.68rem;font-weight:700">{status}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT
# ══════════════════════════════════════════════════════════════════════════════
else:
    if not IMPORTS_OK:
        st.error(f"Import error: {_IMPORT_ERR}"); st.stop()

    @st.cache_resource(show_spinner=False)
    def load_engine():
        try:
            e = SafetyDetectionEngine(HUMAN_WEIGHTS, PPE_WEIGHTS, TOOL_WEIGHTS, device=DEVICE)
            return e, None
        except Exception as ex: return None, str(ex)

    def add_log(msg, level="INFO"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.system_log.append({"t":ts,"l":level,"m":msg})
        if len(st.session_state.system_log) > 200:
            st.session_state.system_log = st.session_state.system_log[-200:]

    if st.session_state.engine is None:
        with st.spinner("⬡ Loading YOLOv11 models…"):
            _eng, _err = load_engine()
            if _eng:
                st.session_state.engine = _eng
                add_log(f"Models: {', '.join(_eng.loaded_models)}", "SUCCESS")
            else:
                st.warning(f"Engine error: {_err}")

    engine = st.session_state.engine

    # Sub-nav
    st.markdown('<div class="sub-nav-wrap" style="padding:.5rem 0 .3rem">', unsafe_allow_html=True)
    page_sel = st.radio("sub_nav", ["◈ HOME","▶ ANALYSE","◉ ANALYTICS","⟳ HISTORY","≡ LOGS","⚡ TRAIN"],
                        horizontal=True, key="sub_nav_r", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── helpers ──────────────────────────────────────────────────────────────
    def calc_compliance(frames, alerts):
        if not frames: return 100.0
        return max(0.0, min(100.0, 100.0 - alerts/max(frames,1)*100*50))

    def compliance_gauge(score):
        color = "#00ff85" if score>=75 else ("#ffd700" if score>=50 else "#ff4444")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score,
            number={"suffix":"%","font":{"size":32,"color":"#e8f9ff","family":"Orbitron"}},
            gauge=dict(axis=dict(range=[0,100],tickfont={"color":"#1e3a52","size":9}),
                bar=dict(color=color,thickness=.22), bgcolor="#000c18",
                bordercolor="#0d1f35", borderwidth=1,
                steps=[dict(range=[0,50],color="rgba(255,68,68,.08)"),
                       dict(range=[50,75],color="rgba(255,215,0,.08)"),
                       dict(range=[75,100],color="rgba(0,255,133,.08)")]),
            title={"text":"Safety Compliance","font":{"size":11,"color":"#4a7a96"}},
            domain={"x":[0,1],"y":[0,1]}))
        fig.update_layout(height=240, margin=dict(t=40,b=5,l=15,r=15),
            paper_bgcolor="#000c18", plot_bgcolor="#000c18")
        return fig

    def process_video_file(inp, out, cb=None):
        cap = cv2.VideoCapture(inp)
        if not cap.isOpened(): return None, "Cannot open video"
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w,h   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer= cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        alerts, fps_r, fc = [], [], 0
        hacc = np.zeros((h,w), dtype=np.float32); t0 = time.time()
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                ft = time.time()
                try: ann, data = engine.process_frame(frame, video_dt=1.0/fps)
                except: ann = frame; data = {"alerts":[]}
                writer.write(ann); fc += 1
                dt = time.time()-ft
                if dt>0: fps_r.append(1.0/dt)
                for a in data.get("alerts",[]):
                    a["frame"]=fc; alerts.append(a)
                    cy2,cx2=h//2,w//2
                    hacc[max(0,cy2-60):min(h,cy2+60),max(0,cx2-60):min(w,cx2+60)]+=1.0
                if cb and fc%10==0: cb(fc/max(total,1), fc, total, fps_r[-1] if fps_r else 0)
        finally: cap.release(); writer.release()
        comp = calc_compliance(fc, len(alerts)); st.session_state.heatmap_acc = hacc
        return {"total_frames":fc,"total_alerts":len(alerts),
                "avg_fps":float(np.mean(fps_r)) if fps_r else 0.0,
                "total_time":time.time()-t0,"alerts":alerts,"compliance":comp}, None

    # ── HOME ─────────────────────────────────────────────────────────────────
    if "HOME" in page_sel:
        comp = st.session_state.compliance
        st.markdown('<div class="sec-hdr">— COMMAND CENTER</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="metric-grid">
<div class="mcard"><div class="mcard-lbl">Safety Compliance</div>
<div class="mcard-val {'vg' if comp>=75 else 'va'}">{comp:.1f}%</div><div class="mcard-sub">Current session</div></div>
<div class="mcard"><div class="mcard-lbl">Frames Analysed</div>
<div class="mcard-val vc">{st.session_state.total_frames:,}</div><div class="mcard-sub">This session</div></div>
<div class="mcard"><div class="mcard-lbl">Alerts Raised</div>
<div class="mcard-val {'va' if st.session_state.total_alerts>0 else 'vg'}">{st.session_state.total_alerts}</div><div class="mcard-sub">Safety violations</div></div>
<div class="mcard"><div class="mcard-lbl">Session ID</div>
<div class="mcard-val vc" style="font-size:1.2rem">{st.session_state.session_id}</div><div class="mcard-sub">Active token</div></div>
<div class="mcard"><div class="mcard-lbl">Models Active</div>
<div class="mcard-val vg">{len(engine.loaded_models) if engine else 0}/3</div><div class="mcard-sub">YOLOv11n loaded</div></div>
</div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr">— QUICK ACTIONS</div>', unsafe_allow_html=True)
        h1,h2,h3 = st.columns(3)
        with h1: st.markdown('<div class="holo-panel"><div class="fcard-title">▶ Video Analysis</div><div class="fcard-body">Upload CCTV recordings for batch PPE + tool detection with heatmap and PDF export.</div></div>', unsafe_allow_html=True)
        with h2: st.markdown('<div class="holo-panel"><div class="fcard-title">📡 Live RTSP</div><div class="fcard-body">Connect IP cameras via RTSP URL for real-time detection with background threading.</div></div>', unsafe_allow_html=True)
        with h3: st.markdown('<div class="holo-panel"><div class="fcard-title">◉ Analytics</div><div class="fcard-body">Compliance gauge, hourly alert charts, session history and audit export.</div></div>', unsafe_allow_html=True)

    # ── ANALYSE ──────────────────────────────────────────────────────────────
    elif "ANALYSE" in page_sel:
        st.markdown('<div class="sec-hdr">— ANALYSE</div>', unsafe_allow_html=True)
        mode = st.radio("inp_mode", ["📁  Video File","📡  RTSP / IP Camera"],
                        horizontal=True, label_visibility="collapsed", key="analyse_mode")

        if "Video" in mode:
            uf = st.file_uploader("Upload CCTV video", type=["mp4","avi","mov","mkv"],
                                  label_visibility="collapsed")
            if uf:
                fsz = len(uf.getbuffer())/1048576
                st.markdown(f'<div class="status-strip">📄 {uf.name} · {fsz:.1f} MB ready</div>', unsafe_allow_html=True)
                if st.button("▶  ANALYSE NOW", use_container_width=True, type="primary"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as ti:
                        ti.write(uf.getbuffer()); inp=ti.name
                    out = inp.replace(".mp4","_out.mp4")
                    pb = st.progress(0.0); pt = st.empty()
                    def upd(p,f,tot,cfps):
                        pb.progress(float(p))
                        pt.markdown(f'<div class="warn-strip">⚡ Frame {f}/{tot} · {p*100:.1f}% · {cfps:.1f} FPS</div>', unsafe_allow_html=True)
                    stats, err = process_video_file(inp, out, upd)
                    try: os.unlink(inp)
                    except: pass
                    if err: st.error(err)
                    else:
                        st.session_state.processed_video_path = out
                        st.session_state.processing_stats = stats
                        st.session_state.total_frames += stats["total_frames"]
                        st.session_state.total_alerts += stats["total_alerts"]
                        st.session_state.compliance = stats["compliance"]
                        for a in stats["alerts"]:
                            st.session_state.alert_log.append({"Frame":a.get("frame",0),"Type":a.get("type","?"),"Tool":a.get("tool_name","N/A"),"Timer(s)":round(float(a.get("timer",0)),1)})
                            log_alert(st.session_state.session_id,"video",a.get("frame",0),a.get("type","?"),a.get("tool_name",""),float(a.get("timer",0)),a.get("missing_ppe",[]))
                        upsert_session(st.session_state.session_id,"video",stats["total_frames"],stats["total_alerts"],stats["compliance"])
                        add_log(f"Video done: {stats['total_frames']} frames, {stats['total_alerts']} alerts","SUCCESS")
                        pb.progress(1.0); pt.empty(); st.rerun()

            vp = st.session_state.processed_video_path
            if vp and Path(vp).exists():
                stats = st.session_state.processing_stats
                st.markdown('<div class="sec-hdr">— RESULTS</div>', unsafe_allow_html=True)
                if stats:
                    st.markdown(f"""<div class="metric-grid">
<div class="mcard"><div class="mcard-lbl">Frames</div><div class="mcard-val vc">{stats['total_frames']}</div></div>
<div class="mcard"><div class="mcard-lbl">Avg FPS</div><div class="mcard-val vc">{stats['avg_fps']:.1f}</div></div>
<div class="mcard"><div class="mcard-lbl">Alerts</div><div class="mcard-val {'va' if stats['total_alerts']>0 else 'vg'}">{stats['total_alerts']}</div></div>
<div class="mcard"><div class="mcard-lbl">Compliance</div><div class="mcard-val {'vg' if stats['compliance']>=75 else 'va'}">{stats['compliance']:.1f}%</div></div>
<div class="mcard"><div class="mcard-lbl">Time</div><div class="mcard-val vp">{stats['total_time']:.1f}s</div></div>
</div>""", unsafe_allow_html=True)
                v1,v2 = st.columns(2)
                with v1:
                    st.markdown("**Detection Output**")
                    try:
                        with open(vp,"rb") as vf: st.video(vf.read())
                    except: st.warning("Preview unavailable")
                with v2:
                    st.markdown("**Violation Heatmap**")
                    hacc = st.session_state.heatmap_acc
                    if hacc is not None and hacc.max()>0:
                        norm = cv2.normalize(hacc,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
                        hm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                        st.image(cv2.cvtColor(hm,cv2.COLOR_BGR2RGB), caption="Red = frequent violation zones", use_container_width=True)
                    else: st.info("No violations — heatmap empty")
                d1,d2,d3 = st.columns(3)
                with d1:
                    try:
                        with open(vp,"rb") as f: st.download_button("📥 VIDEO",f.read(),"output.mp4","video/mp4",use_container_width=True)
                    except: pass
                with d2:
                    if st.session_state.alert_log:
                        df=pd.DataFrame(st.session_state.alert_log)
                        st.download_button("📥 CSV",df.to_csv(index=False),"alerts.csv","text/csv",use_container_width=True)
                with d3:
                    if stats:
                        alts=[{"ts":datetime.datetime.now().isoformat(),"type":a.get("type","?"),"tool":a.get("tool_name",""),"timer_s":float(a.get("timer",0))} for a in stats.get("alerts",[])]
                        src = uf.name if 'uf' in dir() and uf else "video"
                        try:
                            pb2 = generate_pdf({**stats,"source":src,"compliance":stats["compliance"]},alts,st.session_state.session_id)
                            st.download_button("📥 PDF",pb2,"safety_report.pdf","application/pdf",use_container_width=True)
                        except Exception as epdf: st.warning(f"PDF: {epdf}")
        else:
            st.markdown('<div class="sec-hdr">— RTSP LIVE STREAM</div>', unsafe_allow_html=True)
            rtsp = st.text_input("RTSP URL", placeholder="rtsp://192.168.1.100:554/stream", label_visibility="collapsed")
            st.caption("`rtsp://user:pass@192.168.1.100:554/stream1`")
            c1,c2 = st.columns(2)
            with c1: connect_btn = st.button("📡  CONNECT",use_container_width=True,type="primary",disabled=not rtsp)
            with c2: stop_btn = st.button("⬛  STOP",use_container_width=True,disabled=not st.session_state.live_active)
            if stop_btn and st.session_state.live_stop:
                st.session_state.live_stop.set(); st.session_state.live_active=False
                add_log("Stream stopped","INFO"); st.rerun()
            if connect_btn and rtsp:
                stop_ev=threading.Event(); q2=_queue.Queue(maxsize=3)
                def _reader(url,q,ev):
                    cap2=cv2.VideoCapture(url)
                    while not ev.is_set() and cap2.isOpened():
                        ret2,frm2=cap2.read()
                        if not ret2: break
                        try: ann2,data2=engine.process_frame(frm2)
                        except: ann2=frm2; data2={"alerts":[]}
                        for a2 in data2.get("alerts",[]):
                            st.session_state.alert_log.append({"Frame":st.session_state.total_frames,"Type":a2.get("type","?"),"Tool":a2.get("tool_name","N/A"),"Timer(s)":round(float(a2.get("timer",0)),1)})
                        try: q2.put_nowait(ann2)
                        except _queue.Full: pass
                        st.session_state.total_frames+=1
                    cap2.release()
                t2=threading.Thread(target=_reader,args=(rtsp,q2,stop_ev),daemon=True); t2.start()
                st.session_state.live_active=True; st.session_state.live_queue=q2; st.session_state.live_stop=stop_ev
                add_log(f"Stream: {rtsp}","SUCCESS")
            if st.session_state.live_active:
                st.markdown('<div class="status-strip">📡 Live stream active — detection running.</div>', unsafe_allow_html=True)
                ph=st.empty()
                for _ in range(30):
                    try:
                        frm_live=st.session_state.live_queue.get_nowait()
                        ph.image(cv2.cvtColor(frm_live,cv2.COLOR_BGR2RGB),use_container_width=True,caption="Live · AI Detection Active")
                    except _queue.Empty: pass
                    time.sleep(0.04)
                st.rerun()

    # ── ANALYTICS ────────────────────────────────────────────────────────────
    elif "ANALYTICS" in page_sel:
        comp = st.session_state.compliance
        st.markdown('<div class="sec-hdr">— SESSION ANALYTICS</div>', unsafe_allow_html=True)
        a1,a2 = st.columns([1,2])
        with a1: st.plotly_chart(compliance_gauge(comp),use_container_width=True,config={"displayModeBar":False})
        with a2:
            status = "EXCELLENT" if comp>=90 else ("GOOD" if comp>=75 else ("REVIEW NEEDED" if comp>=60 else "CRITICAL"))
            st.markdown(f"""<div class="metric-grid">
<div class="mcard"><div class="mcard-lbl">Compliance</div><div class="mcard-val {'vg' if comp>=75 else 'va'}">{comp:.1f}%</div><div class="mcard-sub">{status}</div></div>
<div class="mcard"><div class="mcard-lbl">Frames</div><div class="mcard-val vc">{st.session_state.total_frames:,}</div><div class="mcard-sub">Processed</div></div>
<div class="mcard"><div class="mcard-lbl">Alerts</div><div class="mcard-val {'va' if st.session_state.total_alerts>0 else 'vg'}">{st.session_state.total_alerts}</div><div class="mcard-sub">Violations</div></div>
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">— HOURLY ALERT DISTRIBUTION</div>', unsafe_allow_html=True)
        hourly = get_alert_counts_by_hour()
        if hourly:
            hdf = pd.DataFrame(hourly, columns=["Hour","Count"])
            fig2 = px.bar(hdf, x="Hour", y="Count", color_discrete_sequence=["#00fff7"])
            fig2.update_layout(height=220, margin=dict(t=10,b=20,l=20,r=10),
                paper_bgcolor="#000c18", plot_bgcolor="#000c18",
                font=dict(color="#4a7a96"),
                xaxis=dict(gridcolor="#0d1f35"), yaxis=dict(gridcolor="#0d1f35"))
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        else: st.info("No historical data — process a video first.")

    # ── HISTORY ──────────────────────────────────────────────────────────────
    elif "HISTORY" in page_sel:
        st.markdown('<div class="sec-hdr">— SESSION HISTORY</div>', unsafe_allow_html=True)
        sess = get_all_sessions()
        if sess: st.dataframe(pd.DataFrame(sess), use_container_width=True, height=200)
        else: st.info("No previous sessions.")
        st.markdown('<div class="sec-hdr">— FULL ALERT LOG</div>', unsafe_allow_html=True)
        all_al = get_all_alerts()
        if all_al:
            adf = pd.DataFrame(all_al); st.dataframe(adf, use_container_width=True, height=320)
            st.download_button("📥 EXPORT CSV", adf.to_csv(index=False), "history.csv", "text/csv")
        else: st.info("No alerts recorded.")

    # ── LOGS ─────────────────────────────────────────────────────────────────
    elif "LOGS" in page_sel:
        st.markdown('<div class="sec-hdr">— SYSTEM EVENT LOG</div>', unsafe_allow_html=True)
        lf,lr = st.columns([3,1])
        with lf: lvl = st.selectbox("Filter",["ALL","INFO","SUCCESS","WARN","ALERT"],label_visibility="collapsed")
        with lr:
            if st.button("⬛ CLEAR",use_container_width=True): st.session_state.system_log=[]; st.rerun()
        logs = st.session_state.system_log
        if lvl!="ALL": logs=[l for l in logs if l["l"]==lvl]
        if logs:
            cm={"INFO":"color:#4a7a96","SUCCESS":"color:#00ff85","WARN":"color:#ffd700","ALERT":"color:#ff4444"}
            html2=""
            for e in reversed(logs[-100:]):
                s=cm.get(e["l"],""); html2+=f'<span style="color:#1e3a52">[{e["t"]}]</span><span style="{s}"> [{e["l"]}]</span> {e["m"]}<br>'
            st.markdown(f'<div class="logbox">{html2}</div>',unsafe_allow_html=True)
        else: st.info("No log entries.")

    # ── TRAIN ────────────────────────────────────────────────────────────────
    elif "TRAIN" in page_sel:
        st.markdown('<div class="sec-hdr">— TRAINING MONITOR</div>', unsafe_allow_html=True)
        HCSV = Path(r"E:\4TH YEAR PROJECT\HUMAN\runs\detect\train_fast\results.csv")
        TCSV = Path(r"E:\4TH YEAR PROJECT\NEW TOOLS\runs\detect\train_fast\results.csv")

        def load_csv(p):
            if not p.exists(): return None
            try: df=pd.read_csv(p); df.columns=[c.strip() for c in df.columns]; return df
            except: return None

        def show_training(df, model, target):
            if df is None or df.empty:
                st.markdown(f'<div class="holo-panel" style="margin-bottom:1rem"><div class="mcard-lbl">{model}</div><div style="color:rgba(0,255,247,.28);font-size:.8rem;padding:.5rem 0">Waiting for training to start…</div></div>', unsafe_allow_html=True)
                return
            ep=len(df); pct=min(ep/target,1.0)
            last=df.iloc[-1]; best=df.get("metrics/mAP50(B)",pd.Series([0])).max()
            box=float(last.get("train/box_loss",0))
            st.markdown(f'<div class="holo-panel" style="margin-bottom:1rem"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem"><div style="font-family:Orbitron;font-size:.7rem;font-weight:700;color:var(--c1)">{model}</div><div style="font-family:Share Tech Mono;font-size:.62rem;color:rgba(0,255,247,.45)">Epoch {ep}/{target} · {"✅ Done" if best>=0.65 else "⏳ Training"}</div></div>', unsafe_allow_html=True)
            st.progress(pct)
            st.markdown(f"""<div class="metric-grid" style="margin:.6rem 0">
<div class="mcard"><div class="mcard-lbl">Best mAP@50</div><div class="mcard-val {'vg' if best>=0.65 else 'vw'}">{best*100:.1f}%</div></div>
<div class="mcard"><div class="mcard-lbl">Box Loss</div><div class="mcard-val va">{box:.4f}</div></div>
<div class="mcard"><div class="mcard-lbl">Progress</div><div class="mcard-val vc">{pct*100:.0f}%</div></div>
</div>""", unsafe_allow_html=True)
            if "metrics/mAP50(B)" in df.columns:
                fig3=go.Figure()
                fig3.add_trace(go.Scatter(y=df["metrics/mAP50(B)"]*100, name="mAP@50",
                    line=dict(color="#00fff7",width=2), fill="tozeroy", fillcolor="rgba(0,255,247,.05)"))
                fig3.add_hline(y=65, line_dash="dot", line_color="#ffd700", annotation_text="65% target")
                fig3.update_layout(height=160, margin=dict(t=10,b=20,l=20,r=10),
                    paper_bgcolor="#000c18", plot_bgcolor="#000c18", showlegend=False,
                    xaxis=dict(title="Epoch",gridcolor="#0d1f35",tickfont={"color":"#1e3a52"}),
                    yaxis=dict(title="mAP%",gridcolor="#0d1f35",tickfont={"color":"#1e3a52"},range=[0,100]))
                st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
            st.markdown('</div>', unsafe_allow_html=True)

        show_training(load_csv(HCSV), "🧍 HUMAN MODEL", 75)
        show_training(load_csv(TCSV), "🔧 TOOLS MODEL", 60)
        tc1,tc2 = st.columns([3,1])
        with tc1: st.caption(f"Last refreshed: {datetime.datetime.now().strftime('%H:%M:%S')}")
        with tc2:
            if st.button("🔄 REFRESH", use_container_width=True): st.rerun()

    # Footer
    st.markdown(f"""
<div style="text-align:center;padding:1.5rem 0 .5rem;font-family:'Share Tech Mono',monospace;
  font-size:.55rem;color:rgba(0,255,247,.18);letter-spacing:.15em;
  border-top:1px solid rgba(0,255,247,.06);margin-top:2.5rem">
⬡ SAFEGUARD AI · 2060-CLASS COMMAND INTERFACE · SESSION {st.session_state.session_id} · YOLOv11n · FP16 · CUDA:RTX4060
</div>""", unsafe_allow_html=True)

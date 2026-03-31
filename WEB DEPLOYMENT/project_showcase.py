"""
SafeGuard AI — Jarvis-Style Project Showcase
=============================================
Run: streamlit run deployment/project_showcase.py
"""
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import plotly.graph_objects as go
from showcase_styles import CSS, JS_SCRIPT

st.set_page_config(
    page_title="SafeGuard AI | Industrial Safety Intelligence",
    page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed"
)
st.markdown(CSS, unsafe_allow_html=True)
components.html(JS_SCRIPT, height=0)

# ── NAV ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="sg-nav">
  <div class="sg-logo">🛡 SAFEGUARD AI</div>
  <div class="sg-links">
    <a href="#overview">Overview</a>
    <a href="#architecture">Architecture</a>
    <a href="#pipeline">Pipeline</a>
    <a href="#usecases">Use Cases</a>
    <a href="#analysis">Analysis</a>
    <a href="#future">Future</a>
    <a href="#models">Models</a>
  </div>
  <div class="sg-badge">RESEARCH BUILD v2.1</div>
</div>""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="sg-hero">
  <div class="sg-badge-pill">⬡ AI-POWERED &nbsp;·&nbsp; REAL-TIME &nbsp;·&nbsp; MULTI-MODEL FUSION</div>
  <div class="sg-h1">Industrial Safety<br>Intelligence System</div>
  <div class="sg-type">
    <span class="sg-type-target" data-text="An autonomous computer vision pipeline monitoring CCTV streams in real time — detecting workers, verifying PPE compliance, tracking hazardous tools, and firing instant safety alerts."></span>
  </div>
  <div class="sg-stats">
    <div class="sg-stat"><div class="sg-num cnt-3">3</div><div class="sg-lbl">Parallel AI Models</div></div>
    <div class="sg-stat"><div class="sg-num cnt-30">30</div><div class="sg-lbl">FPS Real-Time</div></div>
    <div class="sg-stat"><div class="sg-num cnt-11">11</div><div class="sg-lbl">Detection Classes</div></div>
    <div class="sg-stat"><div class="sg-num cnt-94">94%</div><div class="sg-lbl">Human mAP@50</div></div>
    <div class="sg-stat"><div class="sg-num cnt-21">21K+</div><div class="sg-lbl">Training Images</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 01 OVERVIEW ───────────────────────────────────────────────────────
st.markdown('<a name="overview"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec">
  <div class="sg-tag">// 01 — PROJECT OVERVIEW</div>
  <div class="sg-title">What Is SafeGuard AI?</div>
  <div class="sg-desc">SafeGuard AI monitors industrial CCTV streams using three fine-tuned YOLOv11 models running in parallel — detecting workers, verifying PPE compliance, and tracking tools. No human oversight required.</div>
  <div class="sg-grid sg-g3">
    <div class="sg-card"><div class="ci">🧠</div><div class="ct">Autonomous Detection</div><div class="cb">Three specialised models run in parallel — human, PPE, tools — each fine-tuned on domain-specific industrial datasets for maximum CCTV accuracy.</div></div>
    <div class="sg-card"><div class="ci">⚡</div><div class="ct">Real-Time at 30 FPS</div><div class="cb">YOLOv11n nano achieves 28–30 FPS on RTX 4060. Live CCTV monitoring without perceptible lag — critical for real industrial deployment.</div></div>
    <div class="sg-card"><div class="ci">🚨</div><div class="ct">Tiered Alert System</div><div class="cb">PPE violations flagged instantly. Tool abandonment: WARNING at 25s, CRITICAL at 35s. All events logged and exportable as PDF compliance reports.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">🔗</div><div class="ct">Zero Infrastructure Change</div><div class="cb">Works with any existing RTSP/USB camera. Fully on-premise — no cloud dependency, no data leaves the network. Plug and protect.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">📊</div><div class="ct">Analytics Dashboard</div><div class="cb">Streamlit dashboard with live video feeds, historical violation trends, session logs, model metrics, and one-click PDF report generation.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">🏗️</div><div class="ct">Industrial-First Training</div><div class="cb">Trained on 21K+ real CCTV images (CrowdHuman, CCTV-Person, Mechanical-10000). Handles occlusion, lighting variation, and crowd-dense scenes.</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 02 ARCHITECTURE ───────────────────────────────────────────────────
st.markdown('<a name="architecture"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec sg-sec-alt">
  <div class="sg-tag">// 02 — SYSTEM ARCHITECTURE</div>
  <div class="sg-title">Cascade Fusion Pipeline</div>
  <div class="sg-desc">Three independent YOLOv11n models run per-frame. Outputs merge in the Spatial Fusion Engine which computes PPE compliance and tool hazard states in real time.</div>
  <div style="margin-top:1.5rem;">
    <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem;">
      <div class="tooltip-wrap">
        <span class="pill" style="cursor:default;font-size:.8rem;padding:.4rem 1rem;">🧍 MODEL: HUMAN</span>
        <div class="tip"><div class="tip-title">Human Detection</div>YOLOv11n · 21,425 images<br>CCTV-Person + CrowdHuman<br>nc:1 (person) · Target ~94% mAP<br><span style="color:#10b981">✦ CUDA GPU Accelerated</span></div>
      </div>
      <div class="tooltip-wrap">
        <span class="pill-p pill" style="cursor:default;font-size:.8rem;padding:.4rem 1rem;">🦺 MODEL: PPE</span>
        <div class="tip"><div class="tip-title">PPE Detection</div>YOLOv11n · 11 clean classes<br>helmet · vest · glove · mask…<br>Target 85–92% mAP<br><span style="color:#10b981">✦ CUDA GPU Accelerated</span></div>
      </div>
      <div class="tooltip-wrap">
        <span class="pill-a pill" style="cursor:default;font-size:.8rem;padding:.4rem 1rem;">🔧 MODEL: TOOLS</span>
        <div class="tip"><div class="tip-title">Tools Detection</div>YOLOv11n · 9,300 images<br>drill · hammer · pliers · screwdriver · wrench<br>Target 85%+ mAP<br><span style="color:#10b981">✦ CUDA GPU Accelerated</span></div>
      </div>
      <span class="gpu-badge">● GPU ACTIVE · RTX 4060</span>
    </div>
  </div>
""", unsafe_allow_html=True)

ARCH = Path(r"E:\4TH YEAR PROJECT\deployment\architecture_diagram.jpg")
if ARCH.exists():
    st.markdown('<div class="arch-frame">', unsafe_allow_html=True)
    st.image(str(ARCH), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    def arch_fig():
        BG="#0f172a";C="#00f5ff";P="#7c3aed";A="#f59e0b";R="#ef4444";G="#10b981";CD="#1e293b";M="#64748b";W="#e2e8f0"
        fig=go.Figure()
        fig.update_layout(paper_bgcolor=BG,plot_bgcolor=BG,margin=dict(l=5,r=5,t=20,b=5),height=700,
            xaxis=dict(visible=False,range=[0,10]),yaxis=dict(visible=False,range=[0,10]),showlegend=False)
        def n(x0,y0,x1,y1,col,t,s1="",s2="",s3="",ic=""):
            fig.add_shape(type="rect",x0=x0,y0=y0,x1=x1,y1=y1,line=dict(color=col,width=2),fillcolor=CD,layer="below")
            fig.add_shape(type="rect",x0=x0,y0=y1-.05,x1=x1,y1=y1,line=dict(width=0),fillcolor=col,layer="above")
            cx=(x0+x1)/2;cy=(y0+y1)/2;rows=[r for r in [ic,t,s1,s2,s3] if r]
            cs=[col,W,M,M,G];ss=[15,10,8,8,8]
            for i,(tx,c2,s) in enumerate(zip(rows,cs,ss)):
                yp=cy+.22*(len(rows)/2-i-.5)+(.1 if ic else 0)
                fig.add_annotation(x=cx,y=yp,text=tx,showarrow=False,font=dict(color=c2,size=s,family="Inter"),xanchor="center",yanchor="middle")
        def ar(x0,y0,x1,y1,col=C):
            fig.add_annotation(x=x1,y=y1,ax=x0,ay=y0,xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=2,arrowwidth=1.8,arrowcolor=col,arrowsize=1)
        def lb(x,y,t,col=M,s=9):
            fig.add_annotation(x=x,y=y,text=t,showarrow=False,font=dict(color=col,size=s,family="Inter"),xanchor="center",yanchor="middle")
        def hl(y):
            fig.add_shape(type="line",x0=.1,y0=y,x1=9.9,y1=y,line=dict(color="#1e293b",dash="dot",width=1))
        hl(8.3);hl(5.6);hl(3.4)
        lb(.35,9.1,"<b>INPUT</b>");lb(.35,6.9,"<b>INFERENCE</b>");lb(.35,4.4,"<b>FUSION</b>");lb(.35,2.1,"<b>OUTPUT</b>")
        n(.6,8.55,2.2,9.75,C,"CCTV CAMERA","RTSP·USB·File","H.264/H.265",ic="📷")
        ar(2.2,9.15,2.9,9.15);n(2.9,8.55,4.5,9.75,C,"FRAME CAPTURE","OpenCV","30 FPS·Async",ic="🖼️")
        ar(4.5,9.15,5.2,9.15);n(5.2,8.55,7.1,9.75,P,"PRE-PROCESSOR","Resize 1280px","Normalise·Tensor",ic="⚙️")
        ar(6.15,8.55,6.15,8.3,P);lb(5.0,8.15,"<b>PARALLEL GPU INFERENCE — YOLOv11n</b>",C,9)
        ar(6.15,8.0,1.9,7.8,M);ar(6.15,8.0,5.0,7.8,M);ar(6.15,8.0,8.1,7.8,M)
        n(.6,5.8,3.2,7.75,C,"MODEL 1: HUMAN","YOLOv11n · 21,425","CCTV+CrowdHuman","~94% mAP@50",ic="🧍")
        n(3.9,5.8,6.1,7.75,P,"MODEL 2: PPE","YOLOv11n · 11 cls","helmet·vest·glove","85–92% mAP@50",ic="🦺")
        n(6.9,5.8,9.4,7.75,A,"MODEL 3: TOOLS","YOLOv11n · 9,300","drill·hammer·pliers","85%+ mAP@50",ic="🔧")
        ar(1.9,5.8,2.5,5.6,C);ar(5.0,5.8,5.0,5.6,P);ar(8.15,5.8,7.5,5.6,A)
        lb(5.0,5.45,"<b>SPATIAL FUSION ENGINE</b>",A,9)
        n(.6,3.55,3.1,5.3,C,"IoU MATCHER","PPE→nearest worker","IoU ≥ 0.08","Bounding overlap",ic="🧮")
        ar(3.1,4.4,3.6,4.4,C);n(3.6,3.55,6.4,5.3,P,"HAZARD ZONE","Tool abandonment","WARNING→25s","CRITICAL→35s",ic="📐")
        ar(6.4,4.4,6.9,4.4,P);n(6.9,3.55,9.4,5.3,A,"COMPLIANCE","PPE per worker","vs. required rules","% per session",ic="⚖️")
        ar(1.85,3.55,1.2,3.4,C);ar(5.0,3.55,3.1,3.4,P);ar(8.15,3.55,5.9,3.4,A);ar(8.15,3.55,8.7,3.4,A)
        lb(5.0,3.25,"<b>OUTPUT LAYER</b>",M,9)
        n(.5,1.5,2.1,3.2,C,"DASHBOARD","Streamlit·OpenCV","Annotated 30FPS",ic="🖥️")
        n(2.3,1.5,3.9,3.2,R,"ALERT ENGINE","Tiered warnings","Real-time log",ic="🚨")
        n(4.1,1.5,5.7,3.2,P,"PDF REPORTS","Compliance","Auditor export",ic="📄")
        n(5.9,1.5,7.5,3.2,A,"SQLITE DB","Historical","Analytics",ic="🗄️")
        n(7.7,1.5,9.5,3.2,G,"REST API","FastAPI","External",ic="🌐")
        return fig
    st.plotly_chart(arch_fig(), use_container_width=True, config={"displayModeBar":False})

st.markdown("""
  <div style="text-align:center;padding:1.5rem 4rem 3rem;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#f59e0b;letter-spacing:2px;margin-bottom:.75rem;">TECH STACK</div>
    <span class="pill">YOLOv11n</span><span class="pill">Python 3.12</span><span class="pill">OpenCV</span><span class="pill">PyTorch 2.7</span><span class="pill">CUDA 11.8</span>
    <span class="pill-p pill">Ultralytics</span><span class="pill-p pill">Streamlit</span>
    <span class="pill-a pill">SQLite</span><span class="pill-a pill">FastAPI</span><span class="pill-a pill">RTX 4060</span>
  </div>
</div>""", unsafe_allow_html=True)

# ── 03 PIPELINE ───────────────────────────────────────────────────────
st.markdown('<a name="pipeline"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec">
  <div class="sg-tag">// 03 — DETECTION PIPELINE</div>
  <div class="sg-title">6-Step Live Processing Flow</div>
  <div class="sg-desc">Every video frame passes through this pipeline in under 33 milliseconds — achieving 30 FPS real-time monitoring.</div>
  <div class="sg-grid sg-g2" style="margin-top:2rem;">
    <div class="sg-step"><div class="sn">01</div><div><div class="ct">Frame Acquisition</div><div class="cb">Captured from RTSP/USB/file via OpenCV. Resized to 1280 px for high-resolution detection of small PPE items like gloves and goggles often missed at lower resolutions.</div></div></div>
    <div class="sg-step"><div class="sn">02</div><div><div class="ct">Parallel GPU Inference</div><div class="cb">All three YOLO models run on the same frame simultaneously on CUDA. Human → worker boxes; PPE → gear detections; Tools → tool positions and classes.</div></div></div>
    <div class="sg-step"><div class="sn sp">03</div><div><div class="ct">Spatial IoU Fusion</div><div class="cb">Fusion Engine matches PPE detections to the nearest worker via Intersection-over-Union (IoU ≥ 0.08). Each worker gets a personalised PPE inventory — present and missing items.</div></div></div>
    <div class="sg-step"><div class="sn sp">04</div><div><div class="ct">Hazard Zone Analysis</div><div class="cb">Tool positions tracked across frames. A tool in a hazard zone for 25 s triggers WARNING. At 35 s with no worker nearby, CRITICAL ALERT fires and is logged instantly.</div></div></div>
    <div class="sg-step"><div class="sn sa">05</div><div><div class="ct">Compliance Check</div><div class="cb">Each worker's PPE inventory validated against required PPE set. Missing items (no_helmet, no_vest, no_glove) trigger violations. Session compliance % computed per worker.</div></div></div>
    <div class="sg-step"><div class="sn sa">06</div><div><div class="ct">Output & Logging</div><div class="cb">Annotated frame rendered live on dashboard. All violations written to SQLite with ISO timestamps. PDF compliance reports generated on demand for safety auditors.</div></div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 04 USE CASES ──────────────────────────────────────────────────────
st.markdown('<a name="usecases"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec sg-sec-alt">
  <div class="sg-tag">// 04 — DEPLOYMENT CONTEXTS</div>
  <div class="sg-title">Where Can It Be Deployed?</div>
  <div class="sg-desc">Any industrial environment with existing CCTV can deploy SafeGuard AI without hardware changes.</div>
  <div class="sg-grid sg-g3" style="margin-top:2rem;">
    <div class="sg-use"><div class="ui">🏗️</div><div class="ut">Construction Sites</div><div class="ub">Helmet, vest and glove compliance on active floors. Abandoned power tools near scaffolding detected instantly. Reduces accident liability and insurance premiums.</div></div>
    <div class="sg-use"><div class="ui">🏭</div><div class="ut">Manufacturing Plants</div><div class="ub">Enforce PPE zones near heavy machinery. Track tool placement on assembly lines. Alert when equipment enters restricted zones without a worker present.</div></div>
    <div class="sg-use"><div class="ui">⛏️</div><div class="ut">Mining Operations</div><div class="ub">Verify respiratory masks, goggles and boots in hazardous extraction areas. Monitor tool abandonment near high-risk shafts and tunnels.</div></div>
    <div class="sg-use"><div class="ui">🛢️</div><div class="ut">Oil &amp; Gas Facilities</div><div class="ub">Enforce strict PPE near flammable environments. Generate timestamped compliance evidence for mandatory regulatory audits and insurer reviews.</div></div>
    <div class="sg-use"><div class="ui">⚡</div><div class="ut">Power &amp; Utilities</div><div class="ub">Ensure insulated gloves and shields near high-voltage equipment. Detect tool misplacement near electrical panels and substations instantly.</div></div>
    <div class="sg-use"><div class="ui">🏥</div><div class="ut">Hospitals &amp; Labs</div><div class="ub">Mask and glove compliance in sterile zones. Detect sharp instruments left unattended in clinical or research environments.</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 05 ANALYSIS ───────────────────────────────────────────────────────
st.markdown('<a name="analysis"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec">
  <div class="sg-tag">// 05 — SYSTEM DIAGNOSTICS</div>
  <div class="sg-title">Strengths &amp; Limitations</div>
  <div class="pc-grid">
    <div class="pro-box">
      <div class="bh pro">✦ &nbsp;Strengths</div>
      <div class="pc-item"><div class="dot pro-d"></div><div><strong style="color:#fff">Real-time at 30 FPS</strong> — YOLOv11n nano on mid-range GPU, no lag on live CCTV feeds.</div></div>
      <div class="pc-item"><div class="dot pro-d"></div><div><strong style="color:#fff">Three-model cascade</strong> — Separate specialised models maximise individual accuracy vs. one merged model.</div></div>
      <div class="pc-item"><div class="dot pro-d"></div><div><strong style="color:#fff">Fully on-premise</strong> — Footage never leaves the network. No cloud cost or latency.</div></div>
      <div class="pc-item"><div class="dot pro-d"></div><div><strong style="color:#fff">21K+ training images</strong> — CCTV-Person + CrowdHuman for robust real-world generalisation.</div></div>
      <div class="pc-item"><div class="dot pro-d"></div><div><strong style="color:#fff">Zero infra change</strong> — RTSP, USB, and file feeds work out-of-the-box.</div></div>
      <div class="pc-item"><div class="dot pro-d"></div><div><strong style="color:#fff">Legal-grade audit trail</strong> — Every violation timestamped in SQLite with PDF export.</div></div>
    </div>
    <div class="con-box">
      <div class="bh con">⚠ &nbsp;Limitations</div>
      <div class="pc-item"><div class="dot con-d"></div><div><strong style="color:#fff">Occlusion sensitivity</strong> — Workers behind machinery may have PPE missed; IoU degrades in crowds.</div></div>
      <div class="pc-item"><div class="dot con-d"></div><div><strong style="color:#fff">GPU dependency</strong> — Degrades to ~8 FPS on CPU only. Requires discrete GPU for real-time use.</div></div>
      <div class="pc-item"><div class="dot con-d"></div><div><strong style="color:#fff">Fixed camera angles</strong> — Trained on overhead CCTV views. Fish-eye lenses need domain fine-tuning.</div></div>
      <div class="pc-item"><div class="dot con-d"></div><div><strong style="color:#fff">Low-light performance</strong> — Without IR cameras, detection drops significantly in dim conditions.</div></div>
      <div class="pc-item"><div class="dot con-d"></div><div><strong style="color:#fff">No cross-camera tracking</strong> — Each camera is independent; workers re-identified per camera.</div></div>
      <div class="pc-item"><div class="dot con-d"></div><div><strong style="color:#fff">Alert fatigue risk</strong> — High-activity sites may need suppression tuning to reduce alert volume.</div></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 06 FUTURE ─────────────────────────────────────────────────────────
st.markdown('<a name="future"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec sg-sec-alt">
  <div class="sg-tag">// 06 — FUTURE ROADMAP</div>
  <div class="sg-title">Next-Generation Capabilities</div>
  <div class="sg-desc">SafeGuard AI is a strong foundation. These upgrades would evolve it from an intelligent monitor to a proactive safety co-pilot.</div>
  <div class="sg-grid sg-g2" style="margin-top:2.5rem;">
    <div class="sg-card"><div class="ci">👤</div><div class="ct">Re-ID &amp; Multi-Camera Tracking</div><div class="cb">ReID models (OSNet, CLIP-ReID) to track individuals across cameras. Violations tied to specific workers across an entire site.</div></div>
    <div class="sg-card"><div class="ci">🌙</div><div class="ct">Thermal &amp; Low-Light Fusion</div><div class="cb">Fuse RGB with thermal IR for 24/7 detection. Domain-adaptive training on night-shift data via Stable Diffusion augmentation pipelines.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">🤖</div><div class="ct">LLM Safety Analyst</div><div class="cb">Local LLM (Mistral/LLaMA 3) receives structured detection logs and generates natural-language incident reports with root-cause inferences.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">📡</div><div class="ct">Edge Deployment — Jetson Orin</div><div class="cb">TensorRT INT8 export. Each camera becomes an intelligent edge node on Jetson Orin NX — no central GPU server, sub-15ms latency.</div></div>
    <div class="sg-card"><div class="ci">🔮</div><div class="ct">Predictive Near-Miss AI</div><div class="cb">Spatiotemporal transformer trained on incident history to predict near-miss events by analysing movement trajectories and tool proximity.</div></div>
    <div class="sg-card"><div class="ci">🌐</div><div class="ct">Federated Learning Network</div><div class="cb">Multiple sites share model gradients (not images) to a hub, continuously improving global accuracy while preserving full site privacy.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">🗣️</div><div class="ct">Voice &amp; Wearable Integration</div><div class="cb">Connect to smart wearables for bi-directional alerts. Workers notified via earpiece when PPE is missing — real-time closed-loop correction.</div></div>
    <div class="sg-card sg-card-p"><div class="ci">📱</div><div class="ct">Mobile Supervisor App</div><div class="cb">Native iOS/Android app — push notifications on critical violations, live feeds, compliance scores, and one-tap PDF sharing for supervisors.</div></div>
  </div>
</div>""", unsafe_allow_html=True)

# ── 07 MODELS ─────────────────────────────────────────────────────────
st.markdown('<a name="models"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="sg-sec">
  <div class="sg-tag">// 07 — TRAINING INFRASTRUCTURE</div>
  <div class="sg-title">Model Specifications</div>
  <div class="sg-grid sg-g3">
    <div class="sg-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <div class="ci" style="margin:0;font-size:2rem;">🧍</div>
        <div><span class="pill" style="font-size:.68rem;">YOLOv11n</span> &nbsp;<span class="gpu-badge" style="font-size:.6rem;">GPU</span></div>
      </div>
      <div class="ct">Human Detection Model</div>
      <div class="cb" style="margin-top:.6rem;">
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Dataset:</span> CCTV-Person + CrowdHuman</div>
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Split:</span> 15,357 train / 4,046 val / 2,022 test</div>
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Classes:</span> 1 — person</div>
        <div style="margin:.3rem 0;color:var(--g);font-weight:600;">Target mAP@50: ~94%</div>
      </div>
    </div>
    <div class="sg-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <div class="ci" style="margin:0;font-size:2rem;">🦺</div>
        <div><span class="pill" style="font-size:.68rem;">YOLOv11n</span> &nbsp;<span class="gpu-badge" style="font-size:.6rem;">GPU</span></div>
      </div>
      <div class="ct">PPE Detection Model</div>
      <div class="cb" style="margin-top:.6rem;">
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Dataset:</span> Custom merged PPE</div>
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Classes:</span> 11 — helmet, vest, glove, shoe, mask, goggle + negatives</div>
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Resolution:</span> 640 px</div>
        <div style="margin:.3rem 0;color:var(--g);font-weight:600;">Target mAP@50: 85–92%</div>
      </div>
    </div>
    <div class="sg-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
        <div class="ci" style="margin:0;font-size:2rem;">🔧</div>
        <div><span class="pill" style="font-size:.68rem;">YOLOv11n</span> &nbsp;<span class="gpu-badge" style="font-size:.6rem;">GPU</span></div>
      </div>
      <div class="ct">Tools Detection Model</div>
      <div class="cb" style="margin-top:.6rem;">
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Dataset:</span> Mechanical-10000 (Roboflow)</div>
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Classes:</span> 5 — drill, hammer, pliers, screwdriver, wrench</div>
        <div style="margin:.3rem 0;"><span style="color:var(--c)">Resolution:</span> 640 px</div>
        <div style="margin:.3rem 0;color:var(--g);font-weight:600;">Target mAP@50: 85%+</div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────
st.markdown("""
<div class="sg-sep"></div>
<div class="sg-foot">
  <div style="font-family:'Orbitron',monospace;font-size:clamp(1.4rem,4vw,2.2rem);font-weight:900;
    background:linear-gradient(135deg,#fff,var(--c),var(--p));-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;margin-bottom:1rem;">SafeGuard AI</div>
  <div style="color:#64748b;font-size:.95rem;max-width:600px;margin:0 auto 2.5rem;line-height:1.8;">
    Real-time industrial safety intelligence — protecting workers through the power of computer vision.
  </div>
  <div>
    <span class="pill">YOLOv11</span><span class="pill-p pill">Deep Learning</span>
    <span class="pill-a pill">Computer Vision</span><span class="pill">Real-Time AI</span>
    <span class="pill-p pill">Industrial Safety</span><span class="pill-a pill">CCTV Analytics</span>
  </div>
  <div class="sg-foot-copy">4th Year Engineering Project &nbsp;·&nbsp; Computer Vision &amp; AI &nbsp;·&nbsp; PyTorch · Ultralytics · Streamlit &nbsp;·&nbsp; 2025–2026</div>
</div>""", unsafe_allow_html=True)

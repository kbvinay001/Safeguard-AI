"""
ui_styles.py — SafeGuard AI — Dinero-Style Dark Dashboard CSS
==============================================================
Design language adapted from AI-Money-Mentor Dinero frontend:
  • #060913 deep space background
  • Glassmorphism cards with backdrop-blur
  • Playfair Display + Inter typography
  • Per-module colour accent glow on hover
  • Constellation particle bg via injected JS canvas
  • Rounded 18-20px cards (not sharp corners)
  • Smooth fadeSlideUp entrance animations
"""
from pathlib import Path


def get_css(_bg_b64: str = "") -> str:
    """Return the full <style> block for the Streamlit app."""
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══════════════════════════════════════════════════════════════
   DESIGN TOKENS
══════════════════════════════════════════════════════════════ */
:root {
  --bg:          #060913;
  --bg2:         #0A0F1E;
  --bg3:         #111827;
  --card:        rgba(14, 20, 36, 0.60);
  --card-hover:  rgba(14, 20, 36, 0.85);
  --border:      rgba(255, 255, 255, 0.06);
  --border-h:    rgba(255, 255, 255, 0.12);

  --blue:        #3B82F6;
  --blue-d:      #2563EB;
  --green:       #22C55E;
  --green-d:     #16A34A;
  --orange:      #F97316;
  --purple:      #A78BFA;
  --yellow:      #FBBF24;
  --cyan:        #22D3EE;
  --red:         #F87171;

  --txt:         #F1F5F9;
  --txt2:        #94A3B8;
  --muted:       #64748B;
  --bright:      #FFFFFF;

  --radius-card: 20px;
  --radius-pill: 9999px;
  --blur:        blur(16px);
  --trans:       0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ══════════════════════════════════════════════════════════════
   GLOBAL RESET
══════════════════════════════════════════════════════════════ */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--txt) !important;
  -webkit-font-smoothing: antialiased;
}

/* Page background */
.stApp {
  background: var(--bg) !important;
  min-height: 100vh;
}

/* Main content area */
.main .block-container {
  padding-top: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  max-width: 100% !important;
}

/* hide default streamlit chrome */
section[data-testid="stSidebar"]      { display: none !important; }
button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
header[data-testid="stHeader"]        { display: none !important; }
.stDeployButton, footer, #MainMenu    { display: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(59,130,246,.3); border-radius: 4px; }

/* ══════════════════════════════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════════════════════════════ */
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes pulse-glow {
  0%,100% { box-shadow: 0 0 4px #4ADE80; }
  50%     { box-shadow: 0 0 14px #4ADE80, 0 0 28px rgba(74,222,128,.3); }
}
@keyframes float {
  0%,100% { transform: translateY(0); }
  50%     { transform: translateY(-6px); }
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ══════════════════════════════════════════════════════════════
   NAVBAR
══════════════════════════════════════════════════════════════ */
.sg-nav {
  position: sticky; top: 0; z-index: 999;
  background: rgba(6,9,19,0.75);
  backdrop-filter: var(--blur);
  -webkit-backdrop-filter: var(--blur);
  border-bottom: 1px solid var(--border);
}
.sg-nav-inner {
  max-width: 1280px; margin: 0 auto;
  padding: 0 2rem; height: 64px;
  display: flex; align-items: center; justify-content: space-between;
  animation: fadeIn .4s ease;
}
.sg-logo {
  display: flex; align-items: center; gap: 10px;
  font-weight: 800; font-size: 1.2rem; color: var(--bright);
  letter-spacing: -0.02em;
}
.sg-logo-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, var(--blue), var(--green));
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem; color: #fff; font-weight: 800;
}
.sg-badge-beta {
  font-size: .65rem; font-weight: 700; letter-spacing: .05em;
  padding: 3px 10px; border-radius: var(--radius-pill);
  background: rgba(59,130,246,.15); color: #60A5FA;
}
.sg-nav-right {
  display: flex; align-items: center; gap: .5rem;
}
.sg-btn-ghost {
  display: flex; align-items: center; gap: 6px;
  padding: 8px 14px; border-radius: 10px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--txt2); font-size: .85rem; font-weight: 500;
  cursor: pointer; transition: all var(--trans);
  font-family: 'Inter', sans-serif;
}
.sg-btn-ghost:hover {
  background: rgba(255,255,255,.05);
  color: var(--txt); border-color: var(--border-h);
}

/* ══════════════════════════════════════════════════════════════
   SUB NAV  (page tabs)
══════════════════════════════════════════════════════════════ */
.sg-subnav {
  background: rgba(6,9,19,0.6);
  backdrop-filter: var(--blur);
  border-bottom: 1px solid var(--border);
  padding: .6rem 2rem;
  display: flex; gap: .3rem; flex-wrap: wrap;
  max-width: 100%;
  animation: fadeIn .3s ease;
}
.sg-tab {
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .4rem 1rem; border-radius: 10px;
  font-size: .8rem; font-weight: 500;
  color: var(--muted); cursor: pointer;
  border: 1px solid transparent;
  background: transparent;
  transition: all var(--trans);
  white-space: nowrap;
}
.sg-tab:hover {
  color: var(--txt2);
  background: rgba(255,255,255,.04);
}
.sg-tab.active {
  color: var(--bright);
  background: rgba(59,130,246,.1);
  border-color: rgba(59,130,246,.2);
}

/* ══════════════════════════════════════════════════════════════
   HERO
══════════════════════════════════════════════════════════════ */
.sg-hero {
  position: relative; z-index: 2;
  padding: 4rem 2rem 2.5rem;
  max-width: 1280px; margin: 0 auto;
  animation: fadeSlideUp .6s ease both;
}
.sg-welcome-badge {
  display: inline-flex; align-items: center; gap: 8px;
  background: rgba(22,163,74,.12);
  border: 1px solid rgba(22,163,74,.25);
  padding: 7px 16px; border-radius: var(--radius-pill);
  color: #4ADE80; font-size: .82rem; font-weight: 600;
  margin-bottom: 1.4rem;
  animation: fadeIn .6s ease both;
}
.pulse-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: #4ADE80;
  animation: pulse-glow 2s infinite;
  flex-shrink: 0;
}
.sg-hero-title {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 800; line-height: 1.15;
  letter-spacing: -0.03em; margin-bottom: 1rem;
  color: var(--bright);
  animation: fadeSlideUp .6s .1s ease both;
}
.text-gradient {
  background: linear-gradient(135deg, var(--blue), var(--green));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.sg-hero-sub {
  color: var(--muted); font-size: 1.05rem;
  max-width: 540px; line-height: 1.65;
  animation: fadeSlideUp .6s .2s ease both;
}

/* ══════════════════════════════════════════════════════════════
   CONTENT WRAPPER
══════════════════════════════════════════════════════════════ */
.sg-wrap {
  position: relative; z-index: 2;
  max-width: 1280px; margin: 0 auto;
  padding: 0 2rem 4rem;
}

/* ══════════════════════════════════════════════════════════════
   STAT CARDS ROW (4 across)
══════════════════════════════════════════════════════════════ */
.sg-stats-row {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 3rem;
}
.sg-stat-card {
  background: rgba(14,20,36,.6);
  backdrop-filter: var(--blur);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 24px 20px;
  text-align: center;
  transition: border-color var(--trans), box-shadow var(--trans), transform var(--trans);
  animation: fadeSlideUp .5s ease both;
  cursor: default;
}
.sg-stat-card:hover {
  border-color: rgba(59,130,246,.25);
  box-shadow: 0 0 32px rgba(59,130,246,.08);
  transform: translateY(-2px);
}
.sg-stat-icon { font-size: 1.5rem; margin-bottom: 10px; }
.sg-stat-value {
  font-size: 1.35rem; font-weight: 800;
  letter-spacing: -0.02em; color: var(--bright);
  margin-bottom: 4px;
}
.sg-stat-label {
  font-size: .7rem; color: var(--muted);
  font-weight: 600; text-transform: uppercase;
  letter-spacing: .08em;
}

/* ══════════════════════════════════════════════════════════════
   SECTION TITLE
══════════════════════════════════════════════════════════════ */
.sg-section-title {
  font-size: 1.05rem; font-weight: 700;
  color: var(--txt2); letter-spacing: .03em;
  margin-bottom: 1.2rem;
}

/* ══════════════════════════════════════════════════════════════
   FEATURE MODULE CARDS
══════════════════════════════════════════════════════════════ */
.sg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}
.sg-feature-card {
  position: relative;
  background: rgba(14,20,36,.5);
  backdrop-filter: var(--blur);
  border: 1px solid var(--border);
  border-radius: var(--radius-card);
  overflow: hidden;
  cursor: pointer;
  transition: transform var(--trans), border-color var(--trans), box-shadow var(--trans);
  animation: fadeSlideUp .55s ease both;
  text-align: left;
}
.sg-feature-card:hover {
  transform: translateY(-5px);
  border-color: var(--c-border, rgba(59,130,246,.25));
  box-shadow: 0 14px 48px rgba(0,0,0,.35),
              0 0 40px color-mix(in srgb, var(--c-accent, #3B82F6) 12%, transparent);
}
/* top-edge colour glow */
.sg-card-glow {
  position: absolute; top: 0; left: 10%; right: 10%;
  height: 2px;
  background: var(--c-accent, #3B82F6);
  border-radius: 0 0 8px 8px;
  opacity: 0; filter: blur(4px);
  transition: opacity var(--trans);
}
.sg-feature-card:hover .sg-card-glow { opacity: .7; }

.sg-card-inner { padding: 28px 28px 24px; }
.sg-card-emoji {
  font-size: 2rem; margin-bottom: 14px;
  filter: saturate(1.2);
  display: block;
}
.sg-card-title {
  font-size: 1.1rem; font-weight: 700;
  color: var(--bright); margin-bottom: 8px;
  letter-spacing: -0.01em;
}
.sg-card-desc {
  font-size: .88rem; color: var(--muted);
  line-height: 1.65; margin-bottom: 20px;
  min-height: 44px;
}
.sg-card-link {
  display: inline-flex; align-items: center; gap: 6px;
  font-size: .82rem; font-weight: 600;
  color: var(--c-accent, var(--blue));
  transition: gap var(--trans);
}
.sg-feature-card:hover .sg-card-link { gap: 10px; }
.sg-card-arrow { font-size: .9rem; }

/* ══════════════════════════════════════════════════════════════
   INFO / ALERT BOXES
══════════════════════════════════════════════════════════════ */
.sg-box {
  background: rgba(14,20,36,.5);
  backdrop-filter: var(--blur);
  border: 1px solid var(--border);
  border-radius: 16px; padding: 1.2rem 1.4rem;
  margin-bottom: .8rem; animation: fadeIn .4s ease;
}
.sg-box.info  { border-left: 3px solid var(--blue); }
.sg-box.good  { border-left: 3px solid var(--green); }
.sg-box.warn  { border-left: 3px solid var(--orange); }
.sg-box.crit  { border-left: 3px solid var(--red); }

/* ══════════════════════════════════════════════════════════════
   STATUS PILLS / BADGES
══════════════════════════════════════════════════════════════ */
.pill {
  display: inline-flex; align-items: center; gap: 5px;
  padding: .25rem .75rem; border-radius: var(--radius-pill);
  font-size: .72rem; font-weight: 600; letter-spacing: .04em;
}
.pill-on  { background: rgba(34,197,94,.12); color: #4ADE80; border: 1px solid rgba(34,197,94,.25); }
.pill-off { background: rgba(248,113,113,.1); color: var(--red); border: 1px solid rgba(248,113,113,.2); }
.pill-warn{ background: rgba(249,115,22,.1); color: var(--orange); border: 1px solid rgba(249,115,22,.2); }

/* ══════════════════════════════════════════════════════════════
   LOG CONSOLE
══════════════════════════════════════════════════════════════ */
.sg-log {
  background: #020509;
  border: 1px solid rgba(59,130,246,.15);
  border-radius: 14px;
  padding: .9rem 1.1rem;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: .74rem; color: #4a7a6a;
  max-height: 280px; overflow-y: auto;
  line-height: 1.8;
}
.li { color: #60A5FA; } .ls { color: #4ADE80; }
.lw { color: var(--yellow); } .la { color: var(--red); }
.lt { color: var(--muted); }

/* ══════════════════════════════════════════════════════════════
   TABLES
══════════════════════════════════════════════════════════════ */
.sg-table { width:100%; border-collapse:collapse; font-size:.84rem; }
.sg-table th {
  background: rgba(14,20,36,.8); color: var(--muted);
  padding: .6rem 1rem; text-align:left;
  font-size:.68rem; letter-spacing:.1em; text-transform:uppercase;
  border-bottom: 1px solid var(--border);
}
.sg-table td {
  padding: .55rem 1rem;
  border-bottom: 1px solid rgba(255,255,255,.03);
  color: var(--txt2);
}
.sg-table tr:hover td { background: rgba(59,130,246,.04); }

/* ══════════════════════════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
══════════════════════════════════════════════════════════════ */
/* Metric cards */
div[data-testid="metric-container"] {
  background: rgba(14,20,36,.6) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: .8rem 1.1rem !important;
  backdrop-filter: var(--blur) !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
  font-family: 'Inter' !important; color: var(--bright) !important;
  font-size: 1.4rem !important; font-weight: 800 !important;
}

/* Buttons */
.stButton > button {
  background: rgba(59,130,246,.15) !important;
  color: var(--blue) !important;
  border: 1px solid rgba(59,130,246,.3) !important;
  border-radius: 10px !important;
  font-family: 'Inter' !important;
  font-size: .85rem !important;
  font-weight: 600 !important;
  transition: all var(--trans) !important;
}
.stButton > button:hover {
  background: rgba(59,130,246,.25) !important;
  box-shadow: 0 4px 20px rgba(59,130,246,.2) !important;
  transform: translateY(-1px) !important;
}
.stButton > button[kind="primary"] {
  background: var(--blue-d) !important;
  color: #fff !important;
  border-color: var(--blue-d) !important;
  box-shadow: 0 4px 16px rgba(37,99,235,.35) !important;
}

/* File uploader */
.stFileUploader {
  background: rgba(14,20,36,.5) !important;
  border: 1px dashed rgba(59,130,246,.3) !important;
  border-radius: 14px !important;
}

/* Inputs */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
  background: rgba(14,20,36,.7) !important;
  border: 1px solid var(--border) !important;
  color: var(--txt) !important;
  border-radius: 10px !important;
  font-family: 'Inter' !important;
}
.stTextInput input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
}
.stTextInput input::placeholder { color: var(--muted) !important; }

/* Select */
div[data-baseweb="select"] div {
  background: rgba(14,20,36,.8) !important;
  border-color: var(--border) !important;
  color: var(--txt) !important;
  border-radius: 10px !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--blue) !important; }
.stSlider div[class*="StyledTrackHighlight"] { background: var(--blue) !important; }

/* Progress */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--blue), var(--green)) !important;
}

/* Expander */
.streamlit-expanderHeader {
  background: rgba(14,20,36,.7) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  font-family: 'Inter' !important;
  font-size: .85rem !important;
  font-weight: 600 !important;
  color: var(--txt2) !important;
}

/* Dataframe */
.stDataFrame {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* ══════════════════════════════════════════════════════════════
   PLOTLY CHART CONTAINERS
══════════════════════════════════════════════════════════════ */
.js-plotly-plot {
  border-radius: 16px !important;
  overflow: hidden;
}

/* ══════════════════════════════════════════════════════════════
   FOOTER
══════════════════════════════════════════════════════════════ */
.sg-footer {
  text-align: center; padding: 2rem 1rem;
  font-size: .75rem; color: var(--muted);
  border-top: 1px solid var(--border);
  margin-top: 2rem;
  animation: fadeIn .6s ease;
}
.sg-footer span { color: var(--blue); }

/* ══════════════════════════════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════════════════════════════ */
@media (max-width: 900px) {
  .sg-stats-row { grid-template-columns: repeat(2, 1fr); }
  .sg-grid      { grid-template-columns: 1fr; }
  .sg-hero      { padding: 2.5rem 1.5rem 2rem; }
  .sg-wrap      { padding: 0 1.5rem 3rem; }
}
@media (max-width: 600px) {
  .sg-stats-row { grid-template-columns: 1fr 1fr; }
  .sg-hero-title { font-size: 1.9rem; }
}
</style>"""


# Constellation particle canvas JS — injected via st.components.v1.html
CONSTELLATION_JS = """
<canvas id="sg-canvas" style="
  position:fixed;top:0;left:0;width:100%;height:100%;
  z-index:0;pointer-events:none;opacity:0.45;
"></canvas>
<script>
(function(){
  const c = document.getElementById('sg-canvas');
  const ctx = c.getContext('2d');
  let anim;
  const N = 70;
  const pts = [];

  function resize(){
    c.width  = window.innerWidth;
    c.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  for(let i=0;i<N;i++){
    pts.push({
      x: Math.random()*c.width,
      y: Math.random()*c.height,
      vx:(Math.random()-.5)*.35,
      vy:(Math.random()-.5)*.35,
      r: Math.random()*1.4+.4,
    });
  }

  function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    pts.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0||p.x>c.width)  p.vx*=-1;
      if(p.y<0||p.y>c.height) p.vy*=-1;
      ctx.beginPath();
      ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
      ctx.fillStyle='rgba(59,130,246,0.55)';
      ctx.fill();
    });
    for(let i=0;i<pts.length;i++){
      for(let j=i+1;j<pts.length;j++){
        const dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<130){
          ctx.beginPath();
          ctx.moveTo(pts[i].x,pts[i].y);
          ctx.lineTo(pts[j].x,pts[j].y);
          ctx.strokeStyle=`rgba(59,130,246,${.07*(1-d/130)})`;
          ctx.stroke();
        }
      }
    }
    anim=requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""


# ── Placeholder values ──────────────────────────────────────────────────────
BG_B64  = ""
BAN_B64 = ""

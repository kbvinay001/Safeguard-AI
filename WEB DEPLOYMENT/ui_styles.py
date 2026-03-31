"""
ui_styles.py — SafeGuard AI Cyberpunk Terminal CSS
====================================================
All CSS for the cyberpunk / military-HUD aesthetic.
Colours: deep-black backgrounds, #00ffcc (cyber-cyan) accents,
border-corner brackets, animated status dots, monospace fonts.
"""
import base64
from pathlib import Path

_HERE = Path(__file__).parent


def _b64(p: str) -> str:
    try:
        return base64.b64encode(open(p, "rb").read()).decode()
    except Exception:
        return ""


BG_B64  = _b64(str(_HERE / "bg_image.jpg"))
BAN_B64 = _b64(str(_HERE / "industrial_banner.png"))


def get_css(bg_b64: str) -> str:
    bg_css = (
        f"""background:linear-gradient(180deg,rgba(4,8,6,0.95)0%,rgba(4,8,6,0.98)100%),
            url("data:image/jpeg;base64,{bg_b64}") center/cover no-repeat fixed !important;"""
        if bg_b64 else
        "background: #040806 !important;"
    )
    return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══════════════════════════════ VARIABLES ══════════════════════════════ */
:root {{
  --bg:       #040806;
  --bg2:      #080f0b;
  --card:     #0a100d;
  --card2:    #0f1a14;
  --bdr:      #0d2b1e;
  --bdr2:     #1a4030;
  --cyan:     #00ffcc;
  --cyan2:    #00ccaa;
  --cyan-dim: rgba(0,255,204,0.08);
  --cyan-glo: rgba(0,255,204,0.25);
  --red:      #ff3333;
  --orange:   #ff6a00;
  --yellow:   #ffe033;
  --green:    #00ff88;
  --txt:      #a0c4b0;
  --txt2:     #c8e8d8;
  --muted:    #3a6050;
  --bright:   #e0fff0;
  --mono:     'Share Tech Mono', 'Courier New', monospace;
  --head:     'Orbitron', 'Share Tech Mono', monospace;
  --body:     'Inter', sans-serif;
}}

/* ═══════════════════════════════ GLOBAL ════════════════════════════════ */
html, body, [class*="css"] {{
  font-family: var(--body) !important;
  color: var(--txt) !important;
}}
.main .block-container {{
  padding-top: 0 !important;
  max-width: 1600px;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
}}

/* Full-page background */
.stApp {{
  {bg_css}
}}

/* Hide Streamlit chrome */
section[data-testid="stSidebar"] {{ display: none !important; }}
button[data-testid="stSidebarCollapsedControl"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ display: none !important; }}
.stDeployButton {{ display: none !important; }}
footer {{ display: none !important; }}
#MainMenu {{ visibility: hidden !important; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: var(--bg2); }}
::-webkit-scrollbar-thumb {{ background: var(--bdr2); border-radius: 2px; }}
::-webkit-scrollbar-thumb:hover {{ background: var(--cyan2); }}

/* ═══════════════════════════════ ANIMATIONS ════════════════════════════ */
@keyframes fadeInUp  {{ from {{ opacity:0; transform:translateY(20px); }} to {{ opacity:1; transform:translateY(0); }} }}
@keyframes fadeIn    {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
@keyframes blink     {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0; }} }}
@keyframes pulse     {{ 0%,100% {{ opacity:1; box-shadow: 0 0 6px var(--cyan); }} 50% {{ opacity:.6; box-shadow: 0 0 14px var(--cyan); }} }}
@keyframes glow-in   {{ from {{ text-shadow: none; }} to {{ text-shadow: 0 0 18px var(--cyan); }} }}
@keyframes scan-line {{
  0%   {{ transform: translateY(-100%); }}
  100% {{ transform: translateY(100vh); }}
}}
@keyframes border-flash {{
  0%,100% {{ border-color: var(--bdr2); }}
  50%      {{ border-color: var(--cyan); }}
}}

/* ═══════════════════════════════ TOP NAV ═══════════════════════════════ */
.topnav {{
  position: sticky; top: 0; z-index: 999;
  background: rgba(4,8,6,0.96); backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--bdr2);
  padding: 0 2rem;
  display: flex; align-items: center; justify-content: space-between;
  height: 56px;
  animation: fadeIn .4s ease;
}}

.topnav-brand {{
  display: flex; align-items: center; gap: 0.7rem;
}}
.topnav-dots {{
  display: flex; gap: 5px; align-items: center;
}}
.topnav-dot {{
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--cyan); animation: pulse 2s infinite;
}}
.topnav-dot:nth-child(2) {{ background: var(--orange); animation-delay: .3s; }}
.topnav-dot:nth-child(3) {{ background: var(--muted); animation: none; opacity: .4; }}

.topnav-logo-text {{
  font-family: var(--head);
  font-size: 1.05rem; font-weight: 700;
  color: var(--bright); letter-spacing: 0.15em;
  text-transform: uppercase;
}}
.topnav-logo-text em {{
  color: var(--cyan); font-style: normal;
}}
.topnav-subtitle {{
  font-family: var(--mono);
  font-size: 0.58rem; color: var(--muted);
  letter-spacing: 0.08em; text-transform: uppercase;
  margin-left: 0.3rem;
}}

.topnav-tabs {{
  display: flex; gap: 0.4rem;
}}
.topnav-tab {{
  font-family: var(--mono);
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase;
  padding: 0.4rem 1.2rem;
  border: 1px solid var(--bdr2);
  border-radius: 3px;
  background: transparent; color: var(--muted);
  cursor: pointer; transition: all .25s;
  display: flex; align-items: center; gap: 0.4rem;
}}
.topnav-tab:hover {{
  border-color: var(--cyan); color: var(--cyan);
  background: var(--cyan-dim);
}}
.topnav-tab.active {{
  border-color: var(--cyan); color: var(--bright);
  background: var(--cyan-dim);
  box-shadow: 0 0 12px var(--cyan-glo) inset;
}}
.topnav-tab.active .tab-dot {{
  background: var(--red); animation: pulse 1.2s infinite;
}}
.tab-dot {{
  width: 6px; height: 6px; border-radius: 50%; background: var(--muted);
}}
.tab-radio {{
  width: 10px; height: 10px; border-radius: 50%;
  border: 1px solid var(--muted); display: flex; align-items: center; justify-content: center;
}}
.tab-radio-inner {{
  width: 4px; height: 4px; border-radius: 50%; background: transparent;
  transition: background .2s;
}}
.topnav-tab.active .tab-radio-inner {{
  background: var(--cyan);
}}

/* ═══════════════════════════════ SUB NAV ═══════════════════════════════ */
.subnav {{
  background: rgba(4,8,6,0.8);
  border-bottom: 1px solid var(--bdr);
  padding: 0.5rem 2rem;
  display: flex; gap: 0.3rem; flex-wrap: wrap;
  animation: fadeIn .4s ease;
}}
.subnav-btn {{
  font-family: var(--mono);
  font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
  padding: 0.3rem 0.9rem;
  border: 1px solid var(--bdr);
  border-radius: 2px;
  background: transparent; color: var(--muted);
  cursor: pointer; transition: all .2s;
  display: flex; align-items: center; gap: 0.35rem;
}}
.subnav-btn:hover {{
  border-color: var(--cyan2); color: var(--txt2);
  background: rgba(0,255,204,0.04);
}}
.subnav-btn.active {{
  border-color: var(--cyan); color: var(--cyan);
  background: var(--cyan-dim);
  box-shadow: 0 0 8px var(--cyan-glo) inset;
}}
.subnav-icon {{ font-size: 0.7rem; opacity: .7; }}

/* ═══════════════════════════════ SECTION HEADERS ═══════════════════════ */
.sg-section {{
  display: flex; align-items: center; gap: 0.7rem;
  margin: 1.8rem 0 1rem;
  font-family: var(--mono);
  font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.2em; text-transform: uppercase;
  color: var(--cyan);
}}
.sg-section::before {{
  content: '—';
  color: var(--cyan2);
}}
.sg-section::after {{
  content: '';
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--bdr2), transparent);
}}

/* ═══════════════════════════════ HERO ══════════════════════════════════ */
.hero {{
  text-align: center; padding: 3.5rem 2rem 2.5rem;
  animation: fadeInUp .7s ease;
}}
.hero-eyebrow {{
  font-family: var(--mono);
  font-size: 0.65rem; letter-spacing: 0.35em; text-transform: uppercase;
  color: var(--cyan); margin-bottom: 0.8rem;
  opacity: .8;
}}
.hero h1 {{
  font-family: var(--head);
  font-size: 3.2rem; font-weight: 700; color: var(--cyan);
  letter-spacing: 0.05em; margin: 0 0 1rem;
  text-shadow: 0 0 30px rgba(0,255,204,0.35);
  animation: glow-in 2s ease forwards;
}}
.hero-desc {{
  font-family: var(--body);
  font-size: 0.92rem; color: var(--muted); max-width: 620px;
  margin: 0 auto 2rem; line-height: 1.7;
}}

/* Corner-bracket card */
.sg-card {{
  position: relative;
  background: var(--card); border: 1px solid var(--bdr2);
  border-radius: 2px;
  padding: 1.2rem 1.4rem;
  transition: all .3s;
  animation: fadeInUp .6s ease;
}}
.sg-card::before, .sg-card::after {{
  content: '';
  position: absolute;
  width: 12px; height: 12px;
  border-color: var(--cyan);
  border-style: solid;
}}
.sg-card::before {{ top: -1px; left: -1px; border-width: 2px 0 0 2px; }}
.sg-card::after  {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}
.sg-card:hover {{
  border-color: var(--bdr2);
  background: var(--card2);
  box-shadow: 0 4px 24px rgba(0,0,0,.4);
}}

/* ═══════════════════════════════ COMMAND CENTER METRICS ════════════════ */
.cmd-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
  gap: 0.8rem;
  margin: 0.6rem 0 1.4rem;
}}
.cmd-card {{
  position: relative;
  background: var(--card);
  border: 1px solid var(--bdr);
  border-radius: 2px;
  padding: 1rem 1.2rem 0.9rem;
  overflow: hidden;
  transition: border-color .3s;
}}
.cmd-card::after {{
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
}}
.cmd-card.cy::after {{ background: var(--cyan); }}
.cmd-card.gn::after {{ background: var(--green); }}
.cmd-card.rd::after {{ background: var(--red); }}
.cmd-card.or::after {{ background: var(--orange); }}
.cmd-card.ye::after {{ background: var(--yellow); }}
.cmd-card.bl::after {{ background: #4488ff; }}
.cmd-card:hover {{ border-color: var(--bdr2); }}

.cmd-label {{
  font-family: var(--mono);
  font-size: 0.58rem; letter-spacing: 0.18em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 0.5rem;
}}
.cmd-value {{
  font-family: var(--mono);
  font-size: 1.8rem; font-weight: 700; line-height: 1;
  color: var(--cyan);
}}
.cmd-value.gn {{ color: var(--green); }}
.cmd-value.rd {{ color: var(--red); }}
.cmd-value.or {{ color: var(--orange); }}
.cmd-value.ye {{ color: var(--yellow); }}
.cmd-value.bl {{ color: #66aaff; }}
.cmd-value.wh {{ color: var(--bright); }}
.cmd-sub {{
  font-family: var(--mono);
  font-size: 0.62rem; color: var(--muted); margin-top: 0.3rem;
}}

/* ═══════════════════════════════ QUICK ACTIONS ════════════════════════ */
.qa-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px,1fr));
  gap: 0.9rem;
  margin: 0.6rem 0 1.4rem;
}}
.qa-card {{
  position: relative;
  background: var(--card); border: 1px solid var(--bdr);
  border-radius: 2px;
  padding: 1.2rem 1.4rem;
  cursor: pointer; transition: all .25s;
}}
.qa-card::before {{ content:''; position:absolute; top:0; left:0; width:2px; height:0; background:var(--cyan); transition:height .25s; }}
.qa-card::after  {{ content:''; position:absolute; bottom:0; right:0; width:0; height:2px; background:var(--cyan); transition:width .25s; }}
.qa-card:hover::before {{ height:100%; }}
.qa-card:hover::after  {{ width:100%; }}
.qa-card:hover {{ background: var(--card2); border-color: var(--bdr2); }}
.qa-title {{
  font-family: var(--mono);
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--cyan); margin-bottom: 0.5rem;
  display: flex; align-items: center; gap: 0.4rem;
}}
.qa-desc {{
  font-size: 0.8rem; color: var(--muted); line-height: 1.6;
}}

/* ═══════════════════════════════ FEATURE CARDS ════════════════════════ */
.feature-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px,1fr));
  gap: 0.9rem;
  margin: 0.8rem 0 1.6rem;
}}
.feature-card {{
  position: relative;
  background: var(--card); border: 1px solid var(--bdr);
  border-radius: 2px;
  padding: 1.4rem 1.3rem;
  transition: all .3s;
  animation: fadeInUp .7s ease;
}}
.feature-card:hover {{
  border-color: var(--cyan2);
  background: var(--card2);
  transform: translateY(-2px);
  box-shadow: 0 6px 24px rgba(0,0,0,.4);
}}
.feature-icon {{ font-size: 1.6rem; margin-bottom: 0.7rem; }}
.feature-title {{
  font-family: var(--mono);
  font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--cyan); margin-bottom: 0.4rem;
}}
.feature-desc {{ font-size: 0.8rem; color: var(--muted); line-height: 1.6; }}

/* ═══════════════════════════════ LOG CONSOLE ═══════════════════════════ */
.logc {{
  background: #020706;
  border: 1px solid var(--bdr2);
  border-radius: 2px;
  padding: 0.8rem 1rem;
  font-family: var(--mono);
  font-size: 0.73rem; color: #4a8a6a;
  max-height: 280px; overflow-y: auto;
  line-height: 1.75;
}}
.li  {{ color: #3a9aff; }}
.lw  {{ color: var(--yellow); }}
.la  {{ color: var(--red); }}
.ls  {{ color: var(--green); }}
.lt  {{ color: var(--muted); }}

/* ═══════════════════════════════ TABLES ═══════════════════════════════ */
.sg-table {{ width:100%; border-collapse:collapse; font-family:var(--mono); font-size:0.77rem; }}
.sg-table th {{
  background: var(--card2); color: var(--muted);
  padding: 0.5rem 0.9rem; text-align: left;
  font-size: 0.62rem; letter-spacing: 0.12em; text-transform: uppercase;
  border-bottom: 1px solid var(--bdr2);
}}
.sg-table td {{
  padding: 0.5rem 0.9rem; border-bottom: 1px solid var(--bdr);
  color: var(--txt2);
}}
.sg-table tr:hover td {{ background: var(--card2); }}

/* ═══════════════════════════════ STATUS PILLS ░══════════════════════════ */
.pill {{
  display:inline-flex; align-items:center; gap:0.3rem;
  padding:0.18rem 0.6rem; border-radius:2px;
  font-family:var(--mono); font-size:0.62rem; font-weight:700;
  letter-spacing:0.1em; text-transform:uppercase;
}}
.pill-on  {{ background:rgba(0,255,136,.12); color:var(--green); border:1px solid rgba(0,255,136,.3); }}
.pill-off {{ background:rgba(255,51,51,.1);  color:var(--red);   border:1px solid rgba(255,51,51,.25); }}
.pill-warn{{ background:rgba(255,106,0,.1);  color:var(--orange);border:1px solid rgba(255,106,0,.25); }}

/* ═══════════════════════════════ INFO BOXES ════════════════════════════ */
.ic {{
  position:relative;
  background: var(--card); border: 1px solid var(--bdr);
  border-radius: 2px; padding: 0.9rem 1.1rem; margin-bottom: 0.7rem;
}}
.ic.sf {{ border-left:2px solid var(--green); }}
.ic.wn {{ border-left:2px solid var(--orange); }}
.ic.dg {{ border-left:2px solid var(--red); }}
.ic.in {{ border-left:2px solid var(--cyan); }}

/* ═══════════════════════════════ WIDGET OVERRIDES ══════════════════════ */
div[data-testid="metric-container"] {{
  background: var(--card) !important;
  border: 1px solid var(--bdr2) !important;
  border-radius: 2px !important;
  padding: 0.7rem 1rem !important;
}}
div[data-testid="metric-container"] [data-testid="metric-value"] {{
  font-family: var(--mono) !important; color: var(--cyan) !important;
}}

/* Primary buttons */
.stButton > button {{
  background: transparent !important;
  color: var(--cyan) !important;
  border: 1px solid var(--cyan) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  font-weight: 600 !important;
  transition: all .2s !important;
}}
.stButton > button:hover {{
  background: var(--cyan-dim) !important;
  box-shadow: 0 0 12px var(--cyan-glo) !important;
}}
/* Primary type */
.stButton > button[kind="primary"] {{
  background: var(--cyan-dim) !important;
  box-shadow: 0 0 8px var(--cyan-glo) inset !important;
}}

/* File uploader */
.stFileUploader {{
  background: var(--card) !important;
  border: 1px dashed var(--bdr2) !important;
  border-radius: 2px !important;
}}

/* Text input / selectbox */
.stTextInput input, .stSelectbox select, .stTextArea textarea {{
  background: var(--card2) !important;
  border: 1px solid var(--bdr2) !important;
  color: var(--cyan) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: 0.8rem !important;
}}
.stTextInput input::placeholder {{ color: var(--muted) !important; }}
.stTextInput input:focus {{
  border-color: var(--cyan) !important;
  box-shadow: 0 0 8px var(--cyan-glo) !important;
}}

/* Sliders */
.stSlider [data-baseweb="slider"] div[role="slider"] {{
  background: var(--cyan) !important; border: none !important;
}}
.stSlider div[class*="StyledTrackHighlight"] {{ background: var(--cyan) !important; }}

/* Expander */
.streamlit-expanderHeader {{
  background: var(--card2) !important;
  border: 1px solid var(--bdr) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.1em !important;
  color: var(--txt2) !important;
}}

/* Progress bar */
.stProgress > div > div > div {{
  background: var(--cyan) !important;
  box-shadow: 0 0 8px var(--cyan-glo) !important;
}}

/* Select box */
div[data-baseweb="select"] div {{
  background: var(--card2) !important;
  border-color: var(--bdr2) !important;
  color: var(--txt2) !important;
  font-family: var(--mono) !important;
}}

/* Dataframe */
.stDataFrame {{ border: 1px solid var(--bdr2) !important; border-radius: 2px !important; }}

/* ═══════════════════════════════ FOOTER ════════════════════════════════ */
.sg-footer {{
  text-align:center;
  font-family: var(--mono);
  font-size: 0.62rem; color: var(--muted); letter-spacing: 0.1em; text-transform: uppercase;
  padding: 0.8rem 0 1.2rem;
  border-top: 1px solid var(--bdr);
  margin-top: 2rem;
}}
.sg-footer span {{ color: var(--cyan2); }}
</style>"""

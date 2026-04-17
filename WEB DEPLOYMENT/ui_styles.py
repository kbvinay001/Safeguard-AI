"""ui_styles.py — SafeGuard AI Design System"""

def get_css() -> str:
    return """<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg:#060913; --bg2:#0A0F1E; --card:rgba(14,20,36,.62);
  --bdr:rgba(255,255,255,.06); --bdrh:rgba(255,255,255,.14);
  --blue:#3B82F6; --green:#22C55E; --orange:#F97316;
  --purple:#A78BFA; --yellow:#FBBF24; --cyan:#22D3EE; --red:#F87171;
  --txt:#F1F5F9; --txt2:#94A3B8; --muted:#64748B;
  --blur:blur(18px); --ease:.28s cubic-bezier(.4,0,.2,1);
}

html,body,[class*="css"]{
  font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif !important;
  -webkit-font-smoothing:antialiased;
}
.stApp{ background:var(--bg) !important; min-height:100vh; }
.main .block-container{ padding:0 !important; max-width:100% !important; }
section[data-testid="stSidebar"],
button[data-testid="stSidebarCollapsedControl"],
header[data-testid="stHeader"],
.stDeployButton,footer,#MainMenu{ display:none !important; }
::-webkit-scrollbar{width:5px}
::-webkit-scrollbar-track{background:var(--bg)}
::-webkit-scrollbar-thumb{background:rgba(59,130,246,.3);border-radius:4px}

/* ── Animations ── */
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes pulse-glow{0%,100%{box-shadow:0 0 4px #4ADE80}50%{box-shadow:0 0 18px #4ADE80,0 0 36px rgba(74,222,128,.3)}}
@keyframes pulse-glow-warn{0%,100%{box-shadow:0 0 4px #FBBF24}50%{box-shadow:0 0 14px #FBBF24,0 0 28px rgba(251,191,36,.3)}}
@keyframes pulse-glow-alert{0%,100%{box-shadow:0 0 6px #F87171}50%{box-shadow:0 0 20px #F87171,0 0 40px rgba(248,113,113,.4)}}
@keyframes ripple{to{transform:scale(4);opacity:0}}
@keyframes clickPop{0%{transform:scale(1)}50%{transform:scale(.97)}100%{transform:scale(1)}}
@keyframes scanline{0%{top:-2px}100%{top:100%}}
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
@keyframes dotBlink{0%,80%,100%{opacity:.2}40%{opacity:1}}
@keyframes vignette-pulse{0%,100%{box-shadow:inset 0 0 0 rgba(248,113,113,0)}50%{box-shadow:inset 0 0 60px rgba(248,113,113,.12)}}

/* ── Navbar ── */
.sg-nav{
  position:sticky;top:0;z-index:999;
  background:rgba(6,9,19,.88);
  backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
  border-bottom:1px solid var(--bdr);
}
.sg-nav-inner{
  max-width:1240px;margin:0 auto;padding:0 2.5rem;height:66px;
  display:flex;align-items:center;justify-content:space-between;
}
.sg-logo{display:flex;align-items:center;gap:11px;font-weight:800;font-size:1.2rem;color:#fff;letter-spacing:-.02em}
.sg-logo-icon{
  width:38px;height:38px;
  background:linear-gradient(135deg,var(--blue),var(--green));
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  color:#fff;font-size:1.1rem;
}
.sg-badge{font-size:.63rem;font-weight:700;letter-spacing:.07em;padding:3px 9px;border-radius:9999px;background:rgba(59,130,246,.15);color:#60A5FA}
.sg-nav-right{display:flex;align-items:center;gap:.8rem}
.pill-on{
  display:inline-flex;align-items:center;gap:5px;
  padding:.3rem .9rem;border-radius:9999px;font-size:.74rem;font-weight:600;
  background:rgba(22,163,74,.12);color:#4ADE80;border:1px solid rgba(22,163,74,.25);
}

/* ── Back button area ── */
.sg-backrow{padding:.8rem 2.5rem 0;max-width:1240px;margin:0 auto}
.sg-backrow .stButton>button{
  background:rgba(14,20,36,.5) !important;border:1px solid var(--bdr) !important;
  color:var(--txt2) !important;border-radius:10px !important;font-size:.85rem !important;font-weight:500 !important;
}
.sg-backrow .stButton>button:hover{background:rgba(255,255,255,.06) !important;color:#fff !important}

/* ── Hero ── */
.sg-hero{
  max-width:1240px;margin:0 auto;
  padding:4.5rem 2.5rem 2rem;
}
.sg-welcome-badge{
  display:inline-flex;align-items:center;gap:8px;
  background:rgba(22,163,74,.12);border:1px solid rgba(22,163,74,.25);
  padding:8px 18px;border-radius:9999px;
  color:#4ADE80;font-size:.82rem;font-weight:600;margin-bottom:1.6rem;
  animation:fadeIn .6s ease both;
}
.pulse-dot{width:8px;height:8px;border-radius:50%;background:#4ADE80;flex-shrink:0;animation:pulse-glow 2s infinite}

/* BIG HEADING — guaranteed size */
.sg-hero-title{
  font-family:'Playfair Display',Georgia,serif !important;
  font-size:clamp(3rem,6vw,5rem) !important;
  font-weight:800 !important;line-height:1.05 !important;
  letter-spacing:-.03em !important;color:#fff !important;
  margin-bottom:1.1rem !important;
  animation:fadeSlideUp .6s .1s ease both;
}
.text-gradient{
  background:linear-gradient(135deg,var(--blue),var(--green));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.sg-hero-sub{
  color:var(--muted);font-size:1.05rem;max-width:550px;line-height:1.7;
  animation:fadeSlideUp .6s .2s ease both;
}

/* ── Wrapper ── */
.sg-wrap{max-width:1240px;margin:0 auto;padding:0 2.5rem 4rem}

/* ── Stat row ── */
.sg-stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:3.5rem}
.sg-stat-card{
  background:var(--card);backdrop-filter:var(--blur);
  border:1px solid var(--bdr);border-radius:18px;
  padding:26px 22px;text-align:center;
  transition:border-color var(--ease),box-shadow var(--ease),transform var(--ease);
  animation:fadeSlideUp .5s ease both;
  position:relative;overflow:hidden;
}
/* Subtle gradient border on hover */
.sg-stat-card::before{
  content:'';position:absolute;inset:-1px;border-radius:18px;
  background:linear-gradient(135deg,rgba(59,130,246,.4),rgba(34,211,238,.2),rgba(74,222,128,.3));
  opacity:0;z-index:-1;transition:opacity var(--ease);
}
.sg-stat-card:hover::before{opacity:1}
.sg-stat-card:hover{border-color:transparent;box-shadow:0 0 36px rgba(59,130,246,.12);transform:translateY(-3px)}
.sg-stat-icon{margin-bottom:12px;display:flex;align-items:center;justify-content:center}
.sg-stat-icon svg{opacity:.8}
.sg-stat-value{font-size:1.55rem;font-weight:800;letter-spacing:-.03em;color:#fff;margin-bottom:5px}
.sg-stat-label{font-size:.67rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.09em}

/* ── Section heading ── */
.sg-section-title{font-size:1.15rem;font-weight:700;color:var(--txt2);letter-spacing:.02em;margin-bottom:1.5rem}

/* ── MODULE CARDS  HTML visual + Explore button ── */
.mod-card-visual{
  background:var(--card);backdrop-filter:var(--blur);
  border:1px solid var(--bdr);border-radius:20px 20px 0 0;
  padding:30px 28px 22px;
  transition:border-color var(--ease),box-shadow var(--ease),transform var(--ease);
  cursor:pointer;
  animation:fadeSlideUp .55s ease both;
  position:relative;overflow:hidden;
}
/* top glow line */
.mod-card-visual::before{
  content:'';position:absolute;top:0;left:15%;right:15%;height:2px;
  background:var(--accent,#3B82F6);border-radius:0 0 6px 6px;
  opacity:0;filter:blur(3px);transition:opacity var(--ease);
}
.mod-card-visual:hover::before{opacity:.8}
.mod-card-visual:hover{transform:translateY(-5px);border-color:var(--accent,#3B82F6)}

.mod-icon-wrap{
  width:52px;height:52px;border-radius:14px;
  display:flex;align-items:center;justify-content:center;
  margin-bottom:18px;
  background:rgba(255,255,255,.05);
  flex-shrink:0;
}
.mod-icon-wrap svg{width:26px;height:26px;stroke:var(--accent,#3B82F6);fill:none;stroke-width:1.75;stroke-linecap:round;stroke-linejoin:round}
.mod-card-title{font-size:1.15rem;font-weight:700;color:#fff;margin-bottom:10px;letter-spacing:-.01em}
.mod-card-desc{font-size:.88rem;color:var(--muted);line-height:1.65;min-height:52px;margin-bottom:0}

/* Explore button — connects to card bottom */
.mod-explore .stButton>button{
  border-radius:0 0 20px 20px !important;
  border:1px solid var(--bdr) !important;border-top:1px solid rgba(255,255,255,.04) !important;
  background:rgba(14,20,36,.75) !important;
  color:var(--accent,#3B82F6) !important;
  font-size:.85rem !important;font-weight:600 !important;
  height:44px !important;
  transition:all var(--ease) !important;
  position:relative;overflow:hidden;
}
.mod-explore .stButton>button:hover{
  background:rgba(14,20,36,.95) !important;
  box-shadow:0 0 15px rgba(34,211,238,.2),0 0 30px rgba(34,211,238,.08) !important;
  text-shadow:0 0 8px rgba(34,211,238,.4);
  filter:brightness(1.15);
}
/* ripple click animation */
.mod-explore .stButton>button:active{animation:clickPop .2s ease !important}
.mod-explore .stButton>button::after{
  content:'';position:absolute;
  width:20px;height:20px;border-radius:50%;
  background:rgba(255,255,255,.25);
  transform:scale(0);opacity:1;
  transition:transform .5s,opacity .5s;
}
.mod-explore .stButton>button:active::after{animation:ripple .6s ease-out}

/* per-accent colours */
.acc-orange{--accent:#F97316} .acc-cyan{--accent:#22D3EE}
.acc-green{--accent:#4ADE80}  .acc-purple{--accent:#A78BFA}
.acc-yellow{--accent:#FBBF24} .acc-blue{--accent:#3B82F6}

/* ── Info boxes ── */
.sg-box{background:var(--card);backdrop-filter:var(--blur);border:1px solid var(--bdr);border-radius:16px;padding:1.2rem 1.5rem;margin-bottom:.9rem;animation:fadeIn .4s ease}
.sg-box.info{border-left:3px solid var(--blue)}
.sg-box.good{border-left:3px solid var(--green)}
.sg-box.warn{border-left:3px solid var(--orange)}

/* ── FSM State Pulsing Indicators ── */
.fsm-safe{
  display:inline-block;width:10px;height:10px;border-radius:50%;
  background:#4ADE80;animation:pulse-glow 2s infinite;
}
.fsm-warning{
  display:inline-block;width:10px;height:10px;border-radius:50%;
  background:#FBBF24;animation:pulse-glow-warn 1.2s infinite;
}
.fsm-alert{
  display:inline-block;width:10px;height:10px;border-radius:50%;
  background:#F87171;animation:pulse-glow-alert .8s infinite;
}

/* ── Live HUD Status Bar ── */
.sg-hud-status{
  display:flex;align-items:center;gap:12px;
  padding:.4rem .8rem;margin-top:.4rem;
  background:rgba(6,9,19,.7);border:1px solid rgba(255,255,255,.06);
  border-radius:8px;
}
.sg-stream-live{
  font-size:.78rem;font-weight:700;letter-spacing:.08em;
  color:#4ADE80;animation:pulse-glow 2s infinite;
  padding:2px 8px;border-radius:9999px;
  background:rgba(74,222,128,.1);border:1px solid rgba(74,222,128,.25);
}
.sg-stream-reconnecting{
  font-size:.78rem;font-weight:700;letter-spacing:.05em;
  color:#FBBF24;animation:pulse-glow-warn 1.2s infinite;
  padding:2px 8px;border-radius:9999px;
  background:rgba(251,191,36,.1);border:1px solid rgba(251,191,36,.25);
}
.sg-fsm-badge{
  font-size:.75rem;font-weight:700;letter-spacing:.05em;
  padding:2px 10px;border-radius:9999px;
  border:1px solid;background:rgba(0,0,0,.3);
}

/* ── Neon Glow Buttons (primary) ── */
.stButton>button{
  background:rgba(59,130,246,.12) !important;color:var(--blue) !important;
  border:1px solid rgba(59,130,246,.25) !important;border-radius:10px !important;
  font-family:'Inter',sans-serif !important;font-size:.85rem !important;font-weight:600 !important;
  transition:all var(--ease) !important;
}
.stButton>button:hover{
  background:rgba(59,130,246,.22) !important;
  transform:translateY(-1px) !important;
  box-shadow:0 0 15px rgba(34,211,238,.25),0 0 30px rgba(34,211,238,.1) !important;
  border-color:rgba(34,211,238,.4) !important;
  text-shadow:0 0 8px rgba(34,211,238,.5);
}
.stButton>button[kind="primary"]{
  background:#2563EB !important;color:#fff !important;border-color:#2563EB !important;
  box-shadow:0 4px 16px rgba(37,99,235,.35) !important;
}
.stButton>button[kind="primary"]:hover{
  box-shadow:0 0 20px rgba(59,130,246,.5),0 0 40px rgba(59,130,246,.2) !important;
  transform:translateY(-2px) !important;
}

div[data-testid="metric-container"]{background:var(--card) !important;border:1px solid var(--bdr) !important;border-radius:14px !important;padding:.8rem 1rem !important}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:#fff !important;font-size:1.4rem !important;font-weight:800 !important}
.stTextInput input,.stTextArea textarea{background:rgba(14,20,36,.7) !important;border:1px solid var(--bdr) !important;color:var(--txt) !important;border-radius:10px !important}
div[data-baseweb="select"] div{background:rgba(14,20,36,.8) !important;border-color:var(--bdr) !important;color:var(--txt) !important;border-radius:10px !important}
.stFileUploader{background:rgba(14,20,36,.5) !important;border:1px dashed rgba(59,130,246,.3) !important;border-radius:14px !important}
.streamlit-expanderHeader{background:rgba(14,20,36,.7) !important;border:1px solid var(--bdr) !important;border-radius:12px !important;font-size:.85rem !important;font-weight:600 !important;color:var(--txt2) !important}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--blue),var(--green)) !important}

/* ── Log ── */
.sg-log{background:#020509;border:1px solid rgba(59,130,246,.15);border-radius:14px;padding:.9rem 1.1rem;font-family:'JetBrains Mono','Fira Code',monospace;font-size:.74rem;color:#4a7a6a;max-height:300px;overflow-y:auto;line-height:1.9}
.li{color:#60A5FA}.ls{color:#4ADE80}.lw{color:#FBBF24}.la{color:#F87171}.lt{color:var(--muted)}

/* ── Table ── */
.sg-table{width:100%;border-collapse:collapse;font-size:.85rem}
.sg-table th{background:rgba(14,20,36,.8);color:var(--muted);padding:.7rem 1rem;text-align:left;font-size:.67rem;letter-spacing:.1em;text-transform:uppercase;border-bottom:1px solid var(--bdr)}
.sg-table td{padding:.6rem 1rem;border-bottom:1px solid rgba(255,255,255,.025);color:var(--txt2)}
.sg-table tr:hover td{background:rgba(59,130,246,.04)}

/* ── Footer ── */
.sg-footer{text-align:center;padding:2rem 1rem;font-size:.74rem;color:var(--muted);border-top:1px solid var(--bdr);margin-top:2rem}
.sg-footer span{color:var(--blue)}

/* ── Responsive ── */
@media(max-width:900px){.sg-stats-row{grid-template-columns:repeat(2,1fr)}.sg-hero{padding:3rem 1.5rem 2rem}.sg-wrap{padding:0 1.5rem 3rem}}
@media(max-width:640px){.sg-hero-title{font-size:2.5rem !important}}
</style>"""


# SVG icons — clean Feather-style line art
ICONS = {
    "analyse": '<svg viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="2"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>',
    "live":    '<svg viewBox="0 0 24 24"><path d="M1 6s4-4 11-4 11 4 11 4"/><path d="M5 10s3-3 7-3 7 3 7 3"/><path d="M9 14s1-1 3-1 3 1 3 1"/><line x1="12" y1="18" x2="12" y2="18"/><circle cx="12" cy="18" r="1" fill="currentColor"/></svg>',
    "analytics":'<svg viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
    "history": '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>',
    "system":  '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="2" x2="9" y2="4"/><line x1="15" y1="2" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="22"/><line x1="15" y1="20" x2="15" y2="22"/><line x1="20" y1="9" x2="22" y2="9"/><line x1="20" y1="14" x2="22" y2="14"/><line x1="2" y1="9" x2="4" y2="9"/><line x1="2" y1="14" x2="4" y2="14"/></svg>',
    "train":   '<svg viewBox="0 0 24 24"><circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/><path d="M12 7v4M12 11l-5.5 6M12 11l5.5 6"/><line x1="7" y1="19" x2="17" y2="19"/></svg>',
}

# Stat icons
STAT_ICONS = {
    "shield":'<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#4ADE80" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "film":  '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#3B82F6" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="2"/><line x1="7" y1="2" x2="7" y2="22"/><line x1="17" y1="2" x2="17" y2="22"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert": '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#F87171" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "cpu":   '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#A78BFA" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
}

CONSTELLATION_JS = """
<script>
(function(){
  var pd=window.parent.document;
  var old=pd.getElementById('sg-cv'); if(old)old.remove();
  var c=pd.createElement('canvas'); c.id='sg-cv';
  c.style.cssText='position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:0.38;';
  pd.body.appendChild(c);
  var ctx=c.getContext('2d'),anim,pts=[];
  function resize(){c.width=window.parent.innerWidth;c.height=window.parent.innerHeight;}
  resize(); window.parent.addEventListener('resize',resize);
  for(var i=0;i<65;i++) pts.push({x:Math.random()*c.width,y:Math.random()*c.height,vx:(Math.random()-.5)*.3,vy:(Math.random()-.5)*.3,r:Math.random()*1.3+.4});
  function draw(){
    ctx.clearRect(0,0,c.width,c.height);
    pts.forEach(function(p){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>c.width)p.vx*=-1;if(p.y<0||p.y>c.height)p.vy*=-1;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='rgba(59,130,246,.6)';ctx.fill();});
    for(var i=0;i<pts.length;i++)for(var j=i+1;j<pts.length;j++){var dx=pts[i].x-pts[j].x,dy=pts[i].y-pts[j].y,d=Math.sqrt(dx*dx+dy*dy);if(d<130){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(59,130,246,'+(0.07*(1-d/130))+')';ctx.lineWidth=1;ctx.stroke();}}
    anim=window.parent.requestAnimationFrame(draw);
  }
  draw();
})();
</script>
"""

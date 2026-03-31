"""
showcase_styles.py — CSS + JS for SafeGuard AI Jarvis-style interface
JS_HTML is injected via st.components.v1.html() so scripts execute properly.
"""

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');
:root{--c:#00f5ff;--p:#7c3aed;--a:#f59e0b;--r:#ef4444;--g:#10b981;
  --bg:#020817;--card:rgba(10,20,42,0.92);--bdr:rgba(0,245,255,0.18);}
.stApp{background:var(--bg)!important;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#e2e8f0;}
.block-container{padding:0!important;max-width:100%!important;}
header,footer,[data-testid="stDecoration"],[data-testid="collapsedControl"]{display:none!important;}
::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-thumb{background:var(--c);}
html{scroll-behavior:smooth;}

/* PROGRESS */
#sg-bar{position:fixed;top:0;left:0;height:2px;z-index:99999;width:0%;
  background:linear-gradient(90deg,var(--c),var(--p),var(--a));
  box-shadow:0 0 8px var(--c);transition:width .08s linear;pointer-events:none;}

/* NAV */
.sg-nav{position:sticky;top:0;z-index:9990;display:flex;align-items:center;
  justify-content:space-between;padding:.9rem 3rem;
  background:rgba(2,8,23,0.97);border-bottom:1px solid var(--bdr);
  backdrop-filter:blur(24px);}
.sg-logo{font-family:'Orbitron',monospace;font-size:1.25rem;font-weight:900;
  background:linear-gradient(90deg,var(--c),var(--p));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:3px;}
.sg-links{display:flex;gap:2rem;align-items:center;}
.sg-links a{color:#64748b;text-decoration:none;font-size:.78rem;font-weight:600;
  letter-spacing:1.5px;text-transform:uppercase;transition:all .25s;position:relative;padding:.2rem 0;}
.sg-links a::after{content:'';position:absolute;bottom:-2px;left:0;height:1.5px;width:0;
  background:var(--c);box-shadow:0 0 6px var(--c);transition:width .3s;}
.sg-links a:hover{color:var(--c);}
.sg-links a:hover::after,.sg-links a.active::after{width:100%;}
.sg-links a.active{color:var(--c);}
.sg-badge{background:linear-gradient(135deg,var(--c),var(--p));color:#000;
  padding:.35rem .9rem;border-radius:20px;font-size:.7rem;font-weight:800;letter-spacing:1.5px;}

/* CANVAS BG */
#sg-canvas{position:fixed;top:0;left:0;width:100%;height:100%;
  z-index:0;pointer-events:none;opacity:.35;}

/* HERO */
.sg-hero{position:relative;min-height:100vh;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;padding:4rem 2rem;z-index:1;}
.sg-hero::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(124,58,237,.22) 0%,transparent 70%),
             radial-gradient(ellipse 50% 40% at 80% 90%,rgba(0,245,255,.12) 0%,transparent 60%);}
.sg-badge-pill{display:inline-flex;align-items:center;gap:.5rem;
  background:rgba(0,245,255,.07);border:1px solid rgba(0,245,255,.25);
  border-radius:50px;padding:.45rem 1.4rem;font-size:.75rem;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;color:var(--c);margin-bottom:2rem;
  animation:pulse-ring 3s ease infinite;position:relative;z-index:1;}
@keyframes pulse-ring{0%,100%{box-shadow:0 0 0 0 rgba(0,245,255,.25)}
  50%{box-shadow:0 0 0 12px rgba(0,245,255,0)}}
.sg-h1{font-family:'Orbitron',monospace;font-size:clamp(2.4rem,6.5vw,5.5rem);
  font-weight:900;line-height:1.05;
  background:linear-gradient(135deg,#fff 0%,var(--c) 45%,var(--p) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:1.5rem;position:relative;z-index:1;}
.sg-type{font-size:clamp(.95rem,2.2vw,1.25rem);color:#64748b;max-width:700px;
  line-height:1.85;margin:0 auto 3rem;position:relative;z-index:1;}
.sg-type .cursor{display:inline-block;width:2px;height:1.1em;background:var(--c);
  vertical-align:text-bottom;animation:blink .7s step-end infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.sg-stats{display:flex;gap:3rem;justify-content:center;flex-wrap:wrap;position:relative;z-index:1;}
.sg-stat{text-align:center;padding:1.2rem 1.5rem;
  background:rgba(0,245,255,.04);border:1px solid rgba(0,245,255,.15);
  border-radius:16px;min-width:110px;transition:all .3s;position:relative;overflow:hidden;}
.sg-stat::before{content:'';position:absolute;top:0;left:0;right:0;height:1.5px;
  background:linear-gradient(90deg,transparent,var(--c),transparent);
  animation:scan 2s linear infinite;}
@keyframes scan{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
.sg-stat:hover{border-color:var(--c);box-shadow:0 0 20px rgba(0,245,255,.15);transform:translateY(-4px);}
.sg-num{font-family:'Orbitron',monospace;font-size:2rem;font-weight:900;
  background:linear-gradient(90deg,var(--c),var(--p));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sg-lbl{font-size:.68rem;color:#64748b;letter-spacing:2px;text-transform:uppercase;margin-top:.3rem;}

/* SECTIONS */
.sg-sec{padding:5rem 4rem;position:relative;z-index:1;}
.sg-sec-alt{background:rgba(8,16,36,.6);}
.sg-tag{font-family:'JetBrains Mono',monospace;font-size:.7rem;letter-spacing:4px;
  text-transform:uppercase;color:var(--c);margin-bottom:.6rem;}
.sg-title{font-family:'Orbitron',monospace;font-size:clamp(1.5rem,2.8vw,2.3rem);
  font-weight:700;display:inline-block;position:relative;margin-bottom:.8rem;
  background:linear-gradient(135deg,#fff,#94a3b8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.sg-title::after{content:'';position:absolute;bottom:-6px;left:0;height:2px;width:0;
  background:linear-gradient(90deg,var(--c),var(--p));
  box-shadow:0 0 8px var(--c);transition:width 1.1s cubic-bezier(.16,1,.3,1);}
.sg-title.iv::after{width:55%;}
.sg-desc{color:#64748b;font-size:.95rem;line-height:1.9;max-width:680px;margin-top:.4rem;}

/* CARDS */
.sg-grid{display:grid;gap:1.5rem;margin-top:2.5rem;}
.sg-g3{grid-template-columns:repeat(auto-fit,minmax(270px,1fr));}
.sg-g2{grid-template-columns:repeat(auto-fit,minmax(340px,1fr));}
.sg-card{background:var(--card);border:1px solid var(--bdr);border-radius:16px;
  padding:2rem;position:relative;overflow:hidden;cursor:default;
  transition:transform .35s,box-shadow .35s,border-color .35s;}
.sg-card::before{content:'';position:absolute;inset:-1px;border-radius:17px;
  background:linear-gradient(135deg,var(--c),var(--p),var(--a),var(--c));
  background-size:300% 300%;z-index:-1;opacity:0;
  transition:opacity .4s;animation:grad-shift 4s ease infinite;}
@keyframes grad-shift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
.sg-card:hover::before{opacity:1;}
.sg-card:hover{transform:translateY(-7px);
  box-shadow:0 16px 50px rgba(0,245,255,.13),0 0 0 1px rgba(0,245,255,.3);}
.sg-card .ci{font-size:2.2rem;margin-bottom:1rem;display:inline-block;
  transition:transform .3s;}
.sg-card:hover .ci{transform:scale(1.2) rotate(-5deg);}
.sg-card .ct{font-weight:700;font-size:1rem;color:#fff;margin-bottom:.5rem;}
.sg-card .cb{color:#64748b;font-size:.875rem;line-height:1.75;}
.sg-card-p{border-color:rgba(124,58,237,.35);}

/* USE CASES */
.sg-use{background:var(--card);border:1px solid var(--bdr);border-radius:16px;
  padding:2rem;position:relative;overflow:hidden;
  transition:transform .35s,box-shadow .35s,border-color .35s;}
.sg-use::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--a),var(--c));
  transform:scaleX(0);transform-origin:left;transition:.45s;}
.sg-use:hover::after{transform:scaleX(1);}
.sg-use:hover{transform:translateY(-6px) scale(1.01);
  border-color:rgba(245,158,11,.4);
  box-shadow:0 14px 40px rgba(245,158,11,.1);}
.sg-use .ui{font-size:2.5rem;margin-bottom:.75rem;display:inline-block;
  transition:transform .4s;}
.sg-use:hover .ui{transform:scale(1.25) rotate(5deg);}
.sg-use .ut{font-weight:700;font-size:1rem;color:#fff;margin-bottom:.4rem;}
.sg-use .ub{color:#64748b;font-size:.85rem;line-height:1.75;}

/* PIPELINE STEPS */
.sg-step{display:flex;gap:1rem;align-items:flex-start;background:var(--card);
  border:1px solid var(--bdr);border-radius:16px;padding:1.8rem;
  position:relative;overflow:hidden;transition:all .35s;}
.sg-step::before{content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
  background:linear-gradient(180deg,var(--c),var(--p));transform:scaleY(0);
  transform-origin:top;transition:transform .5s cubic-bezier(.16,1,.3,1);}
.sg-step:hover::before{transform:scaleY(1);}
.sg-step:hover{border-color:rgba(0,245,255,.35);transform:translateX(5px);
  box-shadow:0 8px 30px rgba(0,245,255,.1);}
.sn{font-family:'Orbitron',monospace;font-size:.85rem;font-weight:900;
  min-width:42px;height:42px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;position:relative;
  background:rgba(0,245,255,.1);color:var(--c);border:1px solid rgba(0,245,255,.3);}
.sn::after{content:'';position:absolute;inset:-3px;border-radius:50%;
  border:1px solid rgba(0,245,255,.2);animation:ring-pulse 2s ease infinite;}
@keyframes ring-pulse{0%,100%{transform:scale(1);opacity:.5}50%{transform:scale(1.15);opacity:0}}
.sn.sp{background:rgba(124,58,237,.12);color:#a78bfa;border-color:rgba(124,58,237,.3);}
.sn.sp::after{border-color:rgba(124,58,237,.2);}
.sn.sa{background:rgba(245,158,11,.1);color:var(--a);border-color:rgba(245,158,11,.3);}
.sn.sa::after{border-color:rgba(245,158,11,.2);}
.sg-step .ct{font-weight:700;color:#fff;margin-bottom:.4rem;}
.sg-step .cb{color:#64748b;font-size:.875rem;line-height:1.72;}

/* PROS/CONS */
.pc-grid{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-top:2rem;}
@media(max-width:800px){.pc-grid{grid-template-columns:1fr;}}
.pro-box,.con-box{border-radius:16px;padding:2rem;position:relative;overflow:hidden;}
.pro-box{background:rgba(16,185,129,.05);border:1px solid rgba(16,185,129,.22);}
.con-box{background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.22);}
.bh{font-family:'Orbitron',monospace;font-size:.85rem;font-weight:700;
  letter-spacing:2px;text-transform:uppercase;margin-bottom:1.4rem;
  display:flex;align-items:center;gap:.6rem;}
.bh.pro{color:var(--g)}.bh.con{color:var(--r)}
.pc-item{display:flex;gap:.8rem;align-items:flex-start;margin-bottom:.9rem;
  font-size:.875rem;line-height:1.68;opacity:0;transform:translateX(-12px);
  transition:opacity .5s,transform .5s;}
.pc-item.iv{opacity:1;transform:translateX(0);}
.con-box .pc-item{transform:translateX(12px);}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;margin-top:.4rem;}
.dot.pro-d{background:var(--g);box-shadow:0 0 8px var(--g);}
.dot.con-d{background:var(--r);box-shadow:0 0 8px var(--r);animation:dot-pulse 1.5s ease infinite;}
@keyframes dot-pulse{0%,100%{box-shadow:0 0 4px var(--r)}50%{box-shadow:0 0 12px var(--r)}}

/* ARCH IMAGE */
.arch-frame{border-radius:20px;overflow:hidden;margin-top:2.5rem;position:relative;
  border:1px solid var(--bdr);box-shadow:0 0 60px rgba(0,245,255,.07),
  0 0 120px rgba(124,58,237,.05);}
.arch-frame::before{content:'';position:absolute;inset:0;background:
  linear-gradient(135deg,rgba(0,245,255,.04) 0%,transparent 50%,rgba(124,58,237,.04) 100%);
  z-index:1;pointer-events:none;}
.arch-frame img{width:100%;display:block;}

/* PILLS */
.pill{display:inline-block;background:rgba(0,245,255,.07);border:1px solid rgba(0,245,255,.18);
  border-radius:50px;padding:.3rem .9rem;font-size:.72rem;font-weight:600;
  color:var(--c);margin:.2rem;letter-spacing:.5px;transition:all .25s;}
.pill:hover{background:rgba(0,245,255,.16);box-shadow:0 0 12px rgba(0,245,255,.2);transform:translateY(-2px);}
.pill-p{background:rgba(124,58,237,.1);border-color:rgba(124,58,237,.25);color:#a78bfa;}
.pill-p:hover{background:rgba(124,58,237,.2);box-shadow:0 0 12px rgba(124,58,237,.25);}
.pill-a{background:rgba(245,158,11,.08);border-color:rgba(245,158,11,.25);color:var(--a);}
.pill-a:hover{background:rgba(245,158,11,.18);box-shadow:0 0 12px rgba(245,158,11,.2);}

/* GPU BADGE */
.gpu-badge{display:inline-flex;align-items:center;gap:.4rem;
  background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);
  border-radius:50px;padding:.25rem .75rem;font-size:.68rem;font-weight:700;
  color:var(--g);letter-spacing:1px;animation:gpu-blink 2.5s ease infinite;}
@keyframes gpu-blink{0%,100%{box-shadow:0 0 0 0 rgba(16,185,129,.3)}50%{box-shadow:0 0 0 5px rgba(16,185,129,0)}}

/* FOOTER */
.sg-foot{text-align:center;padding:6rem 2rem;
  background:radial-gradient(ellipse 70% 60% at 50% 100%,rgba(124,58,237,.18) 0%,transparent 70%);}
.sg-sep{height:1px;margin:0 4rem 4rem;
  background:linear-gradient(90deg,transparent,var(--c),var(--p),transparent);
  box-shadow:0 0 8px var(--c);animation:sep-glow 3s ease infinite;}
@keyframes sep-glow{0%,100%{opacity:.6}50%{opacity:1}}
.sg-foot-copy{color:rgba(148,163,184,.3);font-size:.75rem;margin-top:3rem;letter-spacing:1.5px;}

/* TOOLTIP */
.tooltip-wrap{position:relative;}
.tooltip-wrap .tip{position:absolute;bottom:calc(100% + 8px);left:50%;transform:translateX(-50%);
  background:#0f172a;border:1px solid var(--c);border-radius:10px;padding:.75rem 1rem;
  min-width:200px;font-size:.78rem;line-height:1.5;color:#e2e8f0;
  z-index:999;pointer-events:none;opacity:0;transition:opacity .2s;white-space:nowrap;}
.tooltip-wrap:hover .tip{opacity:1;}
.tip-title{color:var(--c);font-weight:700;margin-bottom:.4rem;}

/* REVEAL */
.rv{opacity:0;transform:translateY(40px);
  transition:opacity .8s cubic-bezier(.16,1,.3,1),transform .8s cubic-bezier(.16,1,.3,1);}
.rv.iv{opacity:1;transform:translateY(0);}
.rv-l{opacity:0;transform:translateX(-50px);
  transition:opacity .75s cubic-bezier(.16,1,.3,1),transform .75s cubic-bezier(.16,1,.3,1);}
.rv-r{opacity:0;transform:translateX(50px);
  transition:opacity .75s cubic-bezier(.16,1,.3,1),transform .75s cubic-bezier(.16,1,.3,1);}
.rv-l.iv,.rv-r.iv{opacity:1;transform:translateX(0);}
.rv-s{opacity:0;transform:scale(.88);
  transition:opacity .6s cubic-bezier(.34,1.56,.64,1),transform .6s cubic-bezier(.34,1.56,.64,1);}
.rv-s.iv{opacity:1;transform:scale(1);}

/* GLOW FOLLOW EFFECT (Apple/AI-startup style) */
.glow-card{position:relative;overflow:hidden;box-shadow:0 0 30px rgba(0,245,255,0.08);}
.glow-card::after{content:"";position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(circle at var(--gx,50%) var(--gy,50%),rgba(0,245,255,0.18),transparent 42%);
  opacity:0;transition:opacity 0.2s ease;border-radius:inherit;z-index:2;}
.glow-card:hover::after{opacity:1;}
</style>
"""

# JS_SCRIPT: inject via st.components.v1.html() - uses window.parent to reach host page
JS_SCRIPT = """
<script>
(function(){
const D=window.parent.document;
/* create canvas */
function mkEl(tag,id,styles){if(D.getElementById(id))return;const el=D.createElement(tag);el.id=id;Object.assign(el.style,styles);D.body.prepend(el);}
mkEl('canvas','sg-canvas',{position:'fixed',top:0,left:0,width:'100%',height:'100%',zIndex:0,pointerEvents:'none',opacity:'.28'});
mkEl('div','sg-bar',{position:'fixed',top:0,left:0,height:'2px',width:'0',zIndex:99999,pointerEvents:'none',background:'linear-gradient(90deg,#00f5ff,#7c3aed,#f59e0b)',boxShadow:'0 0 8px #00f5ff',transition:'width .08s linear'});
/* canvas animation */
const cv=D.getElementById('sg-canvas');
if(cv&&!cv._r){cv._r=1;const ctx=cv.getContext('2d');let W,H,pts=[];
  function resize(){W=cv.width=window.parent.innerWidth;H=cv.height=window.parent.innerHeight;}
  resize();window.parent.addEventListener('resize',resize,{passive:true});
  for(let i=0;i<65;i++)pts.push({x:Math.random()*W,y:Math.random()*H,vx:(Math.random()-.5)*.25,vy:(Math.random()-.5)*.25,r:Math.random()*1.2+.4});
  function draw(){ctx.clearRect(0,0,W,H);pts.forEach(p=>{p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='rgba(0,245,255,.6)';ctx.fill();});
    for(let i=0;i<pts.length;i++)for(let j=i+1;j<pts.length;j++){const d=Math.hypot(pts[i].x-pts[j].x,pts[i].y-pts[j].y);if(d<120){ctx.beginPath();ctx.moveTo(pts[i].x,pts[i].y);ctx.lineTo(pts[j].x,pts[j].y);ctx.strokeStyle='rgba(0,245,255,'+((.45*(1-d/120)).toFixed(2))+')';ctx.lineWidth=.6;ctx.stroke();}}
    requestAnimationFrame(draw);}draw();}
/* progress bar */
const bar=D.getElementById('sg-bar');
if(bar&&!bar._b){bar._b=1;window.parent.addEventListener('scroll',()=>{const t=D.documentElement.scrollHeight-window.parent.innerHeight;if(t>0)bar.style.width=(window.parent.scrollY/t*100)+'%';},{passive:true});}
/* IntersectionObserver */
const io=new window.parent.IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('iv');io.unobserve(e.target);}});},{threshold:.08,rootMargin:'0px 0px -40px 0px'});
/* counters */
const ctrs=[{s:'.cnt-3',e:3,u:''},{s:'.cnt-30',e:30,u:''},{s:'.cnt-11',e:11,u:''},{s:'.cnt-94',e:94,u:'%'},{s:'.cnt-21',e:21,u:'K+'}];
function aC(el,end,suf){let s=0;const st=ts=>{if(!s)s=ts;const p=Math.min((ts-s)/1400,1);el.textContent=Math.round((1-Math.pow(1-p,3))*end)+suf;if(p<1)requestAnimationFrame(st);};requestAnimationFrame(st);}
const cio=new window.parent.IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){ctrs.forEach(t=>{const el=e.target.querySelector(t.s);if(el)aC(el,t.e,t.u);});cio.unobserve(e.target);}});},{threshold:.4});
/* typing */
function typing(){const el=D.querySelector('.sg-type-target');if(!el||el.dataset.t)return;el.dataset.t=1;const txt=el.dataset.text||el.innerText.trim();el.textContent='';const cur=D.createElement('span');cur.className='cursor';el.appendChild(cur);let i=0;const iv=setInterval(()=>{if(i<txt.length)cur.insertAdjacentText('beforebegin',txt[i++]);else clearInterval(iv);},26);}
/* nav */
function activeNav(){const sy=window.parent.scrollY;D.querySelectorAll('.sg-links a').forEach(a=>{const sec=D.querySelector(a.getAttribute('href'));if(!sec)return;const top=sec.offsetTop-140;a.classList.toggle('active',sy>=top&&sy<top+sec.offsetHeight);}); }
window.parent.addEventListener('scroll',activeNav,{passive:true});
function smoothNav(){D.querySelectorAll('.sg-links a[href^="#"]').forEach(a=>{if(a.dataset.sn)return;a.dataset.sn=1;a.addEventListener('click',e=>{const t=D.querySelector(a.getAttribute('href'));if(t){e.preventDefault();t.scrollIntoView({behavior:'smooth',block:'start'});}});});}
/* decorate & observe */
function decorate(){
  D.querySelectorAll('.sg-tag,.sg-desc').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv');io.observe(el);}});
  D.querySelectorAll('.sg-title').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;io.observe(el);}});
  D.querySelectorAll('.sg-card,.sg-use').forEach((el,i)=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv');el.style.transitionDelay=(i%6*.1)+'s';io.observe(el);}});
  D.querySelectorAll('.sg-step').forEach((el,i)=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv');el.style.transitionDelay=(i*.12)+'s';io.observe(el);}});
  D.querySelectorAll('.pro-box').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv-l');io.observe(el);}});
  D.querySelectorAll('.con-box').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv-r');io.observe(el);}});
  D.querySelectorAll('.pc-item').forEach((el,i)=>{if(!el.dataset.d){el.dataset.d=1;el.style.transitionDelay=(i*.07)+'s';io.observe(el);}});
  D.querySelectorAll('.sg-stat').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv-s');io.observe(el);}});
  D.querySelectorAll('.sg-stats').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;cio.observe(el);}});
  D.querySelectorAll('.arch-frame').forEach(el=>{if(!el.dataset.d){el.dataset.d=1;el.classList.add('rv');io.observe(el);}});
  /* glow-card: add class + mousemove listener */
  D.querySelectorAll('.sg-card,.sg-step,.sg-use').forEach(el=>{
    if(!el.dataset.glow){el.dataset.glow=1;el.classList.add('glow-card');
      el.addEventListener('mousemove',e=>{
        const r=el.getBoundingClientRect();
        el.style.setProperty('--gx',(e.clientX-r.left)+'px');
        el.style.setProperty('--gy',(e.clientY-r.top)+'px');
      },{passive:true});
    }
  });
  typing();smoothNav();activeNav();}
/* hero entrance */
function heroIn(){['.sg-badge-pill','.sg-h1','.sg-type','.sg-stats'].forEach((s,i)=>{D.querySelectorAll(s).forEach(el=>{el.style.opacity=0;el.style.transform='translateY(28px)';el.style.transition='opacity .75s cubic-bezier(.16,1,.3,1),transform .75s cubic-bezier(.16,1,.3,1)';setTimeout(()=>{el.style.opacity=1;el.style.transform='none';},180+i*190);});});}
let tid;new MutationObserver(()=>{clearTimeout(tid);tid=setTimeout(decorate,380);}).observe(D.body,{childList:true,subtree:true});
if(D.readyState==='complete'){decorate();heroIn();}else window.parent.addEventListener('load',()=>{decorate();heroIn();});
setTimeout(()=>{decorate();heroIn();},900);
})();
</script>
"""

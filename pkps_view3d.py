"""
pkps_view3d.py — интерактивный 3D-просмотр корпуса без внешних библиотек.

Модуль собирает одну самодостаточную HTML-страницу (canvas + JavaScript),
которую Streamlit показывает через st.components.v1.html. Интернет и CDN
не нужны. Вращение — перетаскиванием мышью или пальцем, масштаб — колесом
мыши или двумя пальцами.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

import pkps_hull as hull_mod
from pkps_hull import Hull3D

COLORS = {
    "under": "#b3402f",     # подводная часть
    "above": "#2e4560",     # надводный борт
    "deck": "#c9ced3",
    "transom": "#343a40",
    "house": "#e9edf0",
    "water": "#3b8fc4",
}


def _c(v: float, nd: int = 2) -> str:
    """Число с запятой: 3.5 → '3,50'."""
    return f"{v:.{nd}f}".replace(".", ",")


def make_dims(hull: Hull3D) -> list[dict]:
    """Размерные линии L, B, H, T (координаты до вертикального растяжения)."""
    L, B, H, T = hull.L, hull.B, hull.H, hull.T
    off = B / 2.0 + 0.14 * B + 2.0
    return [
        {"p": [0, -off, -0.35 * H, L, -off, -0.35 * H], "t": f"L = {_c(L, 1)} м"},
        {"p": [-0.06 * L, -off, 0, -0.06 * L, -off, H], "t": f"H = {_c(H, 2)} м"},
        {"p": [1.06 * L, -off, 0, 1.06 * L, -off, T], "t": f"T = {_c(T, 2)} м"},
        {"p": [1.12 * L, -B / 2, 0.5 * H, 1.12 * L, B / 2, 0.5 * H], "t": f"B = {_c(B, 1)} м"},
    ]


def build_scene(hull: Hull3D, wave_mode: str = hull_mod.WAVE_NONE, wave_h: float = 0.0,
                fixed_scale: bool = True, z_stretch: float = 1.0) -> dict:
    """Собирает данные сцены (вершины, группы треугольников, линии) в виде словаря.
    z_stretch — растяжение по высоте (1 — истинные пропорции)."""
    L, B = hull.L, hull.B
    if fixed_scale:
        R = max(70.0, 0.62 * L)
        ext_x, ext_y, cell = R * 1.15, max(30.0, 1.2 * B), 5.0
    else:
        R = 0.60 * L
        ext_x, ext_y = 0.75 * L, max(1.2 * B, 0.28 * L)
        cell = max(L / 32.0, 0.5)
    wv, wt = hull_mod.build_water(L, hull.T, wave_h, wave_mode, ext_x, ext_y, cell,
                                  edge=hull.wl_edge())

    zs = float(z_stretch)
    verts = np.vstack([hull.verts, wv])
    verts[:, 2] *= zs
    wline = hull.waterline.copy()
    wline[:, 2] *= zs
    off = hull.verts.shape[0]
    # s: 0 — шов того же цвета (гладкая поверхность), 1 — тёмные рёбра граней, 2 — вода
    style = {"under": 1, "above": 1, "deck": 0, "transom": 0, "house": 0}
    groups = []
    for name in ("under", "above", "deck", "transom", "house"):
        groups.append({"c": COLORS[name], "a": 1.0, "s": style[name],
                       "t": hull.groups[name].ravel().tolist()})
    groups.append({"c": COLORS["water"], "a": 0.30, "s": 2, "t": (wt + off).ravel().tolist()})

    dims = make_dims(hull)
    for d in dims:
        d["p"][2] *= zs
        d["p"][5] *= zs

    return {
        "v": np.round(verts, 3).ravel().tolist(),
        "g": groups,
        "wl": np.round(wline, 3).ravel().tolist(),
        "dims": dims,
        "ctr": [L / 2.0, 0.0, hull.H * 0.35 * zs],
        "R": R,
        "zs": zs,
    }


def info_lines(hull: Hull3D, wave_mode: str, wave_h: float, z_stretch: float = 1.0) -> list[str]:
    lines = [
        f"L {_c(hull.L, 1)} · B {_c(hull.B, 1)} · H {_c(hull.H, 2)} м",
        f"T {_c(hull.T, 2)} · Tн {_c(hull.Tn, 2)} · Tк {_c(hull.Tk, 2)} м",
        f"Cb {_c(hull.cb_target, 3)}"
        + (f" (модель {_c(hull.cb_model, 3)})" if abs(hull.cb_model - hull.cb_target) > 0.005 else "")
        + f" · дифферент {hull.trim_deg:+.2f}°".replace(".", ","),
    ]
    if abs(z_stretch - 1.0) > 1e-9:
        lines.append(f"Высота растянута ×{_c(z_stretch, 1)} (подписи — истинные размеры)")
    if wave_mode == hull_mod.WAVE_HOG:
        lines.append(f"Волна: перегиб, h = {_c(wave_h, 1)} м")
    elif wave_mode == hull_mod.WAVE_SAG:
        lines.append(f"Волна: прогиб, h = {_c(wave_h, 1)} м")
    return lines


def build_viewer_html(scene: dict, info: list[str], height: int = 520) -> str:
    """Готовая HTML-страница с просмотрщиком."""
    payload = dict(scene)
    payload["info"] = info
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", "<\\/")       # безопасно для вставки в <script>
    return (_TEMPLATE.replace("__DATA__", data).replace("__HEIGHT__", str(int(height))))


_TEMPLATE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body{margin:0;padding:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;color:#1c2a36}
  #wrap{position:relative;width:100%;height:__HEIGHT__px;border-radius:10px;overflow:hidden;
        background:linear-gradient(#dfeaf3,#f7f9fb 70%)}
  canvas{display:block;touch-action:none;cursor:grab}
  canvas:active{cursor:grabbing}
  #info{position:absolute;left:12px;top:10px;font-size:13px;line-height:1.45;
        background:rgba(255,255,255,.72);padding:6px 10px;border-radius:8px;pointer-events:none}
  #bar{position:absolute;right:10px;top:10px;display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
  #bar button{font:inherit;font-size:13px;padding:5px 10px;border:1px solid #b9c7d3;border-radius:8px;
        background:rgba(255,255,255,.85);color:#1c2a36;cursor:pointer}
  #bar button:hover{background:#fff}
  #bar button:focus-visible{outline:2px solid #1f5f8b;outline-offset:1px}
  #hint{position:absolute;left:12px;bottom:8px;font-size:12px;color:#53606b;pointer-events:none}
  @media (max-width:560px){
    #info{font-size:12px;right:10px;padding:5px 8px}
    #bar{top:auto;bottom:8px;left:10px;right:10px;justify-content:center}
    #hint{display:none}
  }
</style></head>
<body>
<div id="wrap">
  <canvas id="cv"></canvas>
  <div id="info"></div>
  <div id="bar">
    <button data-v="home">Сброс</button><button data-v="side">Сбоку</button>
    <button data-v="top">Сверху</button><button data-v="front">Спереди</button>
  </div>
  <div id="hint">Вращение — перетаскивание · масштаб — колесо или два пальца</div>
</div>
<script>
(function(){
"use strict";
const D = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d'), wrap = document.getElementById('wrap');
document.getElementById('info').innerHTML = D.info.map(s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;')).join('<br>');

// ---- геометрия ----
const nv = D.v.length/3;
const X = new Float32Array(nv), Y = new Float32Array(nv), Z = new Float32Array(nv);
for (let i=0;i<nv;i++){ X[i]=D.v[3*i]; Y[i]=D.v[3*i+1]; Z[i]=D.v[3*i+2]; }
let ntri = 0; D.g.forEach(g=>{ ntri += g.t.length/3; });
const TI = new Int32Array(ntri*3), TG = new Uint8Array(ntri);
const NX = new Float32Array(ntri), NY = new Float32Array(ntri), NZ = new Float32Array(ntri);
(function(){ let k=0;
  D.g.forEach((g,gi)=>{ for(let j=0;j<g.t.length;j+=3){
    const a=g.t[j], b=g.t[j+1], c=g.t[j+2];
    TI[3*k]=a; TI[3*k+1]=b; TI[3*k+2]=c; TG[k]=gi;
    const ux=X[b]-X[a], uy=Y[b]-Y[a], uz=Z[b]-Z[a], vx=X[c]-X[a], vy=Y[c]-Y[a], vz=Z[c]-Z[a];
    let nx=uy*vz-uz*vy, ny=uz*vx-ux*vz, nz=ux*vy-uy*vx; const l=Math.hypot(nx,ny,nz)||1;
    NX[k]=nx/l; NY[k]=ny/l; NZ[k]=nz/l; k++; } });
})();
const nseg = D.wl.length/3 - 1;
function hex(h){ return [parseInt(h.slice(1,3),16),parseInt(h.slice(3,5),16),parseInt(h.slice(5,7),16)]; }
const GC = D.g.map(g=>hex(g.c));

// ---- камера ----
const ZF = wrap.getBoundingClientRect().width < 560 ? 0.78 : 1;      // на телефоне чуть отдаляем камеру
const HOME = {yaw:-0.75, pitch:0.42, zoom:ZF};
const VIEWS = {home:HOME, side:{yaw:-Math.PI/2,pitch:0.03,zoom:ZF}, top:{yaw:-Math.PI/2,pitch:1.5,zoom:ZF},
               front:{yaw:0,pitch:0.06,zoom:ZF}};
let cam = {yaw:HOME.yaw, pitch:HOME.pitch, zoom:ZF}, spin = true, spinUntil = 0;
try { const s = sessionStorage.getItem('pkpsCam'); if (s) { const o = JSON.parse(s);
      if (isFinite(o.yaw)&&isFinite(o.pitch)&&isFinite(o.zoom)) { cam = o; spin = false; } } } catch(e){}
function saveCam(){ try { sessionStorage.setItem('pkpsCam', JSON.stringify(cam)); } catch(e){} }
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));

// ---- отрисовка ----
let W=300, Hh=__HEIGHT__, dpr=1;
const PX=new Float32Array(nv), PY=new Float32Array(nv), PZ=new Float32Array(nv);
const dep=new Float32Array(ntri+nseg), order=new Uint32Array(ntri+nseg);
function resize(){
  const r = wrap.getBoundingClientRect(); W = Math.max(260, Math.floor(r.width)); Hh = Math.floor(r.height);
  dpr = Math.min(window.devicePixelRatio||1, 2);
  cv.width = W*dpr; cv.height = Hh*dpr; cv.style.width = W+'px'; cv.style.height = Hh+'px';
  draw();
}
function draw(){
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,Hh);
  const cy=Math.cos(cam.yaw), sy=Math.sin(cam.yaw), cp=Math.cos(cam.pitch), sp=Math.sin(cam.pitch);
  const dist = D.R*2.6/cam.zoom, f = 0.50*W*dist/D.R*0.92;
  const ex=D.ctr[0]+dist*cp*cy, ey=D.ctr[1]+dist*cp*sy, ez=D.ctr[2]+dist*sp;
  let fx=D.ctr[0]-ex, fy=D.ctr[1]-ey, fz=D.ctr[2]-ez; let l=Math.hypot(fx,fy,fz); fx/=l; fy/=l; fz/=l;
  let rx=fy, ry=-fx, rz=0; l=Math.hypot(rx,ry)||1; rx/=l; ry/=l;                 // right = fwd × up(0,0,1)
  const ux=ry*fz-rz*fy, uy=rz*fx-rx*fz, uz=rx*fy-ry*fx;                           // up = right × fwd
  const cxs=W/2, cys=Hh*0.55;
  for(let i=0;i<nv;i++){
    const dx=X[i]-ex, dy=Y[i]-ey, dz=Z[i]-ez;
    const zc=dx*fx+dy*fy+dz*fz, xc=dx*rx+dy*ry+dz*rz, yc=dx*ux+dy*uy+dz*uz;
    const k=f/Math.max(zc,0.5); PX[i]=cxs+xc*k; PY[i]=cys-yc*k; PZ[i]=zc;
  }
  // свет: сверху, чуть сбоку и от камеры
  let lx=0.35*rx-0.55*fx, ly=0.35*ry-0.55*fy, lz=0.35*rz-0.55*fz+0.75; l=Math.hypot(lx,ly,lz); lx/=l; ly/=l; lz/=l;
  const N = ntri+nseg;
  for(let t=0;t<ntri;t++){ dep[t]=(PZ[TI[3*t]]+PZ[TI[3*t+1]]+PZ[TI[3*t+2]])/3; order[t]=t; }
  for(let s=0;s<nseg;s++){ const a=3*s, b=3*s+3;
    const z1=zOf(D.wl[a],D.wl[a+1],D.wl[a+2]), z2=zOf(D.wl[b],D.wl[b+1],D.wl[b+2]);
    dep[ntri+s]=(z1+z2)/2 - 0.012*D.R; order[ntri+s]=ntri+s; }
  function zOf(x,y,z){ return (x-ex)*fx+(y-ey)*fy+(z-ez)*fz; }
  order.sort((a,b)=>dep[b]-dep[a]);                                              // дальние — первыми
  ctx.lineJoin='round'; ctx.lineCap='round';
  for(let n=0;n<N;n++){
    const id=order[n];
    if(id<ntri){
      const g=D.g[TG[id]], c=GC[TG[id]];
      const sh = 0.50 + 0.50*Math.abs(NX[id]*lx+NY[id]*ly+NZ[id]*lz);
      const a=TI[3*id], b=TI[3*id+1], d=TI[3*id+2];
      ctx.beginPath(); ctx.moveTo(PX[a],PY[a]); ctx.lineTo(PX[b],PY[b]); ctx.lineTo(PX[d],PY[d]); ctx.closePath();
      const r=Math.min(255,c[0]*sh)|0, gg=Math.min(255,c[1]*sh)|0, bb=Math.min(255,c[2]*sh)|0;
      ctx.fillStyle = g.a<1 ? `rgba(${r},${gg},${bb},${g.a})` : `rgb(${r},${gg},${bb})`; ctx.fill();
      if (g.s===2){ ctx.strokeStyle='rgba(255,255,255,0.30)'; ctx.lineWidth=0.5; }
      else if (g.s===1){ ctx.strokeStyle=`rgb(${r*0.82|0},${gg*0.82|0},${bb*0.82|0})`; ctx.lineWidth=0.6; }
      else { ctx.strokeStyle=`rgb(${r},${gg},${bb})`; ctx.lineWidth=0.8; }
      ctx.stroke();
    } else {
      const s=id-ntri, a=3*s, b=3*s+3;
      const p=proj(D.wl[a],D.wl[a+1],D.wl[a+2]), q=proj(D.wl[b],D.wl[b+1],D.wl[b+2]);
      ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]);
      ctx.strokeStyle='#f4fdff'; ctx.lineWidth=2.2; ctx.stroke();
    }
  }
  function proj(x,y,z){ const dx=x-ex, dy=y-ey, dz=z-ez;
    const zc=dx*fx+dy*fy+dz*fz, xc=dx*rx+dy*ry+dz*rz, yc=dx*ux+dy*uy+dz*uz; const k=f/Math.max(zc,0.5);
    return [cxs+xc*k, cys-yc*k]; }
  // размерные линии поверх сцены
  ctx.font='600 12px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif'; ctx.textAlign='center'; ctx.textBaseline='middle';
  const placed=[];
  D.dims.forEach(d=>{
    const p=proj(d.p[0],d.p[1],d.p[2]), q=proj(d.p[3],d.p[4],d.p[5]);
    ctx.strokeStyle='#14202b'; ctx.lineWidth=1.4; ctx.beginPath(); ctx.moveTo(p[0],p[1]); ctx.lineTo(q[0],q[1]); ctx.stroke();
    [p,q].forEach(e=>{ ctx.beginPath(); ctx.arc(e[0],e[1],2.6,0,6.2832); ctx.fillStyle='#14202b'; ctx.fill(); });
    const mx=(p[0]+q[0])/2, w=ctx.measureText(d.t).width+12; let my=(p[1]+q[1])/2;
    for(const off of [0,24,-24,48,-48,72,-72]){                       // разводим подписи, чтобы не наезжали
      const y=my+off; if(!placed.some(r=>Math.abs(r.x-mx)<(r.w+w)/2 && Math.abs(r.y-y)<22)){ my=y; break; } }
    placed.push({x:mx,y:my,w:w});
    ctx.fillStyle='rgba(255,255,255,0.88)'; ctx.fillRect(mx-w/2,my-10,w,20);
    ctx.fillStyle='#14202b'; ctx.fillText(d.t,mx,my);
  });
}

// ---- управление ----
const ptr=new Map(); let lastD=0;
const pd=()=>{ const a=[...ptr.values()]; return Math.hypot(a[0].x-a[1].x,a[0].y-a[1].y); };
cv.addEventListener('pointerdown',e=>{ cv.setPointerCapture(e.pointerId); ptr.set(e.pointerId,{x:e.clientX,y:e.clientY});
  spin=false; if(ptr.size===2) lastD=pd(); });
cv.addEventListener('pointermove',e=>{ const p=ptr.get(e.pointerId); if(!p) return;
  const dx=e.clientX-p.x, dy=e.clientY-p.y; p.x=e.clientX; p.y=e.clientY;
  if(ptr.size===1){ cam.yaw-=dx*0.008; cam.pitch=clamp(cam.pitch+dy*0.006,-0.05,1.5); }
  else if(ptr.size===2){ const d=pd(); if(lastD>0) cam.zoom=clamp(cam.zoom*d/lastD,0.35,6); lastD=d; }
  draw(); });
const up=e=>{ ptr.delete(e.pointerId); lastD=0; saveCam(); };
cv.addEventListener('pointerup',up); cv.addEventListener('pointercancel',up);
cv.addEventListener('wheel',e=>{ e.preventDefault(); spin=false;
  cam.zoom=clamp(cam.zoom*Math.exp(-e.deltaY*0.0012),0.35,6); draw(); saveCam(); },{passive:false});
document.querySelectorAll('#bar button').forEach(b=>b.addEventListener('click',()=>{
  const v=VIEWS[b.dataset.v]; cam={yaw:v.yaw,pitch:v.pitch,zoom:v.zoom}; spin=false; draw(); saveCam(); }));
window.addEventListener('resize',resize);

// лёгкое вращение при первом показе (несколько секунд)
spinUntil = performance.now()+5000;
(function tick(){ if(spin && performance.now()<spinUntil && !document.hidden){ cam.yaw+=0.006; draw(); requestAnimationFrame(tick); } })();
window.__pkps = {cam:()=>cam, draw, set:(c)=>{cam=c; spin=false; draw();}};
resize();
})();
</script></body></html>
"""

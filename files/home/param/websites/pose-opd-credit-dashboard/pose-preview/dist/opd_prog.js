/* 10-29 OPD page only (loaded by the s3branches_opd.json replace at the end of the template script; ppm.tpl.html untouched).
   1. In-update progress block at the top: update N/110, the five phases in order with a bar each (done = measured seconds, current = elapsed vs
      expected + %, todo = expected), this update's elapsed + ETA, run ETA (PT). Data: opd_prog.json (opd_prog.py, every 5 s; rank-0 [phase] log
      lines, switching to run24/progress.json counts once that exists). Polled every 5 s, clock ticks every 1 s.
   2. A dashed "now" line on every chart at update (completed + fraction of the current update).
   3. ★ hover also shows param b384 at the same update.
   4. Post-train page: hides the pretraining controls (forecast, extrapolation, x-axis override, full history, epochs/answers axis). */
(()=>{
const st=document.createElement('style');
st.textContent=`#opdp{margin:0 0 12px;padding:12px 14px 10px;border:1px solid var(--line);border-radius:10px;background:var(--card)}
#opdp .l1{display:flex;flex-wrap:wrap;gap:2px 16px;align-items:baseline}
#opdp .big{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
#opdp .mid{font-size:15px;font-variant-numeric:tabular-nums}
#opdp .mut{color:var(--muted);font-size:12px}
#opdp .ph{display:flex;gap:6px;margin-top:10px}
#opdp .pc{min-width:0}
#opdp .pn{font-size:12.5px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#opdp .pc.todo .pn{color:var(--muted);font-weight:400}
#opdp .pb{height:16px;border-radius:4px;background:var(--line);overflow:hidden;margin:3px 0 2px}
#opdp .pf{height:100%;background:#40c057;transition:width .8s linear}
#opdp .pc.cur .pf{background:#fd7e14}
#opdp .pc.over .pf{background:#fa5252}
#opdp .pt{font-size:11.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-variant-numeric:tabular-nums}
#opdp .pc.cur .pt{color:var(--fg)}
#opdp .sub{margin-top:6px;font-size:12.5px;color:var(--fg)}
@media (max-width:640px){#opdp .ph{flex-direction:column;gap:4px}#opdp .pc{flex:none!important;display:grid;grid-template-columns:118px 1fr;column-gap:8px;align-items:center}#opdp .pc .pt{grid-column:2}#opdp .big{font-size:19px}}
.seg:has(#fct),.seg:has(#afull),.seg:has(#xct),label:has(#xmaxs),#xm{display:none!important}
.hdr,.bar{max-width:100%;flex-wrap:wrap}#rq{width:150px!important}
@media (max-width:640px){.ct{flex-wrap:wrap}.ct h3{flex:1 1 100%;min-width:0;overflow-wrap:anywhere}}`;
document.head.appendChild(st);
const box=document.createElement('div');box.id='opdp';box.innerHTML='<div class="mut">in-update progress loading…</div>';
const wrap=document.querySelector('.wrap');if(wrap)wrap.prepend(box);
let P=null,skew=0;
const NOW=()=>Date.now()/1000+skew;
const ft=s=>{if(s==null||!isFinite(s))return '–';s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60),x=s%60;return h?`${h}h${String(m).padStart(2,'0')}m`:`${m}:${String(x).padStart(2,'0')}`};
const TZ='America/Los_Angeles';
const pt=t=>{if(!t||!isFinite(t))return '–';const d=new Date(t*1000),day=d.toLocaleDateString('en-US',{timeZone:TZ,weekday:'short'}),td=new Date().toLocaleDateString('en-US',{timeZone:TZ,weekday:'short'});
  return (day!==td?day+' ':'')+d.toLocaleTimeString('en-GB',{timeZone:TZ,hour:'2-digit',minute:'2-digit'})+' PT'};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]);
/* live numbers from P + the clock */
function calc(){if(!P)return null;const n=P.stopped?(P.terminal_at||P.now):NOW(),ph=P.phases||[],c=P.phase,o={n,ph,c};
  o.uel=P.update_start?n-P.update_start:null;
  if(c!=null&&ph[c]){const p=ph[c];o.pel=p.start?n-p.start:null;const sb=P.sub;
    o.pfr=(sb&&sb.total>0&&sb.done!=null)?Math.min(1,sb.done/sb.total):(o.pel!=null&&p.ref>0)?Math.min(0.99,o.pel/p.ref):null;
    const rem=Math.max(0,(p.ref||0)-(o.pel||0))+ph.slice(c+1).reduce((a,q)=>a+(q.ref||0),0);o.eta=n+rem;
    o.ufr=(o.uel!=null&&o.uel+rem>0)?Math.min(0.999,o.uel/(o.uel+rem)):null}
  o.reta=o.eta&&P.per_update_for_run_eta?o.eta+Math.max(0,P.max-P.update)*P.per_update_for_run_eta:null;
  if(P.stopped||!ph.length||ph.some(p=>p.ref==null)){o.eta=null;o.reta=null;o.ufr=null;}o.nowx=P.finished?P.max:(P.completed||0)+(o.ufr||0);return o}
window.OPDNOW=()=>{if(P&&(P.idle||P.stopped))return null;const o=calc();if(!o||!P)return null;const p=o.c!=null&&o.ph[o.c];return {x:o.nowx,t:`now · update ${P.update} · ${p?p.key:''} ${o.pfr!=null?Math.round(o.pfr*100)+'%':''}`}};
function render(){if(!P)return;if(P.idle){box.innerHTML=`<div class="mut">★ ${esc(P.root||'run')}: ${P.stopped?(P.failed?'failed':'stopped'):P.starting?'initializing':'not started yet'}${P.job?' (job '+esc(P.job)+')':''} · in-update progress appears once its log or progress.json exists</div>`;return}const o=calc();   /* 10-29 nextrun: idle */if(P.finished){box.innerHTML=`<div class="l1"><span class="big">run finished · ${P.completed} / ${P.max} updates</span></div>`;return}
  const pct=((P.completed+(o.ufr||0))/P.max*100);
  let h=`<div class="l1"><span class="big">${P.stopped?(P.failed?"failed · ":"stopped · "):""}update ${P.update} of ${P.max}</span><span class="mid">${pct.toFixed(1)}% of the run</span>`+
    `<span class="mid">this update ${ft(o.uel)} elapsed · ETA ${pt(o.eta)} (in ${ft(o.eta==null?null:o.eta-o.n)})</span><span class="mid">run ETA ${pt(o.reta)}</span></div>`;
  h+='<div class="ph">'+o.ph.map((p,i)=>{let cls=p.state,fr=0,t='';
    if(p.state==='done'){fr=1;const d=(p.start&&p.end)?p.end-p.start:null;t=d!=null?`${ft(d)} done`+(p.ref?` (exp ${ft(p.ref)})`:''):'done'}
    else if(p.state==='cur'){fr=o.pfr||0;const over=o.pel!=null&&p.ref&&o.pel>p.ref&&!(P.sub&&P.sub.total);if(over)cls+=' over';
      t=`${ft(o.pel)} / exp ${ft(p.ref)}`+(o.pfr!=null?` · ${Math.round(o.pfr*100)}%`:'')+(over?' · over':'')}
    else t=`exp ${ft(p.ref)}`;
    return `<div class="pc ${cls}" style="flex:${Math.max(p.ref||0,30)} 1 0" title="${esc(p.label)}"><div class="pn">${i+1}. ${esc(p.label)}</div><div class="pb"><div class="pf" style="width:${(fr*100).toFixed(1)}%"></div></div><div class="pt">${t}</div></div>`}).join('')+'</div>';
  const sb=P.sub,cur=o.c!=null?o.ph[o.c]:null;
  h+=`<div class="sub">${cur?`<b>${esc(cur.label)}</b>`:''}${sb&&sb.text?' · '+esc(sb.text):''}${sb&&sb.total>0&&sb.done!=null?` · ${sb.done}/${sb.total} ${esc(sb.unit||'')}`:''}</div>`;
  h+=`<div class="mut" style="margin-top:3px">expected = ${esc(P.ref_src)} · source: ${P.src==='progress.json'?esc(P.root||'run24')+'/progress.json (rank 0)':'rank-0 [phase] log lines ('+esc(P.log)+'), times ±5 s'} · log written ${ft(P.log_age)} ago · run ETA = this update + ${P.max-P.update} × ${ft(P.per_update_for_run_eta)} (median of up to 10 completed updates)</div>`;
  box.innerHTML=h}
async function poll(){try{const r=await fetch('opd_prog.json?t='+Date.now(),{cache:'no-store'});if(r.ok){const t0=Date.now()/1000;P=await r.json();skew=P.now-t0;render();redraw()}}catch(e){}}
/* "now" line on every chart (x in updates) */
const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const NOWP={id:'opdnow',afterDatasetsDraw(c){try{const N=window.OPDNOW&&OPDNOW(),x=c.scales&&c.scales.x;if(!N||!x)return;const tt=((c.options.scales.x||{}).title||{}).text||'';if(!/update/i.test(tt))return;
  if(N.x<x.min||N.x>x.max)return;const px=x.getPixelForValue(N.x),a=c.chartArea,g=c.ctx;g.save();g.strokeStyle='#fd7e14';g.lineWidth=1.5;g.setLineDash([5,3]);
  g.beginPath();g.moveTo(px,a.top);g.lineTo(px,a.bottom);g.stroke();g.setLineDash([]);g.font='600 11px system-ui,sans-serif';const w=g.measureText(N.t).width;
  const tx=px+5+w>a.right?px-5-w:px+5;g.fillStyle=css('--card')||'#000';g.globalAlpha=.85;g.fillRect(tx-2,a.top+2,w+4,14);g.globalAlpha=1;g.fillStyle='#fd7e14';g.fillText(N.t,tx,a.top+13);g.restore()}catch(e){}},
  beforeInit(c){patch(c)},beforeUpdate(c){patch(c)}};
function patch(c){try{const cb=c.config.options&&c.config.options.plugins&&c.config.options.plugins.tooltip&&c.config.options.plugins.tooltip.callbacks;if(!cb||cb._opd||typeof cb.label!=='function')return;const f=cb.label;
  cb.label=function(t){let s=f.call(this,t);try{const L=String(t.dataset.label||'');if(L.startsWith('PPM main')){const pd=t.chart.data.datasets.find(d=>String(d.label||'').startsWith('param b384'));
    if(pd){const xv=t.parsed.x,q=(pd.data||[]).find(p=>p&&Math.abs(p.x-xv)<1e-6);if(q&&q.y!=null)s+=` · param b384 ${Math.abs(q.y)<1e-3&&q.y!==0?q.y.toExponential(2):(+q.y).toPrecision(4)}`}}}catch(e){}return s};cb._opd=1}catch(e){}}
function redraw(){try{Object.values(Chart.instances||{}).forEach(c=>{try{c.draw()}catch(e){}})}catch(e){}}
try{Chart.register(NOWP);Object.values(Chart.instances||{}).forEach(c=>{patch(c);try{c.update('none')}catch(e){}})}catch(e){console.warn('opdnow',e)}
poll();setInterval(poll,5000);setInterval(render,1000);setInterval(redraw,15000);
})();

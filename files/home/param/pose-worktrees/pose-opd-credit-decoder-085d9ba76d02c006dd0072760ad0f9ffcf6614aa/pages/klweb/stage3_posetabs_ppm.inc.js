/* ==== PPM COPY (10-12, stage3_ppm.html only, s3branches_ppm.json posetabs_js): POSE | SPEECH | TEXT tabs with ★ PPM + the pose run vs SAM. Original: Pose page tabs POSE | SPEECH | KL (added 10-10; hardened 10-10). Injected by build_stage3.py ONLY when s3branches_<page>.json sets "posetabs"
  (stage3_pose.html); stage3.html never contains it. INJECTION: one point only, just before the template's LAST closing script tag (after its render() and poll
  set-up), so no anchor inside the template's functions is needed. It hooks the page at runtime by WRAPPING the template's function declarations
  build / fit / fitY / slopeTxt (only for the tab charts; every other chart goes straight to the original) and by seeding the legend's default
  visibility (HID). If anything it needs is missing, or it throws, it shows a red banner "pose tabs failed to load: ..." and logs console.error.
  Data: D.s3.ptabs (pose_tabs.py, polled with s3live_<page>.json) = the ★ chain (s3branches main_run) + non-archived branches, each with its own x
  constants; D.samref (static W&B snapshot of SAM 8B speech+pose, qwen-1ch/infer/sam_pose/wandb_history.json "page_cb").
  Style: our eval = solid blue, our train = dotted blue (10-step moving average), branches = the same, lighter; SAM = dotted black (= --fg, light in
  dark mode); lines only. x: Epochs mode = epoch fraction of each run's own pose data, or clip-hours seen (toggle); Steps = each run's own steps. ==== */
(()=>{
const PT_BANNER=msg=>{try{console.error('pose tabs failed to load: '+msg);const b=document.createElement('div');b.id='pt_fail';
  b.style.cssText='background:#c0392b;color:#fff;font:600 14px/1.4 system-ui,sans-serif;padding:10px 14px;border-radius:8px;margin:8px 0';
  b.textContent='pose tabs failed to load: '+msg;const w=document.querySelector('.wrap')||document.body;w.prepend(b)}catch(e){}};
try{
  /* the template names this fragment relies on: a missing one = the template changed under us -> banner, never a silent blank */
  const NEED={sets:()=>sets,ch:()=>ch,MODE:()=>MODE,HID:()=>HID,RNG:()=>RNG,ds:()=>ds,css:()=>css,CUR:()=>CUR,MD:()=>MD,at:()=>at,TIP:()=>TIP,Y:()=>Y,LV:()=>LV,
    S:()=>S,D:()=>D,key:()=>key,yLog:()=>yLog,build:()=>build,fit:()=>fit,fitY:()=>fitY,slopeTxt:()=>slopeTxt,rngUI:()=>rngUI,zReset:()=>zReset,zSave:()=>zSave};
  const miss=Object.keys(NEED).filter(n=>{try{return NEED[n]()===undefined}catch(e){return true}});
  if(miss.length)throw new Error('template names missing: '+miss.join(', '));
  if(!(S().ptabs))throw new Error('no ptabs data in s3live (pose_tabs.py / build_stage3.py ptabs_doc failed: see the klweb build log)');
  const P=()=>S().ptabs||{runs:[],tcols:[],ecols:[],x:{}},R=D.samref&&D.samref.cb_eval?D.samref:null;
  const XD={pose_rows_per_step:64,pose_epoch_rows:699776,clip_h_per_row:114043.3/703935};
  const RX=r=>Object.assign({},XD,P().x||{},(r&&r.x)||{});      /* each run's own constants (pose_tabs.py), main's for the note */
  const MAIN=()=>(P().runs||[]).find(r=>r.id==='main')||null,X=()=>RX(MAIN());
  const SPL=(R&&R.planned)||69397,SRS=(R&&R.rows_per_step)||40;
  /* Epochs mode has two units (tab-bar toggle, saved per browser): 'ep' = epoch fraction of each run's own pose data; 'h' = clip-hours seen (same unit
     for both runs). Ours: rows x counts.json clip-hours per train row. SAM: its epoch = pose-v3 + pose-v2 (launch_pose.sh:10 POSE_DATA; W&B config data)
     ~ 115k + 245k h minus v2 segments without words ~ 356k clip-hours over its 69,397 steps (an estimate from the dataset totals: SAM logged no hours). */
  const SAM_EPOCH_H=356000;let XU='ep';try{if(localStorage.getItem('s3ppm_xu')==='h')XU='h'}catch(e){}
  const OU=r=>RX(r).pose_rows_per_step/RX(r).pose_epoch_rows,OH=r=>RX(r).pose_rows_per_step*RX(r).clip_h_per_row,SH=()=>SAM_EPOCH_H/SPL;
  const oxr=r=>s=>CUR.xm==='ep'?(XU==='h'?s*OH(r):s*OU(r)):s, sx=s=>CUR.xm==='ep'?(XU==='h'?s*SH():s/SPL):s;
  const BLUE=()=>css('--one'),LIGHT=()=>css('--one')+'73',BLK=()=>css('--fg'),DOT=[2,3];
  const ser=(rows,cols,k,xf)=>{const i=(cols||[]).indexOf(k);return i<0?[]:(rows||[]).filter(r=>r[i]!=null).map(r=>({x:xf(r[0]),y:r[i],s:r[0]}))};
  const roll=(p,w)=>{const o=[];let t=0;for(let q=0;q<p.length;q++){t+=p[q].y;if(q>=w)t-=p[q-w].y;o.push({x:p[q].x,y:t/Math.min(q+1,w),s:p[q].s})}return o};
  const L=(lab,data,c,w,dash)=>ds(lab,data,c,w,dash,{pointRadius:data.length===1?3:0,pointHoverRadius:4});   /* lines only (a lone first eval shows as one point) */
  const lastStep=r=>Math.max(0,...(r.train||[]).map(x=>x[0]),...(r.eval||[]).map(x=>x[0]));
  const runLab=r=>r.id==='main'?LV():`${r.label||r.run} (jobs ${(r.jobs||[]).join(' → ')}, steps 0–${lastStep(r)})`;
  const onTab=(r,tab)=>r.id==='main'||!r.tabs||r.tabs.includes(tab);    /* s3branches posetabs_tabs: which tabs a branch appears on (default all) */
  const mk=(c,tab)=>()=>{const o=[];for(const r of P().runs||[]){if(!onTab(r,tab))continue;const m=r.id==='main',lab=runLab(r),cc=m?BLUE():LIGHT(),nm=m?'blue':'lighter blue',xf=oxr(r);
      if(c.e){const d=ser(r.eval,P().ecols,c.e,xf);o.push(L(`${lab} · eval (solid ${nm})`+(d.length===1?' — 1 eval so far':''),d,cc,m?2.4:1.8,[]))}
      if(c.t)o.push(L(`${lab} · train, 10-step moving average (dotted ${nm})`,roll(ser(r.train,P().tcols,c.t,xf),10),cc,m?2:1.6,DOT))}
    if(R&&c.se)o.push(L('SAM 8B speech+pose · eval (dotted black; in dark mode the theme foreground, light grey)',ser(R.cb_eval,R.cb_eval_cols,c.se,sx),BLK(),2,DOT));
    if(R&&c.st)o.push(L('SAM 8B speech+pose · train, 50-step mean (dotted black, thin; hidden by default)',ser(R.cb_train,R.cb_train_cols,c.st,sx),BLK(),1.1,[1,3]));
    return o.filter(d=>(d.data||[]).length)};   /* PPM page: lines a run does not log are left out */
  const NO_SAM='No SAM reference on this chart: ',PR='on the pose (TikTok) rows';
  const NOPPM='PPM does not log this per-codebook metric yet (only cb0 and the light mean): only the pose run and SAM are shown. ';
  const TABS=[['pose','POSE',`Pose (TikTok) rows: ★ PPM and the pose run q32-pose-s3sam311b vs SAM 8B speech+pose. cbN = codebook N of 32 (cb0 = the cb0 head, cb31 = the last codebook). Lines a run does not log are left out.`,[
      ['p_pcb0','Pose cb0 CE',{e:'pose_cb0',t:'pose_cb0',se:'pose_cb0',st:'pose_cb0'},''],
      ['p_pcb1','Pose cb1 CE (first light codebook)',{e:'pose_cb1',t:'pose_cb1',se:'pose_light_cb1'},'★ PPM logs per-codebook pose CE from the D2 resume (10-12); the pose run logged cb8 / cb16 / cb31 only.'],
      ['p_plight','Pose light CE (mean over pose cb1–31)',{e:'pose_light',t:'pose_light'},'No SAM line: SAM logged single pose codebooks only (cb1 / cb8 / cb16 / cb31), not the light mean.'],
      ['p_pcb8','Pose cb8 CE',{e:'pose_cb8',t:'pose_cb8',se:'pose_light_cb8'},'SAM logged pose cb8 in its evals only.'],
      ['p_pcb16','Pose cb16 CE',{e:'pose_cb16',t:'pose_cb16',se:'pose_light_cb16'},'SAM logged pose cb16 in its evals only.'],
      ['p_pcb31','Pose cb31 CE (last codebook)',{e:'pose_cb31',t:'pose_cb31',se:'pose_light_cb31',st:'pose_light_cb31'},''],
      ['p_pnword','Normalised word CE '+PR+' (speaker channel B)',{e:'pose_nword',t:'pose_nword'},'No SAM line: SAM is TTS (words given), it has no word CE.'],
      ['p_scb0','Speech cb0 CE '+PR,{e:'sp_cb0',t:'sp_cb0',se:'audio_cb0',st:'audio_cb0'},'Ours = channel B (the TikTok speaker); SAM = its TTS speech of the same kind of clips.'],
      ['p_slight','Speech light CE '+PR+' (mean over speech light codebooks)',{e:'sp_light',t:'sp_light'},NO_SAM+'SAM logged only the mean over speech cb1–31 in a different form.'],
      ['p_scb8','Speech cb8 CE '+PR,{e:'sp_cb8'},NO_SAM+'SAM logged only the mean over speech cb1–31.'],
      ['p_scb16','Speech cb16 CE '+PR,{e:'sp_cb16'},NO_SAM+'SAM logged only the mean over speech cb1–31.'],
      ['p_scb31','Speech cb31 CE '+PR+' (last codebook)',{e:'sp_cb31'},NO_SAM+'SAM logged only the mean over speech cb1–31.']]],
    ['speech','SPEECH',`Speech rows: ★ PPM podcast speech rows (its own holdout, salt ppm-v1) and the pose run's conversation rows (tt-s3 eval list). Different eval rows, so read the shapes, not small gaps. No SAM reference here: SAM never trained on conversation rows. The Demeter comparison is in the template chart below ("Speech word CE, normalised: ★ run vs Demeter").`,[
      ['s_nword','Normalised word CE, speech / conversation rows',{e:'conv_nword',t:'conv_nword'},''],
      ['s_cb0','Speech cb0 CE, speech / conversation rows',{e:'conv_cb0',t:'conv_cb0'},''],
      ['s_light','Speech light CE (mean), speech / conversation rows',{e:'conv_light',t:'conv_light'},''],
      ['s_cb8','Speech cb8 CE, conversation rows',{e:'conv_cb8'},'Eval only.'],
      ['s_cb16','Speech cb16 CE, conversation rows',{e:'conv_cb16'},'Eval only.'],
      ['s_cb31','Speech cb31 CE, conversation rows (last codebook)',{e:'conv_cb31'},'Eval only.']]],
    ['text','TEXT',`PPM trains text with plain CE (no teacher KL), so this tab shows normalised word CE on its text micros. No SAM or pose-run reference (the pose run trained text with KL); the Demeter text-duplex comparison is in the PPM chart "Gemma text duplex" further down.`,[
      ['t_gm','Gemma text duplex: normalised word CE',{e:'gm_nword',t:'gm_nword'},'★ PPM eval = its own Gemma text-duplex eval packs (txtdup-full eval split, 2 per rank). Gemma and chat alternate by step, so each training curve has points on its own steps only.'],
      ['t_ch','Chat: normalised word CE',{e:'ch_nword',t:'ch_nword'},'★ PPM eval = chat split-2 holdout packs, 2 per rank.']]]];
  const PTK=new Set(TABS.flatMap(t=>t[3].map(c=>c[0])));
  const st=document.createElement('style');st.textContent='.ptabs{display:grid;gap:12px}.ptab{display:grid;gap:16px}.ptab[hidden]{display:none}.ptabbar button{font-weight:600;letter-spacing:.04em;padding:6px 16px}';document.head.appendChild(st);
  const f=v=>(+v).toLocaleString('en-US');
  const fb=(P().runs||[]).filter(r=>r.x&&(!r.x.derived||!r.x.clip_h_derived)).map(r=>`${runLab(r)}: ${!r.x.derived?'pose rows per step / per epoch not derivable ('+(r.x.why||'unknown')+'), using 64 rows / step and 699,776 rows / epoch':''}${!r.x.clip_h_derived?' clip-hours per row not readable from '+(r.x.counts||'counts.json')+', using 0.162':''}`);
  const box=document.createElement('div');box.className='ptabs';box.id='ptabs';
  box.innerHTML=`<div class="ctl ptabbar" role="tablist">${TABS.map(([id,n])=>`<button role="tab" data-tab="${id}" aria-pressed="false">${n}</button>`).join('')}`+
    `<span style="margin-left:12px">x unit in Epochs mode:</span><button data-xu="ep" aria-pressed="false">Epoch fraction</button><button data-xu="h" aria-pressed="false">Clip-hours seen</button></div>`+
    (fb.length?`<p class="note" style="color:#c0392b;font-weight:600">x fallback (not derived): ${fb.join(' · ')}</p>`:'')+
    `<p class="note" id="ptnote"><b>x (Epochs mode) = epoch fraction of each run's own pose (TikTok) training data, or clip-hours seen</b> (toggle above; Steps = each run's own optimiser steps). `+
    `<b>Ours</b> (★ ${MAIN()?MAIN().run:''}) = step × ${X().pose_rows_per_step} pose rows per step (${X().mix?'mix: '+X().mix+'; ':''}rotating nodes: world × pose share of the cycle × rows per GPU; ga: world × pose micros per step × rows per GPU) ÷ ${f(X().pose_epoch_rows)} rows per pose epoch (128 GPUs × the smallest node's train rows ÷ 8, from the run's pre-flight log) → 1 epoch = ${f(Math.round(X().pose_epoch_rows/X().pose_rows_per_step))} steps; clip-hours = rows × ${(+X().clip_h_per_row).toFixed(4)} h per row = ${OH(MAIN()).toFixed(2)} clip-hours per step (its pose store's counts.json). Each run uses its own constants. `+
    `<b>SAM 8B</b> = step × ${SRS} rows (40 GPUs × 1 packed row of 7,500 positions) ÷ ${f(SRS*SPL)} rows = its single pass of ${f(SPL)} steps; it logged to step ${f((R&&R.last_step)||62136)} (W&amp;B stage2-pose-fast, static snapshot ${(R&&R.fetched)||''} UTC); clip-hours ≈ step × ${SH().toFixed(2)} (≈${f(SAM_EPOCH_H)} clip-hours per epoch, estimated from the dataset totals; SAM logged no hours). `+
    `<b>SAM's data:</b> it read <code>pose-cache-full,pose-cache-v2</code> (its launch_pose.sh:10 and W&amp;B config) = pose-v3 <b>plus</b> pose-v2, its 5 nodes each a disjoint fifth (W&amp;B rank0 stripe 2,400,988 segments × 8 ranks × 5 nodes = 96.0M ≈ v3 28.2M + v2 68.7M segments), dropping only segments without words or with a pose / Mimi length mismatch (pose_prep.py:36); one SAM epoch ≈ ${f(SAM_EPOCH_H)} clip-hours, words laid before the frames (TTS). `+
    `<b>Ours:</b> ${(P().runs||[]).map(r=>{const x=RX(r),eh=x.pose_epoch_rows*x.clip_h_per_row,q=SAM_EPOCH_H/eh;return `${r.id==='main'?'★ '+r.run:runLab(r)} = store ${x.dataset||'?'} (${x.counts||''}: ${x.store_rows?f(x.store_rows)+' train rows = '+f(Math.round(x.store_clip_h))+' clip-hours':'counts not readable'}); one epoch ≈ ${f(Math.round(eh))} clip-hours, so a SAM epoch holds ${q>0.9&&q<1.1?'about the same amount (×'+q.toFixed(2)+')':'≈'+q.toFixed(1)+'× that'}`}).join('; ')}. `+
    `${(P().runs||[]).some(r=>RX(r).dataset==='pose-v3')?'The pose-v3 store keeps English pose-v3 only and drops clips under 3 s or without words (1% of videos held out); at equal epoch fraction SAM has seen ≈3.1× its clip-hours, and per step that store gives ≈2× SAM\'s clip-hours. ':''}Use “Clip-hours seen” to compare per amount of data. `+
    `Same pose tokenizer (pose-tokenizer-32cb-200k-noshift @1d6ff4f) and codebook numbering. Different eval sets (ours 128 held-out pose rows, SAM 40 packed rows), and our light CE uses 25% of eval frames (10% in training) where SAM uses all frames. SAM curves = its eval (dotted black); its training curve is in each legend, hidden by default.</p>`+
    TABS.map(([id,,note,charts])=>`<div class="ptab" data-panel="${id}" hidden><p class="note">${note}</p>`+charts.map(([k,t,,n])=>`<section><h2>${t}</h2><div class="cv"><canvas id="${k}"></canvas></div>${n?`<p class="note">${n}</p>`:''}</section>`).join('')+'</div>').join('');
  const anchor=document.getElementById('runs');if(anchor&&anchor.after)anchor.after(box);else (document.querySelector('.wrap')||document.body).prepend(box);
  const ourMax=()=>{let m=0;for(const r of P().runs||[]){if(r.id!=='main')continue;/* PPM copy: Fit to ★ run = the ★ PPM run only (the pose run would push PPM off the fitted y-range) */const xf=oxr(r);for(const x of [...(r.train||[]),...(r.eval||[])])m=Math.max(m,xf(x[0]))}return m};
  const unitU=()=>XU==='h'?OH(MAIN()):OU(MAIN());
  const xtitle=()=>CUR.xm==='ep'?(XU==='h'?'clip-hours seen (each run\'s own pose data)':'epoch fraction of each run\'s own pose data'):'optimiser steps (each run\'s own)';
  /* ---- runtime hooks (tab charts only; all other charts call the template's originals unchanged) ---- */
  const seedHid=k=>{let ls=[];try{ls=sets[k]().map(d=>d.label)}catch(e){return}const H=HID[k]||(HID[k]={});
    for(const l of ls){const kk=key(l);if(H[kk]!==undefined)continue;if(/^SAM 8B/.test(l))H[kk]=/· train/.test(l);else if(!l.startsWith('★'))H[kk]=false}};   /* legend defaults: SAM eval + branches shown, SAM train hidden */
  const ptFitY=(c,k)=>{const y=c.options.scales.y,xm=c.options.scales.x.max,xl=c.options.scales.x.min;delete y.min;delete y.max;if(xm==null)return;const x0=xl>0?xl:0.2*ourMax();let lo=Infinity,hi=-Infinity;
    for(const d of c.data.datasets){if(d.hidden)continue;for(const p of d.data)if(p&&p.x<=xm&&p.x>=x0&&p.y!=null&&isFinite(p.y)){lo=Math.min(lo,p.y);hi=Math.max(hi,p.y)}}
    if(!isFinite(lo))return;const lg=yLog(k);if(lg&&lo<=0)lo=hi/1e3;if(lg){y.min=lo/1.25;y.max=hi*1.25}else{const pad=(hi-lo)*0.08||Math.abs(hi)*0.05||1;y.min=lo-pad;y.max=hi+pad}};
  const _build=build,_fit=fit,_fitY=fitY,_slope=slopeTxt;
  fitY=function(c,k){return PTK.has(k)?ptFitY(c,k):_fitY(c,k)};
  build=function(k){if(PTK.has(k)){at(k);seedHid(k)}const r=_build(k);if(PTK.has(k)&&ch[k]){at(k);ch[k].options.scales.x.title.text=xtitle();ch[k].update('none')}return r};
  fit=function(){const r=_fit.apply(this,arguments);for(const k of PTK){const c=ch[k];if(!c)continue;const m=at(k),u=(RNG[k]||{})[m.xm]||{},rr=RNG[k]||{};
      let zoomed=false;try{zoomed=!!(c.isZoomedOrPanned&&c.isZoomedOrPanned())}catch(e){}
      if(zoomed)continue;const sx_=c.options.scales.x;if(m.fit&&u.max==null){const v=ourMax();sx_.max=v>0?v*1.5:undefined}
      if(rr.ymin==null&&rr.ymax==null)ptFitY(c,k);sx_.title=Object.assign({},sx_.title||{},{display:true,text:xtitle()});c.update('none')}return r};
  slopeTxt=function(k,d,x){if(!PTK.has(k))return _slope(k,d,x);const Q0=(d.data||[]).filter(p=>p&&p.y!=null&&isFinite(p.y)&&isFinite(p.x));if(Q0.length<3)return '';
    const ep=MD(k).xm==='ep',u=ep?unitU():1,W=100*u,un=ep?(XU==='h'?'steps of ★ data (in clip-hours)':'★ steps of pose data'):'steps';
    const avg=(a,b)=>{const q=Q0.filter(p=>p.x>a&&p.x<=b);return q.length?[q.reduce((s,p)=>s+p.y,0)/q.length,q.length]:null};const now=avg(x-W,x+1e-12),prev=avg(x-2*W,x-W);
    if(now&&prev&&now[1]>=2&&prev[1]>=2){const per=now[0]-prev[0];return `  |  slope (100-step moving average): ${per>0?'+':''}${per.toPrecision(3)} per 100 ${un}`}
    const Q=Q0.map(p=>[Math.abs(p.x-x),p]).sort((a,b)=>a[0]-b[0]).slice(0,5).map(a=>a[1]);const n=Q.length,mx=Q.reduce((a,p)=>a+p.x,0)/n,my=Q.reduce((a,p)=>a+p.y,0)/n,sxx=Q.reduce((a,p)=>a+(p.x-mx)**2,0);
    if(!(sxx>0))return '';const b=Q.reduce((a,p)=>a+(p.x-mx)*(p.y-my),0)/sxx,per=b*100*u;return `  |  slope (sparse curve, fit over nearest ${n} pts): ${per>0?'+':''}${per.toPrecision(3)} per 100 ${un}`};
  /* ---- register the charts (the template's own render() already ran; poll()'s render() rebuilds them with the rest from now on) ---- */
  for(const [tab,,,charts] of TABS)for(const [k,,c] of charts){sets[k]=mk(c,tab);if(!MODE[k])MODE[k]={xm:'ep',fit:true};
    TIP[k]=t=>{const p=t.raw||{},ep=MD(k).xm==='ep';return `${(+t.parsed.y).toPrecision(4)}`+(p.s!=null?` · step ${p.s}`:'')+(ep?(XU==='h'?` · ${Math.round(+t.parsed.x).toLocaleString('en-US')} clip-hours`:` · ${(+t.parsed.x).toFixed(4)} pose epoch`):'')}}
  Object.assign(Y,{k_kl:{type:'logarithmic',title:{display:true,text:'KL (nats, log)'}},k_klne:{type:'logarithmic',title:{display:true,text:'KL (nats, log)'}},
    k_eos:{type:'logarithmic',title:{display:true,text:'KL (nats, log)'}},k_top1:{title:{display:true,text:'top-1 agreement'}}});
  for(const k of PTK)rngUI(k);
  for(const k of PTK)build(k);
  const setXU=u=>{XU=u;try{localStorage.setItem('s3ppm_xu',u)}catch(e){}box.querySelectorAll('button[data-xu]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.xu===u)));
    for(const k of PTK)if(ch[k]){zReset(ch[k]);build(k)}fit()};
  box.querySelectorAll('button[data-xu]').forEach(b=>{b.onclick=()=>setXU(b.dataset.xu);b.setAttribute('aria-pressed',String(b.dataset.xu===XU))});
  let cur='pose';try{const v=localStorage.getItem('s3ppm_tab');if(TABS.some(x=>x[0]===v))cur=v}catch(e){}
  const show=id=>{cur=id;try{localStorage.setItem('s3ppm_tab',id)}catch(e){}
    box.querySelectorAll('.ptabbar button[data-tab]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tab===id)));
    box.querySelectorAll('.ptab').forEach(p=>p.hidden=p.dataset.panel!==id);
    for(const [tid,,,charts] of TABS)if(tid===id)for(const [k] of charts)if(ch[k])ch[k].resize();
    fit()};
  box.querySelectorAll('.ptabbar button[data-tab]').forEach(b=>b.onclick=()=>show(b.dataset.tab));
  show(cur);
  window.PT_OK=true;
}catch(e){PT_BANNER(e&&e.message||String(e))}
})();
/* ==== end pose page tabs ==== */

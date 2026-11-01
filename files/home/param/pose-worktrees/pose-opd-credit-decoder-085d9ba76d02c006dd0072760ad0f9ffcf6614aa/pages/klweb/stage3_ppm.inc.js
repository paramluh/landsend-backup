/* ==== PPM page charts (added 10-12). Injected by build_stage3.py ONLY when s3branches_<page>.json sets "inject_js": "stage3_ppm.inc.js" (stage3_ppm.html);
  no other page contains it. Same mechanism as the pose tabs: its own script after the template's last script, adds its charts to the template's
  `sets` (so the template's render() rebuilds them on every 60 s poll), wires each with the template's rngUI (Epochs / Steps / Fit / zoom / move / slope)
  and the template legend (click to show / hide, saved per browser). Default visible = the ★ run + the references (labels starting "reference",
  "Demeter on our eval rows", "Demeter (Hestia)"); reference training curves and per-step lines are hidden until clicked.
  Data: D.s3 from ppm_doc.py (cols / train_cols = PPM columns sp_* speech, po_* pose rows, gm_* Gemma text duplex, ch_* chat, pose_cb0 / pose_light =
  pose codes); D.s3.refs = stage-3 reference runs (s3_doc.py schema, own epoch_rows; kind "pose" also has pose_x); D.dmour / D.dmtxd / D.hestia = Demeter. ==== */
(()=>{
const BAN=msg=>{try{console.error('PPM charts failed to load: '+msg);const b=document.createElement('div');b.id='pt_fail';
  b.style.cssText='background:#c0392b;color:#fff;font:600 14px/1.4 system-ui,sans-serif;padding:10px 14px;border-radius:8px;margin:8px 0';
  b.textContent='PPM charts failed to load: '+msg;(document.querySelector('.wrap')||document.body).prepend(b)}catch(e){}};
try{
  const NEED={sets:()=>sets,ch:()=>ch,ds:()=>ds,css:()=>css,CUR:()=>CUR,S:()=>S,D:()=>D,LV:()=>LV,build:()=>build,fit:()=>fit,rngUI:()=>rngUI,lx:()=>lx,EPR:()=>EPR,
    HE:()=>HE,HC:()=>HC,hx:()=>hx,Y:()=>Y,TIP:()=>TIP,BR:()=>BR,brc:()=>brc};
  const miss=Object.keys(NEED).filter(n=>{try{return NEED[n]()===undefined}catch(e){return true}});
  if(miss.length)throw new Error('template names missing: '+miss.join(', '));
  window.PT_OK=true;
  const PP=()=>S().ppm||{},REFS=()=>S().refs||[],REF=id=>REFS().find(r=>r.id===id);
  const C=(cols,k)=>(cols||[]).indexOf(k);
  const EVD=[6,4],DOTS=[2,3];
  const roll=(p,w)=>{const o=[];let t=0;for(let q=0;q<p.length;q++){t+=p[q].y;if(q>=w)t-=p[q-w].y;o.push({x:p[q].x,y:t/Math.min(q+1,w)})}return o};
  /* x of the ★ run: speech-like charts = template lx (speech rows / speech epoch); pose charts = pose rows / pose epoch */
  const pxS=s=>CUR.xm==='ep'?s*(PP().pose_rows_per_step||128)/(PP().pose_epoch_rows||2206848):s;
  const serS=(rows,cols,k,xf)=>{const i=C(cols,k);return i<0?[]:(rows||[]).filter(r=>r[i]!=null).map(r=>({x:xf(r),y:r[i]}))};
  const lxr=r=>lx(r),pxr=r=>pxS(r[0]);
  const star=(k,xf,lab,c,w=10)=>{const o=[];const e=serS(S().pts,S().cols,k,xf);
    o.push(ds(`${LV()} · ${lab} · eval (dashed)`+(e.length===1?' — 1 eval so far':''),e,c,2.4,EVD,{pointRadius:e.length<16?3.5:0,pointHoverRadius:5}));
    const t=serS(S().train,S().train_cols,k,xf);
    o.push(ds(`${LV()} · ${lab} · train (${w}-step moving average, solid)`,roll(t,w),c,1.7,[],{pointRadius:0}));
    o.push(ds(`${LV()} · ${lab} · train (per step)`,t,c+'40',1,[],{pointRadius:0}));return o};
  /* PPM branches (s3branches_ppm.json "branches" with "doc" = ppm_doc.py): same PPM columns */
  const brs=(k,xf,lab)=>BR().flatMap(b=>{const e=serS(b.pts,b.cols,k,xf),t=serS(b.train,b.train_cols,k,xf);return [ds(`${b.label} · ${lab} · eval (dashed)`,e,brc(b),2.2,EVD,{pointRadius:e.length<16?3:0}),ds(`${b.label} · ${lab} · train (10-step moving average, solid)`,roll(t,10),brc(b),1.5,[],{pointRadius:0})]});
  /* stage-3 references: x = their own epoch (rows / their epoch_rows) or their steps; pose rows = their pose epoch */
  const rx=R=>r=>CUR.xm==='ep'?(r[1]||0)/(R.epoch_rows||5183000):r[0];
  const rpx=R=>r=>{const x=R.pose_x;return CUR.xm==='ep'?(x?r[0]*x.pose_rows_per_step/x.pose_epoch_rows:r[0]*64/699776):r[0]};
  const rc=R=>(document.documentElement.dataset.theme==='dark'||(!document.documentElement.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches))?(R.color_dark||R.color):R.color;
  const mean2=(rows,cols,a,b,xf)=>{const ia=C(cols,a),ib=C(cols,b);if(ia<0||ib<0)return[];return (rows||[]).filter(r=>r[ia]!=null&&r[ib]!=null).map(r=>({x:xf(r),y:(r[ia]+r[ib])/2}))};
  const refPair=(R,a,b,lab,xf,single)=>{if(!R)return[];const e=single?serS(R.pts,R.cols,a,xf):mean2(R.pts,R.cols,a,b,xf),t=single?serS(R.train,R.train_cols,a,xf):mean2(R.train,R.train_cols,a,b,xf);
    return [ds(`reference · ${R.label} · ${lab} · eval (dashed; its own eval rows)`,e,rc(R),2,EVD,{pointRadius:0,pointHoverRadius:4}),
            ds(`ref. train · ${R.label} · ${lab} (10-step moving average, solid)`,roll(t,10),rc(R),1.3,[],{pointRadius:0})]};
  const DMO=D.dmour,DMT=D.dmtxd,DC=k=>C((DMO||{}).cols,k),TC=k=>C((DMT||{}).cols,k);
  const dmx=r=>CUR.xm==='ep'?r[1]:r[0];
  const dmSpeech=()=>DMO&&DMO.pts?[ds('Demeter on our eval rows · speech, mean of both channels (stage-3 eval rows, not PPM\'s)',
      DMO.pts.filter(r=>r[DC('A_nword')]!=null&&r[DC('B_nword')]!=null).map(r=>({x:dmx(r),y:(r[DC('A_nword')]+r[DC('B_nword')])/2})),css('--dm'),2.2,[12,5],{pointRadius:3,pointHoverRadius:5})]:[];
  const HR=()=>HE.runs.find(r=>r.visible)||HE.runs[0];
  const hMean=(a,b,lab)=>{const R=HR();if(!R)return[];return [ds(`Demeter (Hestia) · its own logged eval (its own 80 rows) · ${lab}, mean of both channels`,R.eval.filter(e=>e[HC(a)]!=null&&e[HC(b)]!=null).map(e=>({x:hx(e),y:(e[HC(a)]+e[HC(b)])/2})),css('--ref'),1.6,[16,6],{pointRadius:0,pointHoverRadius:4})]};
  const dmTxd=()=>DMT&&DMT.pts?[ds('Demeter on our eval rows · Gemma text duplex, pooled (stage-3 text-duplex eval packs, not PPM\'s)',
      DMT.pts.filter(r=>r[TC('mean')]!=null).map(r=>({x:dmx(r),y:r[TC('mean')]})),css('--dm'),2.2,[12,5],{pointRadius:3,pointHoverRadius:5})]:[];
  const BL=()=>css('--one'),A=()=>css('--chA'),B=()=>css('--chB');
  /* ★ PPM on the stage-3 eval sets (eval only, from the D2 resume): like-for-like with the stage-3 run and Demeter-on-our-rows */
  const starS3=(a,b,lab,pooled)=>{const e=pooled?serS(S().pts,S().cols,pooled,lxr):mean2(S().pts,S().cols,a,b,lxr);
    return e.length?[ds(`${LV()} · ${lab} · eval (dotted)`+(e.length===1?' — 1 eval so far':''),e,BL(),2.4,DOTS,{pointRadius:e.length<16?4:0,pointStyle:'rectRot',pointHoverRadius:6})]:[]};
  /* eval-only ★ series (per-codebook light CE, per-channel speech word CE; logged from the D2 resume) */
  const CBC=['#e8590c','#7048e8','#1098ad','#5c940d'];
  const starE=(k,xf,lab,c,dash)=>{const e=serS(S().pts,S().cols,k,xf);return e.length?[ds(`${LV()} · ${lab} · eval`,e,c,1.6,dash||EVD,{pointRadius:e.length<16?2.5:0,pointHoverRadius:4})]:[]};
  const cbLines=(pre,cbs,xf,lab)=>cbs.flatMap((c,i)=>starE(`${pre}${c}`,xf,`${lab} cb${c}`,CBC[i%CBC.length],[4,3]));
  const D2N=()=>{const d=PP().d2;return d?` From step ${d.from.toLocaleString('en-US')} (the D2 resume) every 's3eval-v1' episode is held out of PPM training; ${d.seen_holdout_rows.toLocaleString('en-US')} of its rows had been trained on before that step.`:''};
  const ER=k=>(PP().eval_rows||{})[k]||'';
  const CH=[
   ['q_sp_nword','Speech (podcast rows): normalised word CE — ★ PPM vs stage 3 vs Demeter',()=>[...star('sp_nword',lxr,'speech',BL()),...starS3('s3a_nword','s3b_nword','speech on the stage-3 eval rows (Demeter\'s 103), mean of lanes A, B'),...starE('spa_nword',lxr,'speech, PPM eval rows, lane A (sides flipped)',A(),DOTS),...starE('spb_nword',lxr,'speech, PPM eval rows, lane B',B(),DOTS),...brs('sp_nword',lxr,'speech'),
      ...refPair(REF('s3duplex'),'A_nword','B_nword','speech, mean of both channels',rx(REF('s3duplex')||{})),...refPair(REF('pose'),'A_nword','B_nword','conversation rows, mean of both channels',rx(REF('pose')||{})),
      ...dmSpeech(),...hMean('A_nword','B_nword','speech')],
    ()=>`<b>Like-for-like line:</b> ★ PPM dotted (◆) = ${ER('s3')}; shown as the mean of lanes A and B, as Demeter-on-our-rows and the stage-3 run are.${D2N()} <b>The dashed ★ PPM eval is not like-for-like:</b> ${ER('sp')}; pooled over all word targets. Stage-3 duplex reference and Demeter-on-our-rows: the stage-3 eval rows (salt s3eval-v1, 103 rows at Demeter's eval) — the same rows as each other, so those two compare directly. Pose-run reference: its tt-s3 eval list (644 episodes, 4 rows per rank). Demeter (Hestia) logged: its own 80 rows. The PPM eval episodes and the stage-3 eval episodes are chosen by different salts, so some stage-3 eval episodes may be in PPM's training data. x = each run's own epoch (PPM: speech rows / ${(PP().speech_epoch_rows||0).toLocaleString('en-US')} FULL speech rows; stage 3: rows / 5,183,000; Demeter: its own) or steps.`],
   ['q_gm_nword','Gemma text duplex: normalised word CE — ★ PPM vs Demeter',()=>[...star('gm_nword',lxr,'Gemma text duplex',BL()),...starS3(null,null,'Gemma text duplex on the stage-3 eval packs (gemma-txtdup-v1 0..151), pooled','g3_nword'),...brs('gm_nword',lxr,'Gemma text duplex'),...dmTxd(),...hMean('td_A_nword','td_B_nword','text-duplex rows')],
    ()=>`<b>Like-for-like line:</b> ★ PPM dotted (◆) = ${ER('g3')}; pooled over both lanes, as Demeter-on-our-rows is. <b>The dashed ★ PPM eval is not like-for-like:</b> ${ER('gm')}; pooled over both channels. Demeter on our eval rows = the stage-3 text-duplex eval (gemma-txtdup-v1 packs 0..151), Demeter's exact code, pooled. Demeter (Hestia) logged = its own 80 text-duplex rows. x for ★ = its speech epoch (the run's own epoch).`],
   ['q_sp_cb','Speech codes (podcast rows): cb0 CE and light CE',()=>{const r=REF('s3duplex'),p=REF('pose');return [...star('sp_cb0',lxr,'speech cb0',B()),...star('sp_light',lxr,'speech light',A()),...brs('sp_cb0',lxr,'speech cb0'),...brs('sp_light',lxr,'speech light'),...cbLines('sp_light_cb',[8,16,31],lxr,'speech light'),
      ...refPair(r,'A_cb0','B_cb0','speech cb0, mean of both channels',rx(r||{})),...refPair(r,'A_light','B_light','speech light, mean of both channels',rx(r||{})),
      ...refPair(p,'A_cb0','B_cb0','conversation cb0, mean of both channels',rx(p||{})),...refPair(p,'A_light','B_light','conversation light, mean of both channels',rx(p||{}))]},
    ()=>`cb0 = voice tower codebook 0; light = mean over the light codebooks (PPM evaluates light on up to 2,048 positions). ★ eval rows: ${ER('sp')}. References on their own eval rows (see the speech chart). Demeter logged no audio-code CE.`],
   ['q_time','Step time and its per-micro breakdown',()=>{const t=serS(S().train,S().train_cols,'step_s',r=>r[0]);const v=t.map(p=>p.y).sort((a,b)=>a-b),md=v[v.length>>1]||0;const keep=p=>!md||p.y<5*md;const xf=r=>CUR.xm==='ep'?lx(r):r[0];
      const L=(k,lab,c,w,dash)=>{const d=serS(S().train,S().train_cols,k,xf).filter(keep);return ds(`${LV()} · ${lab}`,d,c,w,dash,{pointRadius:0})};
      return [L('step_s','s / step (total)',BL(),2,[]),L('micro_pose','pose micro (s)',B(),1.4,[]),L('micro_speech','speech micros (s, both)',A(),1.4,[]),L('micro_text','text micro (s)','#2e9e5b',1.4,[]),L('wait_s','data wait (s)',css('--muted'),1.1,DOTS)]},
    ()=>{const s=S(),m=s.mean50_s,left=(s.max_steps&&m&&!s.done)?Math.max(0,(s.max_steps-s.step)*m):null;const eta=left!=null?new Date(Date.now()+left*1000).toLocaleString('en-US',{timeZone:'America/Los_Angeles',weekday:'short',month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})+' PT':'–';
      return `Seconds per optimiser step (GA ${(PP().ga||[]).join(', ')}): total and each micro; eval steps (over 5× the median) dropped. Mean of the last 50 steps ${m||'–'} s → ${s.max_steps?(s.max_steps-s.step).toLocaleString('en-US'):'?'} steps left ≈ ${left!=null?(left/3600).toFixed(1)+' h':'–'}, <b>ETA ${eta}</b> (step ${s.max_steps?s.max_steps.toLocaleString('en-US'):'?'}, at the current pace, evals every 30 min included only through the mean).`}]];
  Y.q_time={min:0,title:{display:true,text:'s / step'}};
  /* DOM: one section per chart, inserted before the template's first chart section */
  const wrap=document.querySelector('.wrap')||document.body,first=wrap.querySelector('section');
  const st=document.createElement('style');st.textContent='.ppmtiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:6px 0 4px}';document.head.appendChild(st);
  const tl=document.createElement('div');tl.className='strip ppmtiles';tl.id='ppm_tiles';(first?first.before(tl):wrap.appendChild(tl));
  for(const [k,title] of CH){const s=document.createElement('section');s.innerHTML=`<h2>${title}</h2><div class="cv"><canvas id="${k}"></canvas></div><p class="note" id="n_${k}"></p>`;first?first.before(s):wrap.appendChild(s)}
  for(const [k,,f] of CH)sets[k]=f;
  const notes=()=>{for(const [k,,,n] of CH){const e=document.getElementById('n_'+k);if(e)try{e.innerHTML=n()}catch(x){}}};
  const f4=v=>v==null?'–':(+v).toFixed(3);
  const tiles=()=>{const s=S(),last=k=>{const i=C(s.cols,k),p=(s.pts||[]).filter(r=>r[i]!=null);return i<0||!p.length?null:p[p.length-1]};
    tl.innerHTML=[['sp_nword','Speech · normalised word CE'],['po_nword','Pose rows · normalised word CE'],['gm_nword','Gemma text duplex · normalised word CE'],['s3a_nword','Speech · stage-3 rows · lane A'],['s3b_nword','Speech · stage-3 rows · lane B'],['g3_nword','Gemma · stage-3 packs (pooled)'],['ch_nword','Chat · normalised word CE'],['pose_cb0','Pose cb0 CE']].map(([k,n])=>{const r=last(k);return `<div class="si"><span class="k">${n} · latest eval</span><span class="v">${r?f4(r[C(s.cols,k)]):'–'}</span><span class="d">${r?'step '+r[0]:'no eval yet'}</span></div>`}).join('')};
  /* ---- every chart on this page (template, tabs, PPM): hover shows EVERY visible series' value at the hovered x (nearest point per series within
     30 px), each line named, sorted by value; drag-zoom / double-click reset / move / slope / y-range come from the template's own build() + rngUI(). ---- */
  const CJ=window.Chart;
  if(CJ&&CJ.Interaction&&CJ.Interaction.modes&&!CJ.Interaction.modes.ppmx){CJ.Interaction.modes.ppmx=function(chart,e){const x=e.x,items=[];
    chart.data.datasets.forEach((d,di)=>{if(!chart.isDatasetVisible(di))return;const meta=chart.getDatasetMeta(di);let best=-1,bd=Infinity;
      (meta.data||[]).forEach((el,i)=>{if(!el||el.skip||!isFinite(el.x))return;const dx=Math.abs(el.x-x);if(dx<bd){bd=dx;best=i}});
      if(best>=0&&bd<=30)items.push({element:meta.data[best],datasetIndex:di,index:best})});return items}}
  const short=l=>{l=String(l||'').replace(/^★ PPM · [^·]*· /,'★ ');return l.length>80?l.slice(0,78)+'…':l};
  const patch=k=>{const c=ch[k];if(!c||c.$ppmx)return;c.$ppmx=1;c.options.interaction={mode:'ppmx',intersect:false};const tt=c.options.plugins.tooltip;
    tt.mode='ppmx';tt.intersect=false;const cb=tt.callbacks||(tt.callbacks={});const lab=cb.label;cb.label=t=>short(t.dataset.label)+': '+(lab?lab(t):(+t.parsed.y).toPrecision(4));
    tt.itemSort=(a,b)=>a.parsed.y-b.parsed.y;c.update('none')};
  const _pb=build;build=function(k){const r=_pb.apply(this,arguments);try{patch(k)}catch(e){console.error(e)}return r};
  window.PPM_PATCH=()=>{for(const k in ch)try{patch(k)}catch(e){}};
  for(const [k] of CH)rngUI(k);
  const up=()=>{try{notes();tiles()}catch(e){console.error(e)}};
  for(const [k] of CH){try{build(k)}catch(e){console.error(k,e)}}fit();up();window.PPM_PATCH();
  /* refresh notes / tiles after each template poll (the template re-renders the charts itself through sets) */
  setInterval(up,15000);
}catch(e){BAN((e&&e.message)||String(e))}
})();

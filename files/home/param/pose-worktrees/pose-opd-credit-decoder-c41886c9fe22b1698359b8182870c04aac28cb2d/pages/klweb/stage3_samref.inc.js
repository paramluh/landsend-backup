/* ==== SAM 8B speech+pose reference (added 10-10). Injected by build_stage3.py ONLY into pages whose s3branches_<page>.json has "samref"
  (= stage3_pose.html); stage3.html never contains it. Data: D.samref, static, from qwen-1ch/infer/sam_pose/wandb_history.json
  (W&B canopy-labs/sam-tts/stage2-pose-fast, snapshot by qwen-1ch/infer/sam_pose/wbpull/snap.py); never polled.
  Adds faded series to c_posecb (pose cb0 / light) and c_cb0 / c_light (speech): its eval (dots), its train 50-step mean, and a flat line at
  its eval at the export step. x = SAM's own optimiser steps (epoch mode: its step / its planned steps). Labels start 'SAM 8B' (shown by default). ==== */
(()=>{const R=D.samref;if(!R||!R.eval)return;
  const SL='SAM 8B speech+pose',PL=R.planned||69397,XS=R.export_step||61258;
  const sx=s=>CUR.xm==='ep'?s/PL:s;
  const ser=(cols,rows,k)=>{const i=(cols||[]).indexOf(k);return i<0?[]:(rows||[]).filter(r=>r[i]!=null).map(r=>({x:sx(r[0]),y:r[i]}))};
  const atX=k=>{const i=(R.eval_cols||[]).indexOf(k);let b=null;if(i<0)return null;for(const r of R.eval)if(r[i]!=null&&(!b||Math.abs(r[0]-XS)<Math.abs(b[0]-XS)))b=r;return b?[b[0],b[i]]:null};
  const C=()=>css('--ref')+'66';   /* faded: the reference colour at 40% */
  const mk=(k,n)=>{const o=[ds(`${SL} · ${n} · eval, its own eval set (dots, its own steps)`,ser(R.eval_cols,R.eval,k),C(),1.3,[3,3],{pointRadius:2.2,pointHoverRadius:4}),
      ds(`${SL} · ${n} · train (50-step mean, its own steps)`,ser(R.train_cols,R.train,k),C(),1.1,[],{pointRadius:0})];
    const e=atX(k);if(e)o.push(ds(`${SL} · ${n} · its eval at export step ${XS.toLocaleString('en-US')} = ${(+e[1]).toFixed(3)} (flat level line)`,[{x:0,y:e[1]},{x:Math.max(XMAX(),sx(R.last_step||XS)),y:e[1]}],C(),1.2,[12,5],{pointRadius:0,pointHoverRadius:0}));
    return o};
  const ADD={c_posecb:[['pose_cb0','pose cb0'],['pose_light','pose light (cb1–31 mean)']],c_cb0:[['audio_cb0','speech cb0']],c_light:[['audio_light','speech light (cb1–31 mean)']]};
  for(const k of Object.keys(ADD)){const f=sets[k];if(!f)continue;sets[k]=()=>[...f(),...ADD[k].flatMap(([c,n])=>mk(c,n))]}
  const f=v=>v==null?'–':(+v).toLocaleString('en-US');
  for(const k of Object.keys(ADD)){const cv=document.getElementById(k),sec=cv&&cv.closest?cv.closest('section'):null;if(!sec)continue;const p=document.createElement('p');p.className='note';
    p.innerHTML=`<b>Faded grey = reference: SAM 8B speech+pose (W&amp;B stage2-pose-fast): different data, eval set and batch size — absolute levels only roughly comparable.</b> `+
      `Its x is its own optimiser steps (40 GPUs × 1 row of 7,500 positions per step; Epochs mode = its step / its planned ${f(PL)} steps). It logged to step ${f(R.last_step)} (run ${R.state||''}); `+
      `the flat line is its eval at the exported checkpoint, step ${f(XS)} (HF sam-speech-pose-stage2-fast). Under “Fit to ★ run” only its first steps show; turn Fit off to see its whole curve. `+
      (k==='c_posecb'?'':'Its speech is single-speaker TTS on its pose clips, while this chart is the ★ run\'s per-channel duplex eval. ')+
      `Static snapshot, fetched ${R.fetched||'–'} UTC.`;sec.appendChild(p)}
})();
/* ==== end SAM 8B reference ==== */

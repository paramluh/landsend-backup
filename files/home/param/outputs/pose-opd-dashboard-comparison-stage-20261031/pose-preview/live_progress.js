/* Observed pose telemetry only: no inferred stages or clock-based completion. */
(()=>{
 const box=document.createElement('section');box.id='opdp';box.style.cssText='padding:14px;margin:0 0 12px;border:1px solid var(--line);border-radius:10px;background:var(--card)';
 document.querySelector('.wrap').prepend(box);
 let data=null;
 const el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n};
 const duration=s=>s==null?'—':Math.floor(Math.max(0,s)/60)+'m '+Math.floor(Math.max(0,s)%60)+'s';
 function render(){
  if(!data)return;box.replaceChildren();
  box.append(el('strong',data.failed?'Training attempt failed · process stopped':data.finished?'Run complete':data.idle?'Waiting for run telemetry':'Live rollout tracking · rank 0'));
  box.append(el('p',`${data.completed} completed updates${data.max?' / '+data.max:''} · ${data.state||'pending'}`));
  if(data.max){const total=el('progress','');total.max=data.max;total.value=data.completed;total.style.width='100%';total.setAttribute('aria-label','Completed training updates');box.append(total);}
  if(data.epochs)box.append(el('p',`${data.epochs} epochs · ${data.max??'—'} planned updates`));
  if(data.learning_rate_peak!=null){const fmt=v=>v==null?'awaiting first update':Number(v).toExponential(2);box.append(el('p',`Learning rate: ${fmt(data.learning_rate_current)} applied · ${fmt(data.learning_rate_peak)} peak`));}
  const phase=data.phase,progress=data.rollout_progress;
  if(phase){const name=progress?'Student rollout':phase.name.replaceAll('_',' ');box.append(el('p',(data.failed?'Stopped during: ':'Current recorded phase: ')+name+(phase.update!=null?' · update '+phase.update:'')));}
  if(progress&&progress.total_frames>0){
   const label=el('p',`Actor batch position ${progress.current_frame} / ${progress.total_frames} submitted · ${(100*progress.current_frame/progress.total_frames).toFixed(1)}% · includes prefill and media-delay slots · rank 0 pair`);
   const bar=el('progress','');bar.max=progress.total_frames;bar.value=progress.current_frame;bar.style.width='100%';bar.setAttribute('aria-label','Submitted rollout frames');box.append(label,bar);
  }
  const note=el('p',data.observed_at?'Last event '+duration(Date.now()/1000-data.observed_at)+' ago · observed counts only':'No events recorded yet');note.style.cssText='font-size:12px;color:var(--muted)';box.append(note);
  const mirror=data.collector;if(mirror){const age=mirror.last_success?duration(Date.now()/1000-mirror.last_success)+' ago':'never';box.append(el('p',`Telemetry mirror: ${mirror.status||'unavailable'} · last successful refresh ${age}${mirror.error?' · '+mirror.error:''}`));}
 }
 async function poll(){try{const r=await fetch('opd_prog.json?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);data=await r.json();render()}catch(e){box.replaceChildren(el('p','Progress feed unavailable; retrying.'))}}
 poll();setInterval(poll,5000);setInterval(render,1000);
})();

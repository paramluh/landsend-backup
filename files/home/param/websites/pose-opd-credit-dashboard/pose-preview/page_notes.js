(()=>{
  const basic=S().kl_metadata?.objective==='basic_opd';
  const notes={critical:'KL(student ‖ teacher), in nats. Tower = codebook 0. Depth = arithmetic mean across codebooks 1–31. Lexical KL is conditional on ordinary word tokens; category KL measures word mass versus special-token categories.',
    gradnorms:'Parameter charts show exact gradients before clipping. Separate direct/credit and KL-origin parameter norms appear only on measured updates, as unsmoothed points. The categorical-logit charts below are inexpensive logit derivatives, not parameter gradients. Norms do not add.',
    extras:'Per-codebook KL before modality and codebook loss weights. Audio and pose each contain codebooks 0–31. Use search to filter by modality or codebook.',
    samples:'Four sampled rows per rollout update. Compare dataset input audio with generated audio and pose on a shared timeline.'};
  if(basic){notes.critical='Basic OPD: sampled forward KL from the frozen starting student for text/audio, plus monoplex CFG3 reverse KL and frozen-self forward KL for pose. Lexical contribution includes teacher WORD mass; training uses category:lexical weights 1:5. Sampled estimates may be slightly negative.';notes.gradnorms='One upper forward/backward, eight pose-light optimizer updates and one audio-light update. No causal-credit term.';notes.extras='Pose curves combine monoplex reverse KL and frozen-self forward KL; audio curves are frozen-self forward KL. Objective coefficients are removed.';}
  for(const [tab,text] of Object.entries(notes)){
    const grid=document.querySelector('.grid[data-tab="'+tab+'"]');if(!grid)continue;
    const note=document.createElement('p');note.className='pose-tab-note';note.textContent=text;grid.prepend(note);
  }
  const style=document.createElement('style');style.textContent='.pose-tab-note{grid-column:1/-1;color:var(--muted);font-size:12px;line-height:1.6;margin:0 0 2px}.pose-missing{color:var(--muted);font-size:12px;margin:6px 0}.grid[data-tab=samples]{display:block}.grid[data-tab=samples][hidden]{display:none!important}.pose-latest{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums;margin:4px 0}.pose-single{position:absolute;inset:25px 35px 60px 65px;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:12px;text-align:center;pointer-events:none;background:var(--card);color:var(--fg)}.pose-single strong{font-size:24px;font-variant-numeric:tabular-nums}.pose-single small{color:var(--muted);font-size:12px;max-width:28em}';document.head.appendChild(style);
  function measurements(){
    document.querySelectorAll('.pose-latest,.pose-single').forEach(n=>n.remove());
    const doc=S(),id=k=>'a_'+k.replace(/[^A-Za-z0-9]/g,'_');
    const entries=[...(doc.critical_charts||[]),...(doc.extra_charts||[])].map(([k])=>[id(k),[[k,'']]]);
    entries.push(['grad_combined',[['gradient_norm','']]]);
    for(const [component] of doc.grad_components||[])entries.push(['grad_'+component,[['grad_norm/combined/'+component,'Combined parameters'],['grad_norm/direct/'+component,'Direct KL parameters'],['grad_norm/credit/'+component,'Credit parameters']]]);
    for(const group of doc.parameter_loss_grad_groups||[])entries.push([group.id,group.series.map(({key,label})=>[key,label])]);
    for(const group of doc.logit_grad_groups||[])entries.push([group.id,group.series.map(({key,label})=>[key,label])]);
    for(const group of doc.conversation_groups||[])entries.push([group.id,group.series.map(({key,label})=>[key,label])]);
    for(const [key] of doc.logit_grad_diagnostics||[])entries.push([id(key),[[key,'']]]);
    const fmt=v=>Math.abs(v)>0&&Math.abs(v)<.0001?Number(v).toExponential(3):Number(v).toLocaleString('en-US',{maximumSignificantDigits:6});
    for(const [chart,series] of entries){
      const card=document.getElementById('cd_'+chart);if(!card)continue;
      const available=series.map(([key,label])=>{const i=(doc.train_cols||[]).indexOf(key),rows=i<0?[]:(doc.train||[]).filter(r=>typeof r[i]==='number'&&Number.isFinite(r[i]));return {label,i,rows}}).filter(s=>s.rows.length);
      if(!available.length)continue;
      const summary=available.map(s=>{const r=s.rows.at(-1);return (s.label?s.label+': ':'')+fmt(r[s.i])+' · update '+r[0]}).join('   |   ');
      const latest=document.createElement('p');latest.className='pose-latest';latest.textContent='Latest recorded · '+summary;card.querySelector('.ct').after(latest);
      if(available.every(s=>s.rows.length===1)){
        const single=document.createElement('div');single.className='pose-single';
        for(const s of available){const value=document.createElement('strong');value.textContent=(s.label?s.label+': ':'')+fmt(s.rows[0][s.i]);single.append(value)}
        const caption=document.createElement('small');caption.textContent='One measurement recorded. A line appears after the next recorded measurement.';single.append(caption);card.querySelector('.cv').append(single);
      }
    }
  }
  window.poseMissingNotes=()=>{
  document.querySelectorAll('.pose-missing').forEach(n=>n.remove());
  const missing=new Set((S().missing_telemetry||[]).map(x=>x.metric));
  for(const [key] of S().critical_charts||[]){
    if(!missing.has(key))continue;const card=document.getElementById('cd_a_'+key.replace(/[^A-Za-z0-9]/g,'_'));
    if(card){const p=document.createElement('p');p.className='pose-missing';p.textContent='Awaiting trainer telemetry for this metric.';card.appendChild(p)}
  }
  for(const [component] of S().grad_components||[]){
    if(!['direct','credit'].some(term=>missing.has('grad_norm/'+term+'/'+component)))continue;
    const card=document.getElementById('cd_grad_'+component);if(card){const p=document.createElement('p');p.className='pose-missing';p.textContent='No separate direct/credit parameter-gradient measurement is recorded for this run.';card.appendChild(p)}
  }
  for(const group of S().parameter_loss_grad_groups||[]){
    const card=document.getElementById('cd_'+group.id);if(!card)continue;
    const interval=S().parameter_grad_cadence?.interval_steps;
    const p=document.createElement('p');p.className='pose-missing';
    p.textContent=interval?'Exact parameter gradients measured every '+interval+' updates. Unmeasured updates are omitted; points are not smoothed or interpolated.':'No parameter-gradient diagnostic cadence is configured for this historical run; only explicitly logged measurements can appear.';
    card.appendChild(p);
  }
  for(const group of S().logit_grad_groups||[]){
    if(group.series.some(({key})=>(S().metric_definitions||{})[key]?.source==='logged'))continue;
    const card=document.getElementById('cd_'+group.id);if(card){const p=document.createElement('p');p.className='pose-missing';p.textContent='No categorical-logit gradient measurements are recorded for this run.';card.appendChild(p)}
  }
  for(const group of S().conversation_groups||[]){
    const card=document.getElementById('cd_'+group.id);if(!card)continue;
    const p=document.createElement('p');p.className='pose-missing';
    p.textContent=group.series.some(({key})=>(S().metric_definitions||{})[key]?.source==='logged')?'Generated text only; prefill and repeated partner refreshes excluded. WORD sums all lexical probabilities. The three categories are not renormalized.':'No conversation sampling measurements are recorded for this run.';
    card.appendChild(p);
  }
  measurements();
  const countCard=document.getElementById('cd_conversation_sampled_fraction'),doc=S();
  if(countCard){
    const lines=[];
    for(const role of ['student','partner']){
      const keys=['decisions','sampled_count/pad','sampled_count/new_word','sampled_count/word'].map(k=>'conversation/'+role+'/'+k),indices=keys.map(k=>(doc.train_cols||[]).indexOf(k));
      const rows=(doc.train||[]).filter(r=>indices.every(i=>i>=0&&Number.isFinite(r[i])));if(!rows.length)continue;
      const row=rows.at(-1),values=indices.map(i=>Number(row[i]).toLocaleString('en-US'));
      lines.push(role+' · update '+row[0]+': '+values[0]+' decisions; PAD '+values[1]+', NEW_WORD '+values[2]+', WORD '+values[3]);
    }
    if(lines.length){const p=document.createElement('p');p.className='pose-latest';p.textContent=lines.join(' | ');countCard.appendChild(p)}
  }
  };window.poseMissingNotes();
})();

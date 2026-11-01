(()=>{
  const notes={critical:'KL(student ‖ teacher), in nats. Tower = codebook 0. Depth = arithmetic mean across codebooks 1–31. Lexical KL is conditional on ordinary word tokens; category KL measures word mass versus special-token categories.',
    gradnorms:'Direct KL and credit-assignment gradients, shown separately for each trainable component. Global L2 norms before clipping. The combined norm is measured independently; the two term norms do not add.',
    extras:'Per-codebook KL before modality and codebook loss weights. Audio and pose each contain codebooks 0–31. Use search to filter by modality or codebook.',
    samples:'Four sampled rows per rollout update. Compare dataset input audio with generated audio and pose on a shared timeline.'};
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
    for(const [component] of doc.grad_components||[])entries.push(['grad_'+component,[['grad_norm/direct/'+component,'Direct KL'],['grad_norm/credit/'+component,'Credit assignment']]]);
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
    const card=document.getElementById('cd_grad_'+component);if(card){const p=document.createElement('p');p.className='pose-missing';p.textContent='Separate term gradients have not been recorded yet.';card.appendChild(p)}
  }
  measurements();
  };window.poseMissingNotes();
})();

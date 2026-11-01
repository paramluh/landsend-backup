function aTabs(){
  const s=S(),id=k=>'a_'+k.replace(/[^A-Za-z0-9]/g,'_');
  const critical=(s.critical_charts||[]).map(([k,title])=>[id(k),title,()=>chA(k,0)]);
  const grads=(s.grad_components||[]).map(([component,title])=>['grad_'+component,title,()=>{
    const sets=[];
    for(const [term,label,color] of [['combined','Combined parameter gradient',css('--one')],['direct','Direct KL parameter gradient',css('--dm')],['credit','Credit parameter gradient',css('--muted')]]){
      const key='grad_norm/'+term+'/'+component;
      const raw=ser(S().train,S().train_cols,key,ppmX),points=term==='combined'?roll(raw,SM):raw;
      if(points.length){const curve=L('PPM main (summed) · '+label,points,color,{on:1,w:1,det:'Global L2 norm before clipping · '+title+' · '+label});if(term!=='combined')Object.assign(curve,{showLine:false,pointRadius:3,pointHoverRadius:5});sets.push(curve)}
    }
    return sets;
  }]);
  grads.unshift(['grad_combined','Combined gradient · global pre-clip norm',()=>chA('gradient_norm',0)]);
  for(const group of s.parameter_loss_grad_groups||[])grads.push([group.id,group.label,()=>{
    const colors=[css('--one'),css('--dm'),'#40c057','#b197fc','#fab005','#e64980'],sets=[];
    group.series.forEach(({key,label},i)=>{
      const points=ser(S().train,S().train_cols,key,ppmX);
      if(points.length)sets.push(Object.assign(L(label,points,colors[i],{on:1,w:1,det:'Exact parameter-gradient L2 norm · measured updates only · '+label}),{showLine:false,pointRadius:3,pointHoverRadius:5}));
    });
    return sets;
  }]);
  for(const group of s.logit_grad_groups||[])grads.push([group.id,group.label,()=>{
    const colors=[css('--one'),css('--dm'),css('--muted'),'#b197fc'],sets=[];
    group.series.forEach(({key,label},i)=>{
      const points=roll(ser(S().train,S().train_cols,key,ppmX),SM);
      if(points.length)sets.push(L(label,points,colors[i],{on:1,w:1,det:'Categorical-logit L2 norm · not a parameter-gradient norm · '+label}));
    });
    return sets;
  }]);
  for(const [key,title] of s.logit_grad_diagnostics||[])grads.push([id(key),title,()=>chA(key,0)]);
  const extras=(s.extra_charts||[]).map(([k,title])=>[id(k),title,()=>chA(k,0)]);
  for(const group of s.conversation_groups||[])extras.unshift([group.id,group.label,()=>{
    const colors=['#74c0fc','#63e6be','#ffd43b','#b197fc','#ff922b','#f783ac'],sets=[];
    group.series.forEach(({key,label},i)=>{
      const points=roll(ser(S().train,S().train_cols,key,ppmX),SM);
      if(points.length)sets.push(L(label,points,colors[i],{on:1,w:1,det:label+' · generated text decisions only'}));
    });
    return sets;
  }]);
  return [['critical','CRITICAL',critical],['samples','SAMPLES',[]],['gradnorms','GRAD NORMS',grads],['extras','EXTRAS',extras]];
}

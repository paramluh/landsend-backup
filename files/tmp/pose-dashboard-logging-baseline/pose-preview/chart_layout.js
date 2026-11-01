function aTabs(){
  const s=S(),id=k=>'a_'+k.replace(/[^A-Za-z0-9]/g,'_');
  const critical=(s.critical_charts||[]).map(([k,title])=>[id(k),title,()=>chA(k,0)]);
  const grads=(s.grad_components||[]).map(([component,title])=>['grad_'+component,title,()=>{
    const sets=[];
    for(const [term,label,color] of [['direct','Direct KL',css('--one')],['credit','Credit assignment',css('--dm')]]){
      const key='grad_norm/'+term+'/'+component;
      const points=roll(ser(S().train,S().train_cols,key,ppmX),SM);
      if(points.length)sets.push(L('PPM main (summed) · '+label,points,color,{on:1,w:1,det:'Global L2 norm before clipping · '+title+' · '+label}));
    }
    return sets;
  }]);
  grads.unshift(['grad_combined','Combined gradient · global pre-clip norm',()=>chA('gradient_norm',0)]);
  const extras=(s.extra_charts||[]).map(([k,title])=>[id(k),title,()=>chA(k,0)]);
  return [['critical','CRITICAL',critical],['samples','SAMPLES',[]],['gradnorms','GRAD NORMS',grads],['extras','EXTRAS',extras]];
}

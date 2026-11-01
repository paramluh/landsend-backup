// Ignore the old Full history setting: the new comparison opens on Current run.
let AFULL=false,POSE_COMPARE=LS.get('comparison_range_v1','current')==='all'?'all':'current';
function poseObservedEnd(run){
  // max_steps is a plan, not evidence of completed training.
  const steps=[Number(run.step),...(run.train||[]).map(row=>Number(row[0]))].filter(Number.isFinite);
  return Math.max(0,...steps);
}
function poseCompareEnd(){
  const current=S(),runs=POSE_COMPARE==='all'?[current,...(current.branches||[])]:[current];
  return Math.max(1,...runs.map(run=>(run===current?ppmX:aXf(run))([poseObservedEnd(run)])));
}
function poseCompareControl(){
  return `<label>Chart range <select id="pose-comparison-range" aria-label="Chart range" title="Zoom every chart to completed updates in the current run, or in all runs. Manual x limits override this range."><option value="current"${POSE_COMPARE==='current'?' selected':''}>Current run</option><option value="all"${POSE_COMPARE==='all'?' selected':''}>All runs</option></select></label>`;
}
function poseCompareBind(){
  const control=document.getElementById('pose-comparison-range');
  if(!control)return;
  control.onchange=()=>{
    POSE_COMPARE=control.value==='all'?'all':'current';LS.set('comparison_range_v1',POSE_COMPARE);hdr();
    for(const key in ch){try{ch[key].resetZoom('none')}catch(e){}ch[key].$z=false;upd(key)}
  };
}

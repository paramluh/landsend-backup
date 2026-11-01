(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const PREFIX = 'pose_opd_credit:';
  const DATA_URL = 's3live_pose_opd_credit.json';
  const VERSION_URL = 'ver_s3_pose_opd_credit.txt';
  const TIMEZONE = 'America/Los_Angeles';
  const stored = (key, fallback) => { try { return localStorage.getItem(PREFIX + key) ?? fallback; } catch { return fallback; } };
  const persist = (key, value) => { try { localStorage.setItem(PREFIX + key, value); } catch {} };
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = value => finite(value) ? value.toLocaleString('en-US', {maximumFractionDigits: 0}) : '—';
  const valueText = value => !finite(value) ? '—' : value === 0 ? '0' : Math.abs(value) < .001 || Math.abs(value) >= 1e6 ? value.toExponential(2) : value.toLocaleString('en-US', {maximumSignificantDigits: 5});
  const humanize = value => String(value || '').replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
  const stamp = value => {
    if (finite(value)) return value;
    if (typeof value === 'string') { const ms = Date.parse(value); if (Number.isFinite(ms)) return ms / 1000; }
    return null;
  };
  const clock = (value, full = false) => {
    const ts = stamp(value); if (ts === null) return '—';
    return new Intl.DateTimeFormat('en-US', {timeZone: TIMEZONE, ...(full ? {month:'short',day:'numeric'} : {}), hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false}).format(new Date(ts * 1000)) + ' PT';
  };
  const age = value => {
    const ts = stamp(value); if (ts === null) return '';
    const seconds = Math.max(0, Math.floor(Date.now()/1000-ts));
    return seconds < 60 ? `${seconds}s ago` : seconds < 3600 ? `${Math.floor(seconds/60)}m ago` : seconds < 86400 ? `${Math.floor(seconds/3600)}h ${Math.floor(seconds%3600/60)}m ago` : `${Math.floor(seconds/86400)}d ago`;
  };
  const duration = value => !finite(value) ? '—' : value < 120 ? `${value.toFixed(1)} s` : `${(value/60).toFixed(1)} min`;
  const element = (tag, className, text) => { const node = document.createElement(tag); if(className) node.className=className; if(text !== undefined) node.textContent=text; return node; };
  const svgElement = (tag, attrs) => { const node = document.createElementNS(NS,tag); for(const [key,val] of Object.entries(attrs||{})) node.setAttribute(key,String(val)); return node; };
  const unpack = data => data && typeof data === 'object' && !Array.isArray(data) ? (data.s3 || data) : {};
  let doc = {};
  try { doc = unpack(JSON.parse($('pose-data').textContent)); } catch { $('poll-error').hidden=false; $('poll-error').textContent='The initial data snapshot could not be read. Refresh to try the live data.'; }
  const version = document.querySelector('meta[name="pose-version"]').content;
  const state = {tab: stored('tab','overview'), modality:stored('modality','pose'), smooth:Number(stored('smooth','1')), search:'', from:null, to:null};
  if(!['overview','codebooks','performance','events'].includes(state.tab)) state.tab='overview';
  if(!['pose','audio'].includes(state.modality)) state.modality='pose';
  if(![1,5,10,25].includes(state.smooth)) state.smooth=1;
  const overview = [
    ['loss/direct','Direct loss','--accent'],['loss/text','Text loss','--blue'],['loss/audio','Audio loss','--purple'],['loss/pose','Pose loss','--orange'],
    ['credit/absolute_sum','Absolute credit sum','--pink'],['gradient_norm','Gradient norm','--blue'],
    ['policy/logprob_mean_abs_difference','Mean absolute log-probability gap','--accent'],['policy/logprob_max_abs_difference','Maximum absolute log-probability gap','--orange'],
    ['train/step_time','Update duration · seconds','--purple']
  ];
  const profileNames = {preparation_seconds:'Data and audio preparation',frontend_restore_seconds:'Audio frontend restore',frontend_offload_seconds:'Audio frontend offload',rollout_seconds:'Rollout duration',replay_seconds:'Replay duration',optimizer_seconds:'Optimizer duration',peak_allocated_gib:'Peak allocated memory',peak_reserved_gib:'Peak reserved memory',physical_microbatch:'Physical microbatch',effective_batch_per_rank:'Effective batch per rank',rollout_batch_per_rank:'Rollout batch per rank',actor_instances:'Actor instances',global_effective_batch:'Global effective batch',replay_segments_per_forward:'Replay segments per forward'};
  let chartInstances=[];
  let busy=false;
  let resizeTimer;

  function applyTheme(theme) {
    document.documentElement.dataset.theme=theme;
    $('theme').textContent=theme==='dark'?'Light':'Dark';
    $('theme').setAttribute('aria-label',`Switch to ${theme==='dark'?'light':'dark'} theme`);
    persist('theme',theme);
  }
  applyTheme(stored('theme','dark')==='light'?'light':'dark');
  $('theme').addEventListener('click',()=>applyTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));

  function phaseObject() {
    return typeof doc.phase==='string' ? {phase:doc.phase} : (doc.phase || doc.latest_event || {});
  }
  function renderHeader() {
    const hasUpdates=finite(doc.step);
    const terminal=doc.done || /completed|bounded_or_stopped|stopped|failed|cancelled/i.test(String(doc.state||''));
    const runName=doc.label || doc.run || 'Upcoming run';
    $('run-name').textContent=runName==='pose_opd_credit'?'Pose OPD credit':runName;
    const status=String(doc.state || (hasUpdates?'Updates recorded':'Waiting for run'));
    $('status').textContent=humanize(status);
    $('status').className='status'+(terminal?' stopped':hasUpdates?' recorded':'');
    $('run-note').textContent=doc.run_dir ? String(doc.run_dir) : (doc.src ? String(doc.src) : 'The monitor is ready for the first recorded update.');
    $('stat-updates').replaceChildren(document.createTextNode(hasUpdates?number(doc.step):'—'));
    if(finite(doc.max_steps)) $('stat-updates').append(element('span','denominator',`/ ${number(doc.max_steps)}`));
    $('stat-total').textContent=finite(doc.max_steps)?`${number(doc.max_steps)} updates in the recorded plan`:'Plan not available';
    $('stat-speed').textContent=duration(doc.mean50_s ?? doc.mean_s);
    $('stat-time').textContent=clock(doc.t_last);
    $('stat-age').textContent=doc.t_last?age(doc.t_last):'No completed update recorded';
    const save=doc.last_save;
    const saveTime=Array.isArray(save)?save[0]:(save?.time ?? save?.timestamp);
    const saveStep=Array.isArray(save)?save[1]:(save?.update ?? save?.step);
    $('stat-save').textContent=finite(saveStep)?`Update ${number(saveStep)}`:saveTime?clock(saveTime):'—';
    $('stat-save-note').textContent=saveTime?`${clock(saveTime)} · ${age(saveTime)}`:finite(saveStep)?'Checkpoint recorded':'No checkpoint recorded';
    const phase=phaseObject();
    const event=doc.latest_event || phase;
    const phaseName=phase.phase || phase.event;
    $('phase').textContent=phaseName?humanize(phaseName):'Waiting for telemetry';
    const phaseDetails=[];
    if(finite(phase.update)) phaseDetails.push(`Update ${number(phase.update)}`);
    if(phase.role) phaseDetails.push(humanize(phase.role));
    if(terminal) phaseDetails.push('Recorded activity before the run ended');
    $('phase-detail').textContent=phaseDetails.join(' · ') || (phaseName?'From the trainer event log':'Activity appears when the trainer writes its event log.');
    const activityTime=event.time ?? phase.time;
    $('activity-time').textContent=activityTime?`Last event ${age(activityTime)}\n${clock(activityTime)}`:'';
    const progress=doc.rollout_progress;
    $('rollout').replaceChildren();
    $('rollout').hidden=!progress;
    if(progress) {
      $('rollout').append(element('strong','',`Latest rollout counters${finite(progress.update)?` · update ${number(progress.update)}`:''}`));
      const fields=Object.entries(progress).filter(([key,val])=>!['event','rank','time','update'].includes(key) && (finite(val)||typeof val==='string')).map(([key,val])=>`${key.replaceAll('_',' ')}: ${finite(val)?valueText(val):val}`);
      $('rollout').append(element('span','',fields.join(' · ') || 'Rollout event recorded'));
    }
    const warnings=Array.isArray(doc.warnings)?doc.warnings:[];
    $('warnings').hidden=!warnings.length;
    $('warnings').replaceChildren(...warnings.map(w=>element('p','',typeof w==='string'?w:JSON.stringify(w))));
  }

  function seriesFor(key, profile=false) {
    const cols=profile?(doc.profile_cols||[]):(doc.train_cols||[]);
    const rows=profile?(doc.profile||[]):(doc.train||[]);
    const index=cols.indexOf(key), stepIndex=Math.max(0,cols.indexOf('step'));
    if(index<0 || !Array.isArray(rows)) return [];
    return rows.filter(row=>Array.isArray(row)&&finite(row[stepIndex])).map(row=>({x:row[stepIndex],y:finite(row[index])?row[index]:null}));
  }
  function smooth(points) {
    if(state.smooth===1) return points;
    const queue=[]; let sum=0;
    return points.map(point=>{
      if(point.y===null) { queue.length=0; sum=0; return point; }
      queue.push(point.y);sum+=point.y;
      if(queue.length>state.smooth) sum-=queue.shift();
      return {x:point.x,y:sum/queue.length};
    });
  }
  function chartSpecs() {
    if(state.tab==='codebooks') {
      const prefix=`loss/${state.modality}_cb`;
      const keys=(doc.train_cols||[]).filter(key=>new RegExp(`^loss/${state.modality}_cb\\d+$`).test(key));
      const selected=keys.length?keys:Array.from({length:32},(_,i)=>prefix+i);
      return selected.sort((a,b)=>Number(a.slice(prefix.length))-Number(b.slice(prefix.length))).map(key=>[key,`${humanize(state.modality)} · codebook ${key.slice(prefix.length)}`,state.modality==='pose'?'--orange':'--purple',false]);
    }
    if(state.tab==='performance') {
      const cols=(doc.profile_cols||[]).filter(k=>!['step','update','rows','time','timestamp','rank'].includes(k));
      const defaults=['preparation_seconds','rollout_seconds','replay_seconds','optimizer_seconds','peak_allocated_gib','peak_reserved_gib'];
      return [['train/step_time','Complete update duration · seconds','--accent',false],...(cols.length?cols:defaults).map(key=>{
        const clean=key.replace(/^profile\//,'');
        return [key,(profileNames[clean]||humanize(clean))+(clean.endsWith('_seconds')?' · seconds':clean.endsWith('_gib')?' · GiB':''),clean.includes('gib')?'--purple':'--blue',true];
      })];
    }
    const standard=new Set(overview.map(s=>s[0]));
    const extra=(doc.train_cols||[]).filter(key=>!['step','rows'].includes(key)&&!standard.has(key)&&!/^loss\/(audio|pose)_cb\d+$/.test(key));
    return [...overview,...extra.map(key=>[key,humanize(key),'--pink'])];
  }

  function renderChart(chart) {
    const {svg,points,key,title,color,wrap}=chart;
    const width=Math.max(260,wrap.clientWidth||520), height=wrap.clientHeight||200;
    const pad={left:54,right:14,top:17,bottom:30}, pw=width-pad.left-pad.right, ph=height-pad.top-pad.bottom;
    const visible=points.filter(p=>(state.from===null||p.x>=state.from)&&(state.to===null||p.x<=state.to));
    const real=visible.filter(p=>p.y!==null);
    const allX=(doc.train||[]).map(row=>row[0]).filter(finite);
    let xmin=state.from ?? (allX.length?Math.min(...allX):0);
    let xmax=state.to ?? (allX.length?Math.max(...allX):finite(doc.max_steps)?doc.max_steps:1);
    if(xmax<=xmin) xmax=xmin+1;
    let ymin=real.length?Math.min(...real.map(p=>p.y)):0, ymax=real.length?Math.max(...real.map(p=>p.y)):1;
    const ypad=(ymax-ymin)*.12 || Math.abs(ymax)*.08 || .1;
    ymin-=ypad;ymax+=ypad;
    const px=x=>pad.left+(x-xmin)/(xmax-xmin)*pw, py=y=>pad.top+(ymax-y)/(ymax-ymin)*ph;
    svg.replaceChildren();svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    svg.setAttribute('aria-label',`${title}. ${real.length} recorded values. ${real.length?'Use left and right arrow keys to inspect updates.':'Waiting for data.'}`);
    for(let i=0;i<4;i++) {
      const y=pad.top+i*ph/3;
      svg.append(svgElement('line',{x1:pad.left,y1:y,x2:width-pad.right,y2:y,class:'grid-line'}));
      const label=svgElement('text',{x:pad.left-9,y:y+3,'text-anchor':'end'});label.textContent=real.length?valueText(ymax-i*(ymax-ymin)/3):'—';svg.append(label);
    }
    for(let i=0;i<5;i++) {
      const tick=xmin+i*(xmax-xmin)/4;
      const label=svgElement('text',{x:px(tick),y:height-9,'text-anchor':'middle'});label.textContent=number(tick);svg.append(label);
    }
    let path='', started=false;
    for(const point of visible) {
      if(point.y===null) { started=false;continue; }
      path+=(started?'L':'M')+px(point.x).toFixed(2)+','+py(point.y).toFixed(2)+' ';started=true;
    }
    svg.append(svgElement('path',{d:path,class:'curve',stroke:`var(${color})`}));
    const line=svgElement('line',{x1:0,y1:pad.top,x2:0,y2:height-pad.bottom,class:'hover-line',visibility:'hidden'});
    svg.append(line);
    const overlay=svgElement('rect',{x:pad.left,y:pad.top,width:pw,height:ph,class:'interaction'});svg.append(overlay);
    const empty=wrap.querySelector('.empty-chart');
    empty.hidden=real.length>1;
    empty.textContent=real.length===1?'One update recorded · curve begins with the next update':points.some(p=>p.y!==null)?'No values in this update range':'Waiting for recorded values';
    let hoverIndex=0;
    const hide=()=>{line.setAttribute('visibility','hidden');$('tooltip').hidden=true;};
    const show=(index,clientX,clientY)=>{
      const point=real[index];if(!point)return;
      hoverIndex=index;
      line.setAttribute('x1',px(point.x));line.setAttribute('x2',px(point.x));line.setAttribute('visibility','visible');
      const tooltip=$('tooltip');tooltip.textContent=`${key}: ${valueText(point.y)} @ update ${number(point.x)}${state.smooth>1?` · mean ${state.smooth}`:''}`;tooltip.hidden=false;
      const rect=svg.getBoundingClientRect();
      tooltip.style.left=Math.max(8,Math.min(clientX??(rect.left+px(point.x)),window.innerWidth-tooltip.offsetWidth-10))+'px';
      tooltip.style.top=Math.max(8,(clientY??(rect.top+pad.top))-tooltip.offsetHeight-12)+'px';
    };
    overlay.addEventListener('pointermove',event=>{
      if(!real.length)return;
      const rect=svg.getBoundingClientRect(), x=(event.clientX-rect.left)*width/rect.width;
      let closest=0;for(let i=1;i<real.length;i++)if(Math.abs(px(real[i].x)-x)<Math.abs(px(real[closest].x)-x))closest=i;
      show(closest,event.clientX+10,event.clientY);
    });
    svg.onpointerleave=hide;svg.onblur=hide;
    svg.onkeydown=event=>{
      if(!real.length||!['ArrowLeft','ArrowRight','Home','End','Escape'].includes(event.key))return;
      event.preventDefault();if(event.key==='Escape'){hide();return;}
      const next=event.key==='Home'?0:event.key==='End'?real.length-1:hoverIndex+(event.key==='ArrowRight'?1:-1);
      show(Math.max(0,Math.min(real.length-1,next)));
    };
  }

  function makeChart(spec) {
    const [key,title,color,profile]=spec;
    const raw=seriesFor(key,profile), points=smooth(raw), last=raw.filter(p=>p.y!==null).at(-1);
    const card=element('section','card'), head=element('div','card-head'), labels=element('div');
    labels.append(element('h3','',title),element('p','metric-key',key));
    const readout=element('div','');readout.append(element('div','metric-readout',last?valueText(last.y):'—'),element('div','readout-update',last?`update ${number(last.x)}`:'not yet recorded'));
    head.append(labels,readout);
    const wrap=element('div','chart-wrap'), svg=svgElement('svg',{class:'chart',role:'img',tabindex:0});
    wrap.append(svg,element('div','empty-chart','Waiting for recorded values'));
    const foot=element('div','card-footer'), legend=element('span','');
    const swatch=element('span','key-swatch');swatch.style.background=`var(${color})`;
    legend.append(swatch,document.createTextNode(profile?'rank 0 profile':'training metrics'));
    foot.append(legend,element('span','',state.smooth===1?'Updates':`Trailing mean · ${state.smooth} updates`));
    card.append(head,wrap,foot);
    const instance={svg,points,key,title,color,wrap};chartInstances.push(instance);
    return card;
  }

  function renderEvents() {
    const source=doc.recent_events || doc.events || (doc.latest_event?[doc.latest_event]:[]);
    const events=Array.isArray(source)?source:[];
    if(!events.length) {
      const empty=element('div','empty-state');empty.append(element('strong','','Waiting for events'),element('p','','Model loading, rollout, replay and checkpoint events will appear here when recorded.'));
      $('content').append(empty);return;
    }
    const panel=element('section','event-panel'), scroll=element('div','event-scroll'), table=element('table','events');
    const thead=element('thead'), tr=element('tr');
    for(const label of ['Time · Pacific','Event','Update','Details'])tr.append(element('th','',label));
    thead.append(tr);const tbody=element('tbody');
    for(const event of [...events].reverse()) {
      const row=element('tr'), time=element('td','',clock(event.time,true)), type=element('td');
      type.append(element('span','event-type',event.event||'event'));
      const detail=element('td'), disclosure=element('details');
      const fields=Object.entries(event).filter(([key])=>!['event','rank','time','update'].includes(key));
      const preview=fields.slice(0,3).map(([key,val])=>`${key}: ${typeof val==='object'?JSON.stringify(val):String(val)}`).join(' · ');
      disclosure.append(element('summary','',preview.length>160?preview.slice(0,157)+'…':preview||'View event'),element('pre','',JSON.stringify(event,null,2)));
      detail.append(disclosure);row.append(time,type,element('td','',number(event.update)),detail);tbody.append(row);
    }
    table.append(thead,tbody);scroll.append(table);panel.append(scroll);$('content').append(panel);
  }

  function renderContent() {
    $('tooltip').hidden=true;chartInstances=[];$('content').replaceChildren();
    document.querySelectorAll('[data-tab]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.tab===state.tab));
    document.querySelectorAll('[data-modality]').forEach(button=>button.setAttribute('aria-pressed',button.dataset.modality===state.modality));
    $('chart-controls').hidden=state.tab==='events';$('codebook-controls').hidden=state.tab!=='codebooks';
    const notes={overview:'Recorded losses, credit and policy agreement. Each curve retains the trainer’s original metric scale.',codebooks:'Individual codebook losses. The readout shows the latest raw value; smoothing applies only to the curve.',performance:'Profile measurements are from rank 0. Complete update duration is the trainer’s reported wall-clock time.',events:'Recent rank 0 events, newest first. These are observations from the trainer log.'};
    $('section-note').textContent=notes[state.tab];
    if(state.tab==='events') {renderEvents();return;}
    const query=state.search.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const specs=chartSpecs().filter(([key,title])=>query.every(part=>(key+' '+title).toLowerCase().includes(part)));
    $('chart-count').textContent=`${specs.length} metric${specs.length===1?'':'s'}`;
    if(!specs.length) {
      const empty=element('div','empty-state');empty.append(element('strong','','No matching metrics'),element('p','','Try another search or clear the metric filter.'));$('content').append(empty);return;
    }
    const grid=element('div','chart-grid'+(state.tab==='codebooks'?' codebooks':''));
    for(const spec of specs)grid.append(makeChart(spec));
    $('content').append(grid);chartInstances.forEach(renderChart);
  }
  function render() {renderHeader();renderContent();}
  document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>{state.tab=button.dataset.tab;persist('tab',state.tab);renderContent();}));
  document.querySelectorAll('[data-modality]').forEach(button=>button.addEventListener('click',()=>{state.modality=button.dataset.modality;persist('modality',state.modality);renderContent();}));
  $('search').addEventListener('input',event=>{state.search=event.target.value;renderContent();});
  $('smooth').value=String(state.smooth);
  $('smooth').addEventListener('change',event=>{state.smooth=Number(event.target.value);persist('smooth',String(state.smooth));renderContent();});
  function updateRange() {
    const from=$('range-from'),to=$('range-to');
    state.from=from.value!==''&&finite(from.valueAsNumber)?Math.max(0,from.valueAsNumber):null;
    state.to=to.value!==''&&finite(to.valueAsNumber)?Math.max(0,to.valueAsNumber):null;
    const invalid=state.from!==null&&state.to!==null&&state.to<state.from;
    to.setCustomValidity(invalid?'Last update must be at least the first update.':'');
    if(invalid){to.reportValidity();return;}
    renderContent();
  }
  $('range-from').addEventListener('change',updateRange);$('range-to').addEventListener('change',updateRange);
  $('reset-range').addEventListener('click',()=>{$('range-from').value='';$('range-to').value='';$('range-to').setCustomValidity('');state.from=state.to=null;renderContent();});
  window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>chartInstances.forEach(renderChart),100);});

  async function fetchText(url) {
    const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),12000);
    try {const response=await fetch(url+'?t='+Date.now(),{cache:'no-store',signal:controller.signal});if(!response.ok)throw new Error(`HTTP ${response.status}`);return await response.text();}
    finally {clearTimeout(timer);}
  }
  async function poll() {
    if(busy)return;
    if(location.protocol==='file:') {$('connection').textContent='Saved snapshot';$('footer-detail').textContent='Saved snapshot · serve this folder for live refresh · times shown in Pacific Time';return;}
    busy=true;$('refresh').disabled=true;$('connection').textContent='Refreshing…';
    try {
      const [dataResult,versionResult]=await Promise.allSettled([fetchText(DATA_URL),fetchText(VERSION_URL)]);
      if(dataResult.status==='rejected')throw dataResult.reason;
      const next=unpack(JSON.parse(dataResult.value));
      if(!Array.isArray(next.train_cols)||!Array.isArray(next.train))throw new Error('Unexpected data format');
      doc=next;render();$('poll-error').hidden=true;$('connection').classList.remove('error');$('connection').textContent='Refresh every 60s';
      if(versionResult.status==='fulfilled') {
        const nextVersion=versionResult.value.trim();
        if(/^[a-f0-9]{8,64}$/i.test(nextVersion)&&version&&nextVersion!==version)location.reload();
      }
    } catch(error) {
      $('connection').textContent='Refresh unavailable';$('connection').classList.add('error');$('poll-error').hidden=false;
      $('poll-error').textContent=`Could not refresh the run data (${error.message || 'network error'}). The last loaded snapshot remains visible; retrying in 60 seconds.`;
    } finally {busy=false;$('refresh').disabled=false;}
  }
  $('refresh').addEventListener('click',poll);
  render();poll();setInterval(poll,60000);setInterval(renderHeader,15000);
})();

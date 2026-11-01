from pathlib import Path
p=Path('/home/param/websites/pose-playground/index.html')
s=p.read_text()
def replace(old,new):
 global s
 count=s.count(old)
 if count!=1: raise RuntimeError(f'expected one anchor, got {count}: {old[:90]}')
 s=s.replace(old,new,1)
replace('<span class="chip" id="vidchip" title="10-26 user: every playground avatar (live stream + replay render) uses this video model">video: Vansh sequential sampler + Param student</span>', '<span class="chip" id="vidchip" title="Generated clips use Param avatar replay. Duplex Live mic shows a decoded pose preview alongside live audio.">video: Param replay · live pose preview</span>')
replace('const ENGINE_SKIP=new Set([\'seed\',\'n_samples\',\'clip_s\',\'prefill_person\',\'prefill_s\',\'prefill_pose\',\'prefill_audio\',\'prefill_words\',\'prefill_src\',\'voice\',\'persona\',\'voice_clip\']);', '''const ENGINE_SKIP=new Set(['seed','n_samples','clip_s','prefill_person','prefill_s','prefill_pose','prefill_audio','prefill_words','prefill_src','voice','persona','voice_clip']);
window.GENWEB_CAPS={renderers:['param'],batch_stream:false};
const RENDER_UNAVAILABLE='Unavailable: only the Param avatar renderer is hosted on this site.';
const STREAM_UNAVAILABLE='Unavailable for batch generation; use Duplex Live mic for live audio and pose preview.';
function renderAvailable(v){ const r=window.GENWEB_CAPS.renderers; return v==='both'?r.includes('param')&&r.includes('vansh'):r.includes(v); }
function applyHostCaps(d){ if(Array.isArray(d.renderers)) window.GENWEB_CAPS.renderers=d.renderers;
  if(typeof d.batch_stream==='boolean') window.GENWEB_CAPS.batch_stream=d.batch_stream;
  window.POSE_PREFILL_DATA=d.prefill_data||window.POSE_PREFILL_DATA||{};
  if(!renderAvailable(VW_X||'param')){ VW_X=null; VS50=false; } if(!window.GENWEB_CAPS.batch_stream) STREAM_X=null;
  if(typeof updStream==='function') updStream(); if(typeof updVw==='function') updVw(); }''')
replace("STREAM_X=q.get('stream')==='1'&&q.get('stream_set_by_user')==='1'?true:null; VW_X=['vansh','both'].includes(q.get('vw'))?q.get('vw'):null;", "STREAM_X=window.GENWEB_CAPS.batch_stream&&q.get('stream')==='1'&&q.get('stream_set_by_user')==='1'?true:null; VW_X=['vansh','both'].includes(q.get('vw'))&&renderAvailable(q.get('vw'))?q.get('vw'):null;")
replace("async function loadSchema(){ try{ const d=await (await fetch('api/schema',{cache:'no-store'})).json(); OPTS=d.options||{};", "async function loadSchema(){ try{ const d=await (await fetch('api/schema',{cache:'no-store'})).json(); OPTS=d.options||{}; applyHostCaps(d);")
replace("function effStream(){ return STREAM_X===true; }", "function effStream(){ return window.GENWEB_CAPS.batch_stream&&STREAM_X===true; }")
replace("function updStream(){ const c=$('streamck'), h=$('streamhint'); if(!c) return; const on=effStream(); if(c.checked!==on) c.checked=on;\n  setTxt(h,on?'on: this Generate streams live':'off by default: switch on to watch it live'); $('streamfld').classList.toggle('changed',on); }", """function updStream(){ const c=$('streamck'), h=$('streamhint'); if(!c) return; const hosted=window.GENWEB_CAPS.batch_stream, on=effStream(); if(!hosted) STREAM_X=null; if(c.checked!==on) c.checked=on;
  c.disabled=!hosted; c.title=hosted?'Stream this batch generation':STREAM_UNAVAILABLE;
  setTxt(h,hosted?(on?'on: this Generate streams live':'off by default: switch on to watch it live'):'batch streaming unavailable; Duplex Live mic is available');
  $('streamfld').title=hosted?'Stream this batch generation':STREAM_UNAVAILABLE; $('streamfld').classList.toggle('changed',on); }""")
start=s.index("function updVw(){ const s=$('vwsel')")
end=s.index('// 10-26 user ("can\'t find the Both setting")',start)
s=s[:start]+'''function updVw(){ const s=$('vwsel'), h=$('vwhint'); if(!s) return;
  if(!renderAvailable(VW_X||'param')){ VW_X=null; VS50=false; }
  const v=VW_X||'param', compare=renderAvailable('both'); if(s.value!==v) s.value=v;
  for(const o of s.options){ const ok=renderAvailable(o.value); o.disabled=!ok; o.title=ok?'':RENDER_UNAVAILABLE; }
  setTxt(h,compare?'Choose the renderer for this generation':'Param avatar replay; Vansh and Both are unavailable');
  $('vwfld').title=compare?'Choose the avatar renderer':RENDER_UNAVAILABLE;
  $('vwfld').classList.toggle('changed',v!=='param');
  for(const b of document.querySelectorAll('.vwseg button')){ const on=b.dataset.v===v, ok=renderAvailable(b.dataset.v); b.disabled=!ok; b.title=ok?'Param avatar replay':RENDER_UNAVAILABLE; b.setAttribute('aria-disabled',ok?'false':'true'); b.classList.toggle('on',on); b.setAttribute('aria-checked',on?'true':'false'); }
  const l50=$('vs50fld'), c50=$('vs50'); if(l50){ l50.hidden=v==='param'; l50.title=compare?'Vansh rendering steps':RENDER_UNAVAILABLE; } if(c50){ c50.disabled=!compare; if(c50.checked!==VS50) c50.checked=VS50; }
  const sg=document.querySelector('.vwseg'); if(sg) sg.title=compare?'Video weights':RENDER_UNAVAILABLE;
  const note=$('rendercapnote'); if(note){ note.hidden=compare; setTxt(note,'Only Param is hosted'); } }
''' +s[end:]
replace("g.innerHTML='<button type=\"button\" role=\"radio\" data-v=\"param\">Param</button><button type=\"button\" role=\"radio\" data-v=\"vansh\">Vansh</button><button type=\"button\" role=\"radio\" data-v=\"both\">Both</button>';", "g.innerHTML='<button type=\"button\" role=\"radio\" data-v=\"param\">Param</button><button type=\"button\" role=\"radio\" data-v=\"vansh\">Vansh</button><button type=\"button\" role=\"radio\" data-v=\"both\">Both</button><small class=\"note\" id=\"rendercapnote\">Only Param is hosted</small>';")
replace("for(const b of g.querySelectorAll('button')) b.onclick=()=>{ VW_X=b.dataset.v==='param'?null:b.dataset.v; urlSave(); };", "for(const b of g.querySelectorAll('button')) b.onclick=()=>{ if(!renderAvailable(b.dataset.v)) return; VW_X=b.dataset.v==='param'?null:b.dataset.v; urlSave(); };")
replace("$('vwsel').onchange=()=>{ VW_X=['vansh','both'].includes($('vwsel').value)?$('vwsel').value:null; urlSave(); };", "$('vwsel').onchange=()=>{ const v=$('vwsel').value; VW_X=['vansh','both'].includes(v)&&renderAvailable(v)?v:null; urlSave(); };")
replace("VANSH_LIVE=!!d.vansh_live; if(d.vansh_note) VANSH_NOTE=d.vansh_note;", "applyHostCaps(d); VANSH_LIVE=!!d.vansh_live; if(d.vansh_note) VANSH_NOTE=d.vansh_note;")
replace("b.onclick=async()=>{ b.disabled=true; try{ const r=await (await fetch('api/compare/'", "b.onclick=async()=>{ if(!renderAvailable('both')) return; b.disabled=true; try{ const r=await (await fetch('api/compare/'")
replace("setTimeout(()=>{ b.disabled=false; },3000); }; }", "setTimeout(()=>{ b.disabled=!renderAvailable('both'); },3000); }; }")
replace("    const vws=new Set(cards.map(c=>c.video_weights||'param')), other=", "    if(!renderAvailable('both')){ b.disabled=true; b.title=RENDER_UNAVAILABLE; b.setAttribute('aria-label','Compare weights unavailable: only Param is hosted'); b.lastChild.textContent='Compare weights (unavailable: Param only)'; return; }\n    const vws=new Set(cards.map(c=>c.video_weights||'param')), other=")
replace("  const DEF_PRESET='06_74807fe0';   // 10-26 16:33 PT user pick: 06 (Kate), SAM clip (was 01_ab63d745)", "  function liveDefaultPreset(P=LD.pre||SNAP){ const ids=(P.presets||[]).map(p=>p.id); return ids.includes(P.default_preset)?P.default_preset:(ids[0]||''); }")
replace("const pids=pres.map(p=>p.id), def=pids.includes(DEF_PRESET)?DEF_PRESET:((P&&P.default_preset)||pids[0]||'');", "const pids=pres.map(p=>p.id), def=liveDefaultPreset(P||{});")
replace("const pl={}; for(const p of pres){ const v=vp[p.id]||{}; pl[p.id]=(p.label||p.id)+(v.person?'':' (no video: default avatar, no prefill)'); }\n    if(pl['06_74807fe0']) pl['06_74807fe0']='Voice: 06 (Kate), SAM clip';   // 10-26 user pick (default)", "const pl={}; for(const p of pres) pl[p.id]=p.label||p.name||p.id;")
replace('hint:"B\'s preset (the live demo\'s list): its voice clip, and for the video people (Person 00-19) the avatar is that person and the call starts from their real conversation opening. Default: 06 (Kate), SAM clip (user pick 10-26)."', 'hint:"Presets and the default come from the live backend. The selected voice supplies its raw-code clip and available pose prefill."')
replace("for(const p of persons) pn[p]='Person '+p.slice(0,2);", "for(const p of persons){ const entry=pres.find(x=>(vp[x.id]||{}).person===p); pn[p]=entry?(entry.label||entry.name||p):p; }")
replace("    if(q0.get('pdv')!=='6'&&q0.get('preset')==='01_ab63d745') q0.set('preset','06_74807fe0'); q0.set('pdv','6');", "    q0.set('pdv','6');   // preserve explicit presets; new calls use the backend default")
replace("const _ck=ckOf; ckOf=function(m){ if(m==='livedup') return String(LD.step||498);", "const _ck=ckOf; ckOf=function(m){ if(m==='livedup') return String(LD.step||SNAP.step||'');")
s=s.replace('V.preset||DEF_PRESET','V.preset||liveDefaultPreset()')
start=s.index('  const START={')
end=s.index('\n  function clipPick()',start)
s=s[:start]+"  const openingAvailable=id=>!!(id&&(window.POSE_PREFILL_DATA||{})[id]&&(window.POSE_PREFILL_DATA||{})[id].raw);"+s[end:]
s=s.replace('id&&START[id]','openingAvailable(id)')
s=s.replace("+START[id].seconds", "Math.min(10,+c.seconds||10)")
start=s.index("    setTxt($l('ldclipnote'),")
end=s.index("\n    const a=$l('ldclipaudio')",start)
s=s[:start]+"    setTxt($l('ldclipnote'),eff==='start'?'In use: the selected voice’s raw conversation opening; the backend limits prefill to its available frames.':('In use: '+(c.src||'the selected voice clip')+'. The prefill uses the selected source.'+(has?'':' Conversation-start data is unavailable for this voice.')));"+s[end:]

replace("setTxt(s,!live?'':idle&&LD.down?(LD.pending?'Restarting · auto-start':'Backend restarting'):LD.running?'Live · '+mmss(LD.frames):LD.connecting?'Connecting…':'Live mic · '+vw);", "setTxt(s,!live?'':idle&&LD.down?(LD.pending?'Restarting · auto-start':'Backend restarting'):LD.running?'Live · '+mmss(LD.frames):LD.connecting?'Connecting…':'Live mic · pose preview');")
replace("s.title=live&&idle&&LD.down?DOWN_MSG:'Live mic (duplex) · video '+vw", "s.title=live&&idle&&LD.down?DOWN_MSG:'Live mic (duplex) · decoded pose preview and live audio'")
p.write_text(s)
print('Updated frontend capabilities and live preset defaults.')

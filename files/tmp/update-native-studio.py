from pathlib import Path
p=Path('/home/param/pose-demo-website/comparison/studio.html')
s=p.read_text()
a='<p id="modelhint" class="hint"></p><div class="fields">'
b='<p id="modelhint" class="hint"></p><label>Text mode<select id="text_mode"><option value="forced">Controlled · words forced</option><option value="native">Native · model generates text</option></select></label><p id="modehint" class="hint"></p><div id="text-settings" class="fields hidden"><label>Text temperature<input id="text_temp" type="number" min="0" max="3" step="0.05" value="1" required disabled></label><label>Text top-k<input id="text_topk" type="number" min="0" max="2048" step="1" value="0" required disabled></label></div><div class="fields">'
assert a in s;s=s.replace(a,b)
s=s.replace('Audio / pose seed','Generation seed')
s=s.replace('Top-k 0 is unfiltered. Each modality uses the same settings for all its codebooks.','Temperature 0 is greedy; top-k 0 is unfiltered. Audio and pose each use the same settings for all their codebooks.')
s=s.replace('<p class="hint">Duration takes the beginning of this clip; its upfront transcript covers that window.</p>','<p id="durationhint" class="hint"></p>')
a='<h2>Shared conditioning</h2><p>Every model receives this clip’s <b>same framewise text</b> and <b>same 8-second voice reference</b>.</p>'
b='<h2>Conditioning</h2><p>Every mode keeps this clip’s <b>same 8-second voice reference</b>, initial pose geometry, and video reference images.</p><p id="conditioning-text"></p>'
assert a in s;s=s.replace(a,b)
s=s.replace('<summary>Transcript</summary><div id="transcript" class="transcript"></div>','<summary>Source transcript</summary><p id="transcript-hint" class="hint" style="margin-top:8px"></p><div id="transcript" class="transcript"></div>')
s=s.replace('<p class="muted" style="font-size:13px">Aligned and compact models also receive their trained upfront transcript format. The original Xabi model receives framewise text only.</p>','<p id="timing-note" class="muted" style="font-size:13px"></p>')
start=s.index('<script>');end=s.index('</script>',start)
js='''<script>
const $=id=>document.getElementById(id);
const fields=['audio_temp','pose_temp','audio_topk','pose_topk','seed','video_seed','duration_seconds'];
let config=null,csrf='',rows=new Map();
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e}
function label(recipe){return config?.models.find(m=>m.id===recipe)?.label||recipe}
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json','X-CSRF-Token':csrf}:{},body:body?JSON.stringify(body):undefined,cache:'no-store'});if(r.status===401){$('login').classList.remove('hidden');$('app').classList.add('hidden');return null}const data=await r.json();if(!r.ok)throw new Error(data.error||'Request failed');return data}
function conditioning(){
 const recipe=$('recipe').value,allowed=(config.options.native_text_recipes||[]).includes(recipe);
 const nativeOption=$('text_mode').querySelector('[value="native"]');nativeOption.disabled=!allowed;
 nativeOption.textContent=allowed?'Native · model generates text':'Native · unavailable for aligned model';
 if(!allowed)$('text_mode').value='forced';
 const native=$('text_mode').value==='native',compact=recipe==='compact-all',aligned=recipe==='aligned-media';
 $('text-settings').classList.toggle('hidden',!native);$('text_temp').disabled=!native;$('text_topk').disabled=!native;
 $('modelhint').textContent=aligned?'Full frame-aligned transcript upfront.':compact?'Plain transcript upfront.':'Original Xabi checkpoint; no upfront transcript.';
 $('modehint').textContent=native?(compact?'The upfront transcript conditions the model; it chooses framewise words and timing.':'No transcript is supplied. The model generates its own text and timing.'):(aligned?'This model requires the supplied framewise text.':'The supplied framewise words stay fixed during generation.');
 $('conditioning-text').textContent=native?(compact?'Text input: the plain upfront transcript for the selected window.':'Text input: none. The source transcript below is for comparison only.'):'Text input: the same supplied framewise words, plus the model’s upfront transcript format where applicable.';
 $('transcript-hint').textContent=native&&!compact?'Not supplied to this model.':'A shorter duration uses the beginning of this source window.';
 $('timing-note').textContent=native?'Native output can differ in wording and timing from the source clip.':'Controlled mode keeps the supplied word timing fixed across models.';
 $('durationhint').textContent=native&&!compact?'Duration sets the generated output length; no source transcript is supplied.':'Duration takes the beginning of this clip; any upfront transcript covers that window.';
}
function chosen(resetSeed=true){const clip=config.clips.find(c=>c.id===$('clip_id').value);if(!clip)return;$('transcript').textContent=clip.transcript;$('voice').src=clip.voice_reference?.audio||'';if(resetSeed)$('seed').value=config.options.seed.defaults[clip.id];conditioning()}
async function init(){
 config=await api('/api/config');if(!config)return;csrf=config.csrf;$('login').classList.add('hidden');$('app').classList.remove('hidden');
 $('clip_id').replaceChildren(...config.clips.map(c=>{const o=el('option','',c.title);o.value=c.id;return o}));
 $('recipe').replaceChildren(...config.models.map(m=>{const o=el('option','',m.label);o.value=m.id;return o}));
 for(const name of ['text_temp','text_topk']){const option=config.options[name];if(option){$(name).min=option.min;$(name).max=option.max;$(name).value=option.default}}
 chosen();await refresh();
}
function payload(){const p={recipe:$('recipe').value,clip_id:$('clip_id').value};fields.forEach(k=>p[k]=Number($(k).value));if($('text_mode').value==='native'){p.text_mode='native';p.text_temp=Number($('text_temp').value);p.text_topk=Number($('text_topk').value)}return p}
async function submit(p){$('submit').disabled=true;$('formerror').textContent='';try{const r=await api('/api/requests',p);if(r){$('submitted').textContent='Queued. Your request appears below.';await refresh()}}catch(e){$('formerror').textContent=e.message}finally{$('submit').disabled=false}}
$('order').onsubmit=e=>{e.preventDefault();submit(payload())};$('clip_id').onchange=()=>chosen();$('recipe').onchange=()=>chosen(false);$('text_mode').onchange=conditioning;
const stages={'queued':'Queued','waiting-for-comparison':'Waiting for the comparison batch','waiting-for-gpus':'Waiting for assigned GPUs','generating':'Generating audio and pose','rendering-pose':'Rendering pose','rendering-video':'Generating video','validating':'Checking output','complete':'Complete','failed':'Failed'};
function param(text,value){const e=el('div');e.append(el('span','',text),el('b','',String(value)));return e}
function render(job){
 let row=rows.get(job.id);if(!row){row=el('article','card');rows.set(job.id,row);$('jobs').prepend(row)}
 const signature=job.status+'|'+job.stage+'|'+job.updated_at;if(row.dataset.signature===signature)return;row.dataset.signature=signature;
 const keep=row.querySelector('video');row.replaceChildren();const req=job.request,native=req.text_mode==='native',head=el('div','jobhead');
 head.append(el('h3','',`${label(req.recipe)} · ${req.clip_id.replace('clip-','Clip ')}`),el('span','pill '+job.status,stages[job.stage]||job.status));row.append(head);
 const body=el('div','jobbody');if(job.video){const v=keep||el('video');if(!keep){v.controls=true;v.preload='none';v.playsInline=true;v.src=job.video}body.append(v)}
 const params=el('div','jobparams');
 params.append(param('Text mode',native?'Native · generated words':'Controlled · words forced'),param('Transcript input',native?(req.recipe==='compact-all'?'Plain transcript upfront':'None'):req.recipe==='xabi-original'?'Framewise only':'Upfront + framewise'));
 if(native)params.append(param('Text temperature',req.text_temp??1),param('Text top-k',(req.text_topk??0)===0?'0 · unfiltered':req.text_topk));
 params.append(param('Audio temperature',req.audio_temp),param('Audio top-k',req.audio_topk===0?'0 · unfiltered':req.audio_topk),param('Pose temperature',req.pose_temp),param('Pose top-k',req.pose_topk===0?'0 · unfiltered':req.pose_topk),param('Generation seed',req.seed),param('Duration',req.duration_seconds+' seconds'),param('Video seed',req.video_seed),param('Voice / appearance','Same voice + geometry + references'));body.append(params);row.append(body);
 const generated=job.metadata?.ppm?.generated_transcript;
 if(native&&typeof generated==='string'&&generated.trim()){const detail=el('details');detail.append(el('summary','','Generated transcript'),el('div','transcript',generated));row.append(detail)}
 if(native)row.append(el('p','muted','The model generates its own framewise text and timing.'));
 if(job.error)row.append(el('p','error',job.error));const foot=el('div','jobfoot');foot.append(el('span','',new Date(job.created_at).toLocaleString()));const again=el('button','','Run again');again.onclick=()=>submit(req);foot.append(again);row.append(foot);
}
async function refresh(){if(!config)return;try{const data=await api('/api/requests');if(!data)return;const pending=data.requests.filter(j=>j.status==='queued').length,running=data.requests.filter(j=>j.status==='running').length;$('queuecount').textContent=`${running} running · ${pending} queued`;if(!data.requests.length){if(!$('jobs').querySelector('.empty'))$('jobs').append(el('div','empty','No requests yet. Choose a clip and generate your first video.'))}else{$('jobs').querySelector('.empty')?.remove();[...data.requests].reverse().forEach(render)}}catch(e){$('formerror').textContent=e.message}}
async function login(token){const r=await fetch('/auth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token})});if(!r.ok)throw new Error('Invalid access token');$('token').value='';await init()}
$('login').onsubmit=async e=>{e.preventDefault();try{await login($('token').value);$('autherror').textContent=''}catch(e){$('autherror').textContent=e.message}};
const token=new URLSearchParams(location.hash.slice(1)).get('token');if(token){history.replaceState(null,'',location.pathname);login(token).catch(e=>$('autherror').textContent=e.message)}else init().catch(e=>$('autherror').textContent=e.message);setInterval(refresh,5000);
'''
s=s[:start]+js+s[end:];p.write_text(s)
p=Path('/home/param/pose-demo-website/comparison/README.md');s=p.read_text().replace('Every request forces the prepared framewise text and the same\neight-second voice reference. Defaults are audio/pose temperature 1 and top-k 0','Controlled mode forces the prepared framewise text. Native mode is available\nfor compact transcript and original Xabi: the compact model receives its plain\nupfront transcript and samples framewise text, while original Xabi receives no\ntranscript and generates text freely. Aligned transcript remains controlled only.\nEvery mode keeps the same eight-second voice, initial pose geometry, and visual\nreferences. Native words and timing can differ from the source clip. Defaults\nare text/audio/pose temperature 1 and top-k 0');p.write_text(s)

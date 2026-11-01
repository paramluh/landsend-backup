"""page/index.html = page/index.html.orig (the PPM 2-side duo page) + the per-speaker skeleton video lane, patched exactly the way
ppm/samples/aloe-pose-ft-430/work/make_html.py patched the 2-side page (aloe_handover video lane: videos synced to the audio clock,
blob-URL load, rate follows the speed buttons, seeks follow), plus: a version without videos (run without POSE_DONE) hides the lane
and keeps the audio-only behaviour; params line adds the pose sampling only when the run's sampling has pose_topk.
   python3 patch_page_pose.py      (re-runnable: always rebuilds index.html from index.html.orig)"""
import os

P = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/page"
s = open(f"{P}/index.html.orig").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep("--mark:#7a3fc4;\n", "--mark:#7a3fc4; --vbg:#000000;\n")
rep(".empty{color:var(--muted);font-style:italic}",
    ".empty{color:var(--muted);font-style:italic}\n"
    ".vids{display:flex;flex-wrap:wrap;gap:14px;justify-content:center}\n"
    ".vids[hidden]{display:none}\n"
    ".vids figure{margin:0;display:grid;gap:4px;justify-items:center;min-width:0}\n"
    ".vids video{display:block;height:min(360px,60vw);max-width:100%;aspect-ratio:270/480;background:var(--vbg);border-radius:6px;border:2px solid var(--line)}\n"
    ".vids figure.a video{border-color:var(--a)} .vids figure.b video{border-color:var(--b)}\n"
    ".vids figcaption{font-size:12px;color:var(--muted);text-align:center;font-variant-numeric:tabular-nums;max-width:300px}\n"
    ".vids figure.a figcaption b{color:var(--a)} .vids figure.b figcaption b{color:var(--b)}")
rep('    <div class="scroller" id="scroller">',
    '    <div class="vids" id="vids" hidden>\n'
    '      <figure class="a"><video id="vA" muted playsinline preload="auto"></video><figcaption id="capA"></figcaption></figure>\n'
    '      <figure class="b"><video id="vB" muted playsinline preload="auto"></video><figcaption id="capB"></figcaption></figure>\n'
    '    </div>\n    <div class="scroller" id="scroller">')
rep("const A=$('aA'),Bq=$('aB');const AU=[A,Bq];",
    "const A=$('aA'),Bq=$('aB');const AU=[A,Bq];const VA=$('vA'),VB=$('vB');const VS=[VA,VB];let AV=[A,Bq];")
rep("function rate(){for(const a of AU){a.playbackRate=RATE;a.preservesPitch=true}",
    "function rate(){for(const a of AU){a.playbackRate=RATE;a.preservesPitch=true}for(const v of VS)v.playbackRate=RATE;")
# load: audio + (when the version has them) both videos as blob URLs; seek all
rep("let LTOK=0;async function load(t=0,autoplay=false){const v=ver(),tok=++LTOK;pause();const[sa,sb]=await Promise.all([bsrc(v.tracks[0].src),bsrc(v.tracks[1].src)]);if(tok!==LTOK)return;A.src=sa;Bq.src=sb;await Promise.all(AU.map(a=>new Promise(res=>{a.addEventListener('loadedmetadata',res,{once:true});a.load()})));for(const a of AU){try{a.currentTime=t}catch(e){}}",
    "const ready=el=>new Promise(res=>{if(el.readyState>=1)return res();el.addEventListener('loadedmetadata',res,{once:true})});\n"
    "const hasV=v=>!!(v.video&&v.video.A&&v.video.B);\n"
    "function caps(){const v=ver();for(const C of ['A','B']){const x=v.video&&v.video[C];$('cap'+C).innerHTML='<b>'+esc(role(C))+'</b> · '+esc(x?x.cap:'')}}\n"
    "let LTOK=0;async function load(t=0,autoplay=false){const v=ver(),tok=++LTOK,hv=hasV(v);pause();const[sa,sb,va,vb]=await Promise.all([bsrc(v.tracks[0].src),bsrc(v.tracks[1].src),hv?bsrc(v.video.A.src):null,hv?bsrc(v.video.B.src):null]);if(tok!==LTOK)return;"
    "if(A.src!==sa){A.src=sa;A.load()}if(Bq.src!==sb){Bq.src=sb;Bq.load()}"
    "if(hv){if(VA.src!==va){VA.src=va;VA.load()}if(VB.src!==vb){VB.src=vb;VB.load()}}else{for(const x of VS)if(x.getAttribute('src')){x.removeAttribute('src');x.load()}}"
    "AV=hv?[A,Bq,VA,VB]:[A,Bq];$('vids').hidden=!hv;"
    "await Promise.all(AV.map(ready));if(tok!==LTOK)return;for(const a of AV){try{a.currentTime=t}catch(e){}}caps();")
rep("function play(){const t=A.currentTime;Bq.currentTime=t;", "function play(){const t=A.currentTime;Bq.currentTime=t;for(const v of AV.slice(2))v.currentTime=t;")
rep("Promise.all(AU.map(a=>a.play().catch(()=>{}))).then(", "Promise.all(AV.map(a=>a.play().catch(()=>{}))).then(")
rep("function pause(){for(const a of AU)a.pause();", "function pause(){for(const a of AU)a.pause();for(const v of VS)v.pause();")
rep("const f=()=>{if(Math.abs(A.currentTime-Bq.currentTime)>0.04)Bq.currentTime=A.currentTime;",
    "const f=()=>{if(Math.abs(A.currentTime-Bq.currentTime)>0.04)Bq.currentTime=A.currentTime;for(const v of AV.slice(2))if(Math.abs(A.currentTime-v.currentTime)>0.08)v.currentTime=A.currentTime;")
rep("s.onclick=()=>{for(const a of AU)a.currentTime=tw;tick()}", "s.onclick=()=>{for(const a of AV)a.currentTime=tw;tick()}")
rep("for(const a of AU)a.currentTime=Math.max(0,Math.min(t,ver().dur-0.01));tick()});", "for(const a of AV)a.currentTime=Math.max(0,Math.min(t,ver().dur-0.01));tick()});")
rep("!['INPUT','AUDIO','BUTTON'].includes(", "!['INPUT','AUDIO','VIDEO','BUTTON'].includes(")
# params line: pose sampling, only when the run's sampling carries pose_topk
rep("· no repetition penalty · CFG off · bf16`+",
    "· no repetition penalty · CFG off · bf16`+(s.pose_topk!=null?` · pose: assistant pose generated by the model on both sides (pose temp ${s.pose_temp??1}, top-k ${s.pose_topk}, no CFG)`:'')+")
tmp = f"{P}/index.html.tmp"
open(tmp, "w").write(s)
os.replace(tmp, f"{P}/index.html")
print("ok", len(s))

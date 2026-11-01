"""PPM pose server: HTTP front (stdlib, CPU) = sam-pose-server/code/server.py's design for the PPM 32B pose FT. Requests become dirs under
requests/<id>; the resident PPM workers (2 GPUs each) and video workers (1 GPU each) claim them from disk, so the queue survives restarts
(a resubmitted identical request returns the cached result).
  GET  /                    the request form (every parameter editable, defaults shown) + recent requests
  POST /generate            JSON body (fields = /api/defaults)  -> {"id", "state", "view"}
  GET  /view/<id>           results: ground truth | samples (video + stick figure synced, audio, the model's word lane), live progress bars
  GET  /status/<id>         state, per-sample progress, meta (params, timings)
  GET  /files/<id>/<path>   s<k>/stick.mp4 s<k>/video-flash.mp4 s<k>/gen.wav s<k>/codes.npz s<k>/meta.json ... (Range ok)
  GET  /gt/<clip>/video|stick   the clip's ground-truth video / stick figure
  GET  /requests  /health  /api/defaults
   python3 server.py --port 8840"""
import argparse
import base64
import hashlib
import html
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srv_common as C  # noqa: E402

TYPES = {".mp4": "video/mp4", ".wav": "audio/wav", ".json": "application/json", ".npz": "application/octet-stream", ".js": "application/javascript"}
KL = "/data/home/amartyavarma/setup-logs/klweb"
_src = open(f"{KL}/ppm_aloe_posemono_poseonly/index.html").read()
CSS = _src[_src.index("<style>"):_src.index("</style>") + 8]
FIT = ("<style>.w{max-width:2000px}.q{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(420px,1fr))}@media (max-width:700px){.q{grid-template-columns:1fr}}"
       ".vv{display:flex;gap:6px;align-items:flex-start;justify-content:center}"
       ".vv video{flex:1 1 0;min-width:0;width:100%;max-width:100%;height:auto;max-height:62vh;object-fit:contain}"
       ".vv .ph{flex:1 1 0;min-width:0;aspect-ratio:9/16;max-height:62vh;display:flex;align-items:center;justify-content:center;"
       "border:1px dashed var(--line);border-radius:6px;color:var(--mu);font-size:13px;text-align:center}"
       ".pg{display:flex;align-items:center;gap:8px;margin:6px 0 0;font-size:12.5px;color:var(--mu)}"
       ".pg .bar{flex:1;height:8px;border-radius:4px;background:var(--line);overflow:hidden}"
       ".pg .bar i{display:block;height:100%;width:0;background:var(--ac);transition:width 1.8s linear}"
       ".pg.done .bar i{background:#2e9d5b}.pg .pl{white-space:nowrap;font-variant-numeric:tabular-nums}"
       ".tr{font-size:12.5px;line-height:1.55;margin:6px 0 0;max-height:8.5em;overflow:auto;padding:4px 7px;border:1px solid var(--line);border-radius:6px}"
       ".tr .th{font-size:11.5px;color:var(--mu);margin-bottom:2px}.tr .tl{font-weight:600;color:var(--mu)}.tr .pf{color:var(--mu)}"
       ".tr .w{border-radius:3px;padding:0 1px;transition:background .12s}.tr .w.past{opacity:.55}.tr .w.cur{background:var(--ac);color:#fff;opacity:1}"
       "form fieldset{border:1px solid var(--line);border-radius:8px;margin:0 0 10px;padding:6px 10px}form legend{font-weight:600;font-size:13px}"
       ".fl{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:6px 12px}.fl label{display:flex;flex-direction:column;font-size:12px;color:var(--mu)}"
       ".fl input,.fl select{font:inherit;font-size:13px;padding:3px 6px;border:1px solid var(--line);border-radius:5px;background:transparent;color:inherit}"
       ".fl .d{font-size:11px;opacity:.75}button.go{font:inherit;font-weight:600;padding:7px 18px;border-radius:7px;border:0;background:var(--ac);color:#fff;cursor:pointer}"
       "table.rq{border-collapse:collapse;font-size:12.5px;width:100%}table.rq td,table.rq th{border-bottom:1px solid var(--line);padding:3px 6px;text-align:left;vertical-align:top}"
       ".err{color:#d4483b;font-weight:600}</style>")
GROUPS = [("Prompt", ["prompt", "clip", "prefill_s", "gen_s", "seed", "n_samples", "ckpt", "voice_clip"]),
          ("Text (default: the fantastic set)", ["factorised_text", "cat_temp", "text_temp", "text_topk", "text_topp", "min_p", "freq_pen", "freq_win",
                                                 "pad_bias", "text_rep", "text_rep_win"]),
          ("Speech / audio (default: the fantastic set)", ["cb0_temp", "cb0_topk", "cb0_topp", "light_temp", "light_topk", "light_topp"]),
          ("Pose", ["pose_temp", "pose_topk", "pose_topp", "pose_light_temp", "pose_light_topk", "pose_light_topp", "pose_rep", "pose_light_rep",
                    "pose_rep_win"]),
          ("CFG (default off)", ["pose_cfg", "pose_cfg_scope", "cfg_uncond", "audio_cfg"]),
          ("Output", ["stick", "video", "video_stab_steps", "video_res_steps", "video_seed"])]
CHOICES = {"pose_cfg_scope": ["all", "cb0", "light"], "cfg_uncond": ["silence_pad", "silence"], "stick": ["true", "false"], "video": ["true", "false"],
           "prompt": ["clip", "upload"], "voice_clip": ["rule", "none", "upload"]}


def form_html():
    D = C.defaults()
    D["ckpt"] = C.default_ckpt()
    cl = C.clips()
    out = []
    for g, keys in GROUPS:
        fs = []
        for k in keys:
            v = D[k]
            h = html.escape(C.HELP.get(k, ""))
            if k == "ckpt":
                opts = "".join(f'<option value="{html.escape(e["path"])}"{" selected" if e["path"] == v else ""}{"" if e["available"] else " disabled"}>'
                               f'{html.escape(e["label"])}{"" if e["available"] else " (not staged yet)"}</option>' for e in C.exports())
                inp = f'<select name="{k}">{opts}</select>'
            elif k == "clip":
                opts = "".join(f'<option value="{c["id"]}"{" selected" if c["id"] == v else ""}>{html.escape(c["label"])} ({html.escape(c["group"])})</option>' for c in cl)
                inp = f'<select name="{k}">{opts}</select>'
            elif k in CHOICES:
                vv = str(v).lower()
                inp = f'<select name="{k}">' + "".join(f'<option{" selected" if o == vv else ""}>{o}</option>' for o in CHOICES[k]) + "</select>"
            else:
                inp = f'<input name="{k}" value="{html.escape(str(v))}">'
            fs.append(f'<label title="{h}">{k}{inp}<span class="d">default {html.escape(str(v))} · {h[:90]}</span></label>')
        extra = ""
        if g == "Prompt":
            pers = "".join(f'<option value="{i}">{html.escape(p["label"])}</option>' for i, p in enumerate(C.personas()))
            extra = ('<div class="fl" style="margin-top:8px"><label>persona picker (held-out Aloe personas, as the live demo)<select id="pp">'
                     f'<option value="">— none (default) —</option>{pers}</select><span class="d">fills the editable box below</span></label>'
                     '<label>uploaded prefill wav (prompt = upload)<input type="file" id="up_wav" accept="audio/*"><span class="d">first prefill_s seconds; '
                     'mono 24 kHz -23 LUFS Mimi; word lane PAD (no ASR)</span></label>'
                     '<label>optional pose npz for the upload<input type="file" id="up_pose" accept=".npz"><span class="d">prefill_pose [32,F], global_state [3], '
                     'frame0_pose [133,3], sink_latent [48,30,16] (video); none = pose generated from frame 0</span></label>'
                     '<label>uploaded voice-clip wav (voice_clip = upload)<input type="file" id="up_voice" accept="audio/*"><span class="d">3-10 s</span></label></div>'
                     '<label style="display:block;font-size:12px;color:var(--mu);margin-top:6px">persona text (empty = no persona; default)'
                     '<textarea name="persona" id="persona" rows="4" style="width:100%;font:inherit;font-size:13px"></textarea></label>')
        out.append(f'<fieldset><legend>{g}</legend><div class="fl">{"".join(fs)}</div>{extra}</fieldset>')
    pre = " ".join(f'<button type="button" class="pre" data-set="{html.escape(json.dumps(p["set"]))}">{html.escape(p["label"])}</button>' for p in C.PRESETS)
    return f'<p class="m">Presets: {pre}</p>' + "".join(out)


def req_rows(n=40):
    rows = []
    for rid in reversed(C.list_requests()[-n:]):
        if rid.startswith("_warm"):
            continue
        s = C.state(rid) or {}
        q = s.get("request", {})
        diff = {k: v for k, v in q.items() if k in C.defaults() and C.defaults()[k] != v and k not in ("ckpt",)}
        t = time.strftime("%m-%d %H:%M", time.localtime(s.get("submitted") or 0))
        wall = (s.get("t_done") or {}).get("wall_s")
        rows.append(f'<tr><td><a href="/view/{rid}">{rid}</a></td><td>{t}</td><td>{html.escape(s.get("state", ""))}</td><td>{q.get("clip")}</td>'
                    f'<td>{html.escape(json.dumps(diff))}</td><td>{"" if wall is None else f"{wall:.0f} s"}</td></tr>')
    return ('<table class="rq"><tr><th>id</th><th>submitted (UTC)</th><th>state</th><th>clip</th><th>params changed from the defaults</th><th>wall</th></tr>'
            + "".join(rows) + "</table>")


HOME = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPM Pose Server</title>
__CSS__</head><body><div class="w"><h2 style="margin:0">PPM 32B pose FT server (model kept hot)</h2>
<p class="m">Serving <b>__CK__</b>. One request = the clip's real 2 s (prefill) then the model generates its own words, speech and pose; the
model stays resident on its GPUs (no reload), the video model too. CFG is off unless you set a scale. __HEALTH__</p>
<form id="f">__FORM__<button class="go" type="submit">Generate</button> <span id="msg"></span></form>
<h3>Recent requests <a href="/gallery" style="font-size:13px">(gallery with every parameter)</a></h3>__ROWS__</div>
<script>var DEF=__DEFJSON__,PERS=__PERSJSON__;
document.querySelectorAll('button.pre').forEach(function(bt){bt.addEventListener('click',function(){var st=JSON.parse(bt.dataset.set),f=document.getElementById('f');
Object.keys(DEF).forEach(function(k){if(k==='clip'||k==='prompt'||k==='ckpt'||k==='persona'||k==='voice_clip'||k==='seed'||k==='n_samples'||k==='prefill_s'||k==='gen_s')return;
var el=f.elements[k];if(!el)return;var v=(k in st)?st[k]:DEF[k];el.value=(typeof v==='boolean')?String(v):v});document.getElementById('msg').textContent='preset: '+bt.textContent})});
document.getElementById('pp').addEventListener('change',function(e){var i=e.target.value;document.getElementById('persona').value=i===''?'':PERS[+i].persona});
function b64(id){var f=document.getElementById(id).files[0];if(!f)return Promise.resolve(null);return new Promise(function(res,rej){var r=new FileReader();
r.onload=function(){res(r.result.split(',')[1])};r.onerror=rej;r.readAsDataURL(f)})}
document.getElementById('f').addEventListener('submit',function(e){e.preventDefault();var b={};
new FormData(e.target).forEach(function(v,k){b[k]=v});document.getElementById('msg').textContent='submitting…';
Promise.all([b64('up_wav'),b64('up_pose'),b64('up_voice')]).then(function(u){if(u[0])b.upload_wav_b64=u[0];if(u[1])b.upload_pose_b64=u[1];if(u[2])b.voice_wav_b64=u[2];
return fetch('/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})}).then(function(r){return r.json()}).then(function(j){
if(j.error){document.getElementById('msg').innerHTML='<span class="err">'+j.error+'</span>';return}location.href='/view/'+j.id}).catch(function(x){
document.getElementById('msg').textContent='error '+x})});</script></body></html>"""

VIEW = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPM Pose Request</title>
__CSS__</head><body><div class="w"><p class="m"><a href="/">&larr; new request</a></p><h2 style="margin:0" id="h">__RID__</h2>
<p class="m" id="st">…</p><div class="pg" id="gpg"><div class="bar"><i id="gbi"></i></div><span class="pl" id="gpl">generation: queued</span></div>
<div class="q" id="q"></div><h3>Request</h3><pre id="rq" style="font-size:12px;white-space:pre-wrap"></pre></div>
<script src="/vsync.js?v=1"></script><script>
var RID='__RID__',built=false,have={};
function f(s){if(s==null)return'';s=Math.round(s);return s<60?s+' s':Math.floor(s/60)+' min '+(s%60)+' s'}
function esc(t){return String(t).replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]})}
function vid(src){return '<video controls preload="metadata" playsinline src="'+src+'"></video>'}
function words(ws,ps){var pre=[],gen=[];ws.forEach(function(w,i){var e=i+1<ws.length?ws[i+1].t:w.t+0.5;e=Math.min(e,w.t+1.2);
var s='<span class="w" data-s="'+w.t+'" data-e="'+e.toFixed(2)+'">'+esc(w.text)+'</span>';(w.prompt?pre:gen).push(s)});
return '<div class="tr" data-sync="1"><div class="th">the model&#x27;s own word lane (prefill words real; highlighted with playback)</div><span class="tl">prefill 0–'+ps+' s (real):</span> <span class="pf">'+(pre.join(' ')||'<i>(no words)</i>')+'</span><br><span class="tl">generated:</span> '+(gen.join(' ')||'<i>(no words)</i>')+'</div>'}
function build(s){var q=s.request,h='';
if(q.prompt==='upload'){h+='<figure><figcaption>Uploaded prompt (first '+q.prefill_s+' s are the prefill)</figcaption><audio controls style="width:100%" src="/files/'+RID+'/upload.wav"></audio>'+
'<p class="fn">voice clip: '+esc(q.voice_clip)+(q.persona?' · persona: '+esc(q.persona.slice(0,160)):' · no persona')+'</p></figure>'}
else{var gt='/gt/'+q.clip;
h+='<figure><figcaption>Ground truth (the clip&#x27;s real 12 s)</figcaption><div class="vv">'+vid(gt+'/video')+vid(gt+'/stick')+'</div><p class="fn">left: rendered video (real pose + real audio); right: stick figure · voice clip: '+esc(q.voice_clip)+(q.persona?' · persona: '+esc(q.persona.slice(0,160)):' · no persona')+'</p></figure>'}
s.samples.forEach(function(x){h+='<figure id="s'+x.k+'"><figcaption>Sample '+(x.k+1)+' · PPM ('+esc(q.ckpt.split('/').slice(-2).join('/'))+')</figcaption><div class="vv"><div class="ph" id="v'+x.k+'">'+(q.video?'video: waiting for the pose':'video off')+'</div><div class="ph" id="k'+x.k+'">generating…</div></div>'+
'<div class="pg" id="p'+x.k+'"><div class="bar"><i></i></div><span class="pl">video: '+(q.video?'waiting…':'off')+'</span></div><div id="t'+x.k+'"></div><p class="fn" id="m'+x.k+'"></p></figure>'});
document.getElementById('q').innerHTML=h;built=true}
function swap(id,src){var e=document.getElementById(id);if(!e)return;var d=document.createElement('div');d.innerHTML=vid(src);e.replaceWith(d.firstChild)}
function tick(){fetch('/status/'+RID+'?t='+Date.now()).then(function(r){return r.json()}).then(function(s){if(s.error){document.getElementById('st').textContent=s.error;return}
if(!built){build(s);document.getElementById('rq').textContent=JSON.stringify(s.request,null,1)}
var g=s.gen_progress||{};var st=s.state+(s.error?' — '+s.error:'');if(s.t_done)st+=' · wall '+f(s.t_done.wall_s)+' from submit to the last video';
document.getElementById('h').textContent=RID+': '+s.state;document.getElementById('st').textContent=st;
if(g.total){document.getElementById('gbi').style.width=(100*g.step/g.total).toFixed(1)+'%';document.getElementById('gpl').textContent='generation: '+g.stage+' · step '+g.step+' / '+g.total+(g.eta_s?' · ETA '+f(g.eta_s):'');
if(g.stage==='done')document.getElementById('gpg').classList.add('done')}
s.samples.forEach(function(x){var k=x.k,b='/files/'+RID+'/s'+k+'/';
if(x.stick&&!have['k'+k]){have['k'+k]=1;swap('k'+k,b+'stick.mp4')}
if(x.video&&!have['v'+k]){have['v'+k]=1;swap('v'+k,b+'video-flash.mp4')}
if(x.gen&&!have['t'+k]){have['t'+k]=1;fetch(b+'words.json').then(function(r){return r.json()}).then(function(w){document.getElementById('t'+k).innerHTML=words(w.s1,w.prompt_s)});
var m=x.meta||{};var tm=m.timing||{};document.getElementById('m'+k).textContent='gen '+(tm.gen_s||'?')+' s ('+(tm.ms_per_frame||'?')+' ms/frame) · '+(m.gen_words||0)+' words · seed '+s.request.seed+' row '+k+(x.meta&&x.meta.held_out?' · held-out clip':' · clip in training data');
var a=document.createElement('audio');a.controls=true;a.src=b+'gen.wav';a.style.width='100%';document.getElementById('m'+k).after(a)}
var p=x.video_progress||{},pg=document.getElementById('p'+k);if(pg){if(x.video){pg.classList.add('done');pg.querySelector('i').style.width='100%';
pg.querySelector('.pl').textContent='video: done'+(x.video_status?' · '+x.video_status.stabilizer_total_steps+' / '+x.video_status.resolver_steps+' steps · '+f(x.video_status.total_seconds):'')}
else if(x.video_fail){pg.querySelector('.pl').textContent='video failed: '+x.video_fail}
else if(p.total){pg.querySelector('i').style.width=(100*p.step/p.total).toFixed(1)+'%';pg.querySelector('.pl').textContent='video: '+p.stage+' · step '+p.step+' / '+p.total+(p.eta_s?' · ETA '+f(p.eta_s):'')}
else if(p.stage){pg.querySelector('.pl').textContent='video: '+p.stage}}});
if(s.state!=='done'&&s.state!=='failed')setTimeout(tick,1500)}).catch(function(){setTimeout(tick,3000)})}tick();
(function(){function sync(v){var fg=v.closest('figure');if(!fg)return;var tr=fg.querySelector('.tr[data-sync="1"]');if(!tr)return;var t=v.currentTime;
tr.querySelectorAll('.w[data-s]').forEach(function(w){var s=+w.dataset.s,e=+w.dataset.e;w.classList.toggle('cur',s<=t&&t<Math.max(e,s+0.12));
w.classList.toggle('past',t>=Math.max(e,s+0.12))})}
['timeupdate','seeked','pause','play'].forEach(function(n){document.addEventListener(n,function(e){if(e.target.tagName==='VIDEO'||e.target.tagName==='AUDIO')sync(e.target)},true)})})();
</script></body></html>"""


def gallery_html(n=200):
    D = C.defaults()
    cards = []
    for rid in reversed(C.list_requests()[-n:]):
        s = C.state(rid) or {}
        q = s.get("request", {})
        t = time.strftime("%m-%d %H:%M", time.localtime(s.get("submitted") or 0))
        wall = (s.get("t_done") or {}).get("wall_s")
        ck = next((e["label"] for e in C.EXPORTS if e["path"] == q.get("ckpt")), q.get("ckpt"))
        chg = {k: v for k, v in q.items() if k in D and D[k] != v and k not in ("ckpt",)}
        prow = "".join(f'<li{" style=&quot;font-weight:600&quot;" if k in chg else ""}><span>{html.escape(k)}</span><b>{html.escape(str(v)[:120])}</b></li>'
                       for k, v in q.items() if k not in ("frames_prefill", "frames_gen"))
        prow = prow.replace("&quot;", '"')
        smp = s.get("samples") or []
        vids = "".join(f'<video controls preload="none" playsinline src="/files/{rid}/s{x["k"]}/{"video-flash.mp4" if x["video"] else "stick.mp4"}"></video>'
                       for x in smp if x["video"] or x["stick"])
        txt = " · ".join(html.escape(((x.get("meta") or {}).get("gen_text") or "")[:140]) for x in smp[:1])
        cards.append(f'<figure><figcaption><a href="/view/{rid}">{rid}</a> · {t} UTC · {html.escape(s.get("state", ""))}'
                     f'{"" if wall is None else f" · wall {wall:.0f} s"}</figcaption><p class="fn">{html.escape(str(ck))} · '
                     f'{html.escape(str(q.get("clip") or "upload"))} · changed: {html.escape(json.dumps(chg))}</p>'
                     f'<div class="vv">{vids}</div><p class="fn">{txt}</p><details><summary>all parameters</summary><ul class="pp">{prow}</ul></details></figure>')
    return ('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPM Pose Gallery</title>'
            f'{CSS}{FIT}</head><body><div class="w"><p class="m"><a href="/">&larr; new request</a></p><h2 style="margin:0">Past requests</h2>'
            f'<p class="m">{len(cards)} requests, newest first; bold = changed from the defaults.</p><div class="q">{"".join(cards)}</div></div>'
            '<script src="/vsync.js?v=1"></script></body></html>')


def health():
    ws = []
    for fn in sorted(os.listdir(C.WORKERS)) if os.path.isdir(C.WORKERS) else []:
        if fn.endswith(".json"):
            w = C.rj(f"{C.WORKERS}/{fn}")
            if w:
                w["age_s"] = round(time.time() - w["t"], 1)
                ws.append(w)
    live = [w for w in ws if w["age_s"] < C.STALE_S]
    ids = C.list_requests()
    pend = [r for r in ids if not C.has(r, "GEN_DONE") and not C.has(r, "GEN_FAIL")]
    return {"workers_live": len(live), "ppm_workers": sum(w.get("kind") == "ppm" and bool(w.get("ready")) for w in live),
            "ppm_loading": sum(w.get("kind") == "ppm" and not w.get("ready") for w in live),
            "video_workers": sum(w.get("kind") == "video" and bool(w.get("ready")) for w in live), "requests": len(ids), "gen_queue": len(pend),
            "workers": ws}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write(f"[front] {time.strftime('%H:%M:%S')} {self.address_string()} {fmt % a}\n")

    def _send(self, b, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(b)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, indent=1).encode(), "application/json", code)

    def _html(self, s):
        self._send(s.encode(), "text/html; charset=utf-8")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/swap":
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                ck = body.get("ckpt")
                if not C.ckpt_ok(ck):
                    raise ValueError(f"{ck}: not a COMPLETE PPM export")
                C.write_json(f"{C.CTRL}/ckpt.json", {"default": ck, "t": time.time()})
                return self._json({"default": ck, "note": "new requests default to this export; a worker holding another export reloads (~90 s) when it picks one up"})
            except Exception as e:  # noqa: BLE001
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        if u.path != "/generate":
            return self._json({"error": "POST /generate | /swap"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            ups = {}
            for fld, name, key in (("upload_wav_b64", "upload.wav", "_upload_sha"), ("upload_pose_b64", "upload_pose.npz", "_upload_pose_sha"),
                                   ("voice_wav_b64", "voice.wav", "_voice_sha")):
                raw = body.pop(fld, None)
                if raw:
                    raw = base64.b64decode(raw)
                    if len(raw) > 64 << 20:
                        raise ValueError(f"{name}: upload larger than 64 MB")
                    ups[name] = raw
                    body[key] = hashlib.sha1(raw).hexdigest()[:16]
            rid, req = C.normalize(body)
        except Exception as e:  # noqa: BLE001
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        d = C.rdir(rid)
        cached = os.path.exists(f"{d}/request.json")
        if not cached:
            os.makedirs(d, exist_ok=True)
            for name, raw in ups.items():
                open(f"{d}/{name}.tmp", "wb").write(raw)
                os.replace(f"{d}/{name}.tmp", f"{d}/{name}")
            C.write_json(f"{d}/request.json", {**req, "_submitted": time.time()})
        s = C.state(rid)
        self._json({"id": rid, "cached": cached, "state": s["state"], "view": f"/view/{rid}"})

    def _file(self, path, name):
        if not os.path.isfile(path):
            return self._json({"error": "no such file"}, 404)
        size = os.path.getsize(path)
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b) if a and b else size - 1, size - 1)
        n = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", TYPES.get(os.path.splitext(name)[1], "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(n))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = n
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_GET(self):
        u = urlparse(self.path)
        p = [x for x in u.path.split("/") if x]
        if not p:
            hl = health()
            ld = f", {hl['ppm_loading']} loading" if hl["ppm_loading"] else ""
            hs = f"Workers: {hl['ppm_workers']} PPM ready{ld}, {hl['video_workers']} video; queue {hl['gen_queue']}."
            dj = json.dumps(dict(C.defaults(), ckpt=C.default_ckpt())).replace("</", "<\\/")
            pj = json.dumps(C.personas()).replace("</", "<\\/")
            return self._html(HOME.replace("__CSS__", CSS + FIT).replace("__CK__", html.escape(C.default_ckpt())).replace("__FORM__", form_html())
                              .replace("__ROWS__", req_rows()).replace("__HEALTH__", hs).replace("__DEFJSON__", dj).replace("__PERSJSON__", pj))
        if p[0] == "gallery":
            return self._html(gallery_html())
        if p[0] == "vsync.js":
            return self._send(open(f"{KL}/vsync.js", "rb").read(), "application/javascript")
        if p[0] == "health":
            return self._json(health())
        if p[0] == "api" and p[1:] == ["defaults"]:
            return self._json({"defaults": C.defaults(), "help": C.HELP, "clips": C.clips(), "ckpt": C.default_ckpt(), "exports": C.exports(),
                               "presets": C.PRESETS, "personas": C.personas()})
        if p[0] == "requests":
            return self._json([{"id": r, "state": (C.state(r) or {}).get("state")} for r in C.list_requests()])
        if p[0] == "gt" and len(p) == 3:
            c = C.clip(p[1])
            if not c or p[2] not in ("video", "stick"):
                return self._json({"error": "not found"}, 404)
            return self._file(c["gt_video"] if p[2] == "video" else c["gt_stick"], "x.mp4")
        if len(p) < 2:
            return self._json({"error": "not found"}, 404)
        rid = p[1]
        if p[0] == "status":
            s = C.state(rid)
            return self._json(s if s else {"error": "unknown id"}, 200 if s else 404)
        if p[0] == "view":
            return self._html(VIEW.replace("__CSS__", CSS + FIT).replace("__RID__", html.escape(rid)))
        if p[0] == "files" and len(p) >= 3:
            rel = "/".join(p[2:])
            if ".." in rel or rel.startswith("/") or "claim" in rel:
                return self._json({"error": "bad path"}, 400)
            return self._file(f"{C.rdir(rid)}/{rel}", rel)
        return self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8840)
    a = ap.parse_args()
    os.makedirs(C.REQ, exist_ok=True)
    os.makedirs(C.WORKERS, exist_ok=True)
    ThreadingHTTPServer.daemon_threads = True
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(f"[front] listening on :{a.port}", flush=True)
    srv.serve_forever()

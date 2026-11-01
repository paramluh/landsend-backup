"""speakers/<slug>/data.json + ecapa.json -> speakers/<slug>/page/index.html (the Daisy-page format). usage: build_speaker.py SLUG NAME"""
import collections, html, json, math, os, shutil, sys
SLUG, NAME = sys.argv[1], sys.argv[2]
D = f"/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/speakers/{SLUG}"
A = json.load(open(f"{D}/data.json")); EC = json.load(open(f"{D}/ecapa.json"))["loo_cos"]
ST = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/actor_stats_acc.json"))
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
e = html.escape
rank = next(i for i, a in enumerate(ST["actors"], 1) if a["actor_id"] == A["account"])
tr = ST["actors"][rank - 1]["train"]
tot_train_h = sum(a["train"].get("speech_s", 0) for a in ST["actors"]) / 3600
al = A["all"]
rows_tr = [r for r in A["rows"] if r["split"] == "train"]; rows_ev = [r for r in A["rows"] if r["split"] != "train"]
first_tr = [r for r in rows_tr if r["chunk"] == 0]
steps = math.ceil(len(rows_tr) / 64)
alts = ", ".join(n for n, _ in (NM.get(A["account"]) or {}).get("alts", [])[:2])
os.makedirs(f"{D}/page/audio", exist_ok=True)
for x in A["excerpts"]:
    shutil.copy(f"{D}/audio/{x['file']}", f"{D}/page/audio/{x['file']}")
role = {"S00": "assistant lane (S00, B)", "S01": "user lane (S01, A)"}
cards = "\n".join(f'''<article class="card">
<div class="ch"><span class="crk">#{i}</span><span class="who">with {e(x["partner_name"] or "a partner (name unknown)")}</span></div>
<div class="meta">conversation <code>{e(x["conv"])}</code> · her {role[x["lane"]]} · {x["start_s"]:.0f}–{x["start_s"] + 60:.0f} s into the call</div>
<audio controls preload="none" src="audio/{x["file"]}"></audio>
<div class="sm">60 s of her lane only (SAM-isolated); she speaks {x["her_s"]:.0f} s of it. Voice match to her other excerpts: cosine {EC[i - 1]:.2f}.</div>
<details><summary>Her words in this minute</summary><p class="tr1">{e(x["transcript"])}</p></details>
</article>''' for i, x in enumerate(A["excerpts"], 1))
page = f'''<title>{e(NAME)} in Aloe</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; who she is and how much of her there is, then her excerpts */
:root {{ --bg:#f7f6f3; --panel:#ffffff; --fg:#1d1c1a; --muted:#625f58; --line:#e2dfd8; --a:#8a6a12; --track:#efece5;
  --body:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151513; --panel:#1d1c1a; --fg:#ebe8e2; --muted:#a29e95; --line:#2f2d29;
  --a:#e0b84f; --track:#26241f; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#151513; --panel:#1d1c1a; --fg:#ebe8e2; --muted:#a29e95; --line:#2f2d29; --a:#e0b84f; --track:#26241f; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; }}
.wrap {{ max-width:900px; margin:0 auto; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:32px 56px; display:flex; flex-direction:column; gap:24px; }}
h1 {{ font-size:26px; font-weight:700; margin:0; text-wrap:balance; line-height:1.25; }}
h2 {{ font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 12px; }}
.cap {{ color:var(--muted); margin:8px 0 0; max-width:74ch; }}
.facts {{ display:flex; flex-wrap:wrap; gap:8px 24px; font-variant-numeric:tabular-nums; color:var(--muted); font-size:14px; }}
.facts b {{ color:var(--fg); font-weight:600; font-size:20px; display:block; }}
.list {{ display:flex; flex-direction:column; gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; display:flex; flex-direction:column; gap:6px; min-width:0; }}
.ch {{ display:flex; gap:10px; align-items:baseline; }}
.crk {{ font-weight:700; color:var(--a); font-variant-numeric:tabular-nums; }}
.who {{ font-weight:600; }}
.meta {{ font-size:13.5px; color:var(--muted); overflow-wrap:anywhere; }}
.sm {{ font-size:12.5px; color:var(--muted); }}
code {{ font-family:var(--mono); font-size:.9em; }}
audio {{ width:100%; height:36px; }}
details summary {{ cursor:pointer; font-size:13px; color:var(--a); }}
details summary:focus-visible {{ outline:2px solid var(--a); outline-offset:2px; }}
.tr1 {{ margin:8px 0 0; font-size:14px; background:var(--track); padding:10px 12px; border-radius:6px; }}
.notes {{ font-size:13px; color:var(--muted); max-width:78ch; border-top:1px solid var(--line); padding-top:14px; display:flex; flex-direction:column; gap:6px; }}
.notes p {{ margin:0; }}
</style>
<div class="wrap">
<header>
<h1>{e(NAME)} (account {e(A["account"][:8])})</h1>
<p class="cap">The name is the one she goes by in her own calls{f" (also heard: {e(alts)})" if alts else ""}. Account <code>{e(A["account"])}</code>.
All 8 clips below match one voice by ECAPA speaker embedding (cosine {min(EC):.2f}–{max(EC):.2f} to the other seven).</p>
</header>
<div class="facts">
<span><b>{al["own_s"] / 3600:.1f} h</b>her own speech, all chunks of her {al["convs"]} none-crossbleed calls</span>
<span><b>{tr.get("speech_s", 0) / 3600:.1f} h</b>of it in the FT plan ({len(first_tr)} first-chunk train rows)</span>
<span><b>{al["as_A"]} / {al["as_B"]}</b>calls as A (user lane) / as B (assistant lane)</span>
<span><b>{al["partners"]}</b>different partners</span>
<span><b>#{rank}</b>of {len(ST["actors"]):,} speakers in the FT plan ({100 * tr.get("speech_s", 0) / 3600 / tot_train_h:.1f}% of its hours)</span>
</div>
<p class="cap"><b>A single-speaker FT on her</b> (all chunks, her always on B, 64 rows per step): the sam-sep stores already hold {len(rows_tr)} train rows of hers
({len(first_tr)} first chunks + {len(rows_tr) - len(first_tr)} later chunks) = {steps} steps for one epoch, plus {len(rows_ev)} eval rows.
Cutting every 10-minute chunk of all her none-crossbleed calls would give about {A["chunks_if_all"]} rows ({math.ceil(A["chunks_if_all"] / 64)} steps).
Quality flags: {al["xbleed_convs"]} more calls of hers are in cross-bleed buckets (not counted); {al["purged_convs"]} were purged upstream.</p>
<section>
<h2>Listen: 8 minutes of {e(NAME)}</h2>
<div class="list">{cards}</div>
</section>
<footer class="notes">
<p><b>Audio</b>: each clip is 60 s of her own lane from a different FT train conversation, taken at the minute where she speaks most. It is decoded
from the SAM-isolated Mimi codes the FT trains on, so the partner is removed.</p>
<p><b>Transcripts</b> are her word lane from the training rows.</p>
<p><b>Hours</b>: own speech = time inside her word spans. All chunks = her lane over the full length of every none-crossbleed handover call
with sam-sep codes. FT plan = the first-chunk train rows of the current Aloe FT.</p>
</footer>
</div>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok", rank, round(al["own_s"] / 3600, 2), round(tr.get("speech_s", 0) / 3600, 2), len(rows_tr), steps, len(rows_ev), A["chunks_if_all"])

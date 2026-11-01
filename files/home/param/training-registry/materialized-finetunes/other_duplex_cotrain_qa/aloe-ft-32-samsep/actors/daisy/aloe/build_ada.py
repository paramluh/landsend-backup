"""aloe/ada.json + ada_ecapa.json + cands.json -> daisy/page/index.html: Daisy = Aloe account fQ85gun7 (goes by Ada in her calls)."""
import collections, html, json, os, shutil
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
A = json.load(open(f"{D}/aloe/ada.json")); EC = json.load(open(f"{D}/aloe/ada_ecapa.json"))["loo_cos"]
ST = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/actor_stats_acc.json"))
e = html.escape
rank = next(i for i, a in enumerate(ST["actors"], 1) if a["actor_id"] == A["account"])
me = ST["actors"][rank - 1]; tr = me["train"]
tot_train_h = sum(a["train"].get("speech_s", 0) for a in ST["actors"]) / 3600
convs = A["convs"]; n_conv = len(convs); conv_min = sum(c["min"] for c in convs); n_ft = sum(c["ft_train"] for c in convs)
names = collections.Counter(c["name"] for c in convs)
ft_rows = [r for r in A["rows"] if r["chunk"] == 0 and r["split"] == "train"]
lane = collections.Counter(r["lane"] for r in ft_rows)
partners = collections.Counter((r["partner_name"] or "name unknown") for r in ft_rows)
os.makedirs(f"{D}/page/audio", exist_ok=True)
for x in A["excerpts"]:
    shutil.copy(f"{D}/aloe/audio/{x['file']}", f"{D}/page/audio/{x['file']}")
role = {"S00": "assistant lane (S00)", "S01": "user lane (S01)"}
cards = "\n".join(f'''<article class="card">
<div class="ch"><span class="crk">#{i}</span><span class="who">with {e(x["partner_name"] or "a partner (name unknown)")}</span></div>
<div class="meta">conversation <code>{e(x["conv"])}</code> · her {role[x["lane"]]} · {x["start_s"]:.0f}–{x["start_s"] + 60:.0f} s into the call</div>
<audio controls preload="none" src="audio/{x["file"]}"></audio>
<div class="sm">60 s of her lane only (SAM-isolated); she speaks {x["her_s"]:.0f} s of it. Voice match to her other excerpts: cosine {EC[i - 1]:.2f}.</div>
<details><summary>Her words in this minute</summary><p class="tr1">{e(x["transcript"])}</p></details>
</article>''' for i, x in enumerate(A["excerpts"], 1))
plist = ", ".join(f"{e(n)}{f' ×{c}' if c > 1 else ''}" for n, c in partners.most_common())
page = f'''<title>Daisy in Aloe</title>
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
<h1>Daisy (account fQ85gun7; partner calls her Daisy, she goes by Ada in calls)</h1>
<p class="cap">How she was identified: in one call the partner says "nice picture Daisy" before anyone has introduced themselves, which reads as her app
profile name. In her calls she introduces herself as Ada ({names.get("Ada", 0)} of {n_conv}). No other speaker in the Aloe data goes by Daisy.</p>
</header>
<div class="facts">
<span><b>{tr.get("speech_s", 0) / 60:.0f} min</b>her own speech in the FT train set</span>
<span><b>{n_ft} of {n_conv}</b>of her conversations are in the FT train set</span>
<span><b>{conv_min / 60:.1f} h</b>total length of all {n_conv} of her calls</span>
<span><b>{lane.get("S00", 0)} / {lane.get("S01", 0)}</b>assistant / user lane (FT train)</span>
<span><b>#{rank}</b>of {len(ST["actors"]):,} speakers by hours ({100 * tr.get("speech_s", 0) / 3600 / tot_train_h:.2f}% of the set)</span>
</div>
<p class="cap">Partners in the FT train set: {plist}.</p>
<section>
<h2>Listen: 8 minutes of Daisy</h2>
<div class="list">{cards}</div>
</section>
<footer class="notes">
<p><b>Audio</b>: each excerpt is 60 s of her own lane from a different FT train conversation, taken at the minute where she speaks most.
It is decoded from the SAM-isolated Mimi codes the FT trains on, so the partner is removed. All 8 match one voice by ECAPA speaker embedding
(cosine {min(EC):.2f}–{max(EC):.2f} to the other seven).</p>
<p><b>Transcripts</b> are her word lane from the training rows.</p>
<p><b>Counts</b>: own speech = time inside her word spans in the FT's first-chunk train rows. Total call length counts all her handover conversations
in every bucket, both speakers included.</p>
</footer>
</div>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok rank", rank, round(tr.get("speech_s", 0) / 60, 1), n_conv, n_ft, round(conv_min / 60, 2), dict(lane), partners.most_common())

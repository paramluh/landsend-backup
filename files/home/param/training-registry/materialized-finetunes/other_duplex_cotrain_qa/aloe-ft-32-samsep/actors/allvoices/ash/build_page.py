"""clips.json + ecapa.json + ../search.json -> ash/page/index.html: is Ashley also in the pool under another account? Her clips next to the nearest voices."""
import html, json, os, shutil
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/allvoices/ash"
SR = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/allvoices/search.json"))
C = json.load(open(f"{D}/clips.json")); EC = json.load(open(f"{D}/ecapa.json"))
HA = {r["account"]: r for r in json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist/hours_all.json"))["rows"]}
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
e = html.escape
os.makedirs(f"{D}/page/audio", exist_ok=True)
ORDER = ["TyypZEov3LSf5Gp8N6kSnBTZYXy2", "bhYemu3qfvMe4qPuYeFnmattA4o1", "Ukt6wzWnq6b5qSEzBBgetqYzQvz1"]
cards = []
for acc in ORDER:
    h = HA[acc]; nm = (NM.get(acc) or {}).get("name") or "name unknown"
    au = ""
    for x in [c for c in C if c["acc"] == acc]:
        shutil.copy(f"{D}/audio/{x['file']}", f"{D}/page/audio/{x['file']}")
        au += (f'<div class="clip"><audio controls preload="none" src="audio/{x["file"]}"></audio>'
               f'<p class="txt">{e(x["text"])}</p><p class="sm">call <code>{e(x["key"][:8])}</code> · {x["start_s"]:.0f} s in · '
               f'{"assistant" if x["lane"] == "00" else "user"} lane</p></div>')
    cards.append(f'''<article class="card"><div class="ch"><span class="who">{e(nm)}</span><code class="id">{e(acc[:8])}</code></div>
<div class="meta">{h["convs"]} calls · {h["own_s"] / 60:.0f} min of own speech (all chunks)</div>{au}</article>''')
accs = EC["accs"]
th = "".join(f"<th>{e(a[:8])}</th>" for a in accs)
tr = "".join(f'<tr><th>{e(a[:8])}</th>' + "".join(f'<td class="{"d" if i == j else ""}">{v:.2f}</td>' for j, v in enumerate(r)) + "</tr>"
             for i, (a, r) in enumerate(zip(accs, EC["M"])))
page = f'''<title>Ashley Voice Search</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: a wrapping grid of account cards to compare by ear; the embedding matrix small at the bottom */
:root {{ --bg:#f6f7f5; --panel:#ffffff; --fg:#1b201d; --muted:#5b655f; --line:#dde2de; --a:#2f6f5e; --track:#eef1ee;
  --body:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121614; --panel:#1a1f1c; --fg:#e6ebe8; --muted:#9aa49e; --line:#2b322e;
  --a:#62b8a0; --track:#232a26; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#121614; --panel:#1a1f1c; --fg:#e6ebe8; --muted:#9aa49e; --line:#2b322e; --a:#62b8a0; --track:#232a26; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; }}
.wrap {{ max-width:1200px; margin:0 auto; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:28px 48px; display:flex; flex-direction:column; gap:22px; }}
h1 {{ font-size:28px; margin:0; }}
.cap {{ color:var(--muted); margin:6px 0 0; max-width:72ch; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; display:flex; flex-direction:column; gap:8px; min-width:0; }}
.ch {{ display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }}
.who {{ font-size:18px; font-weight:600; }}
.id {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); }}
.meta {{ font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.clip {{ border-top:1px solid var(--line); padding-top:8px; display:flex; flex-direction:column; gap:4px; }}
audio {{ width:100%; height:34px; }}
.txt {{ margin:0; font-size:13.5px; }}
.sm {{ margin:0; font-size:12px; color:var(--muted); }}
code {{ font-family:var(--mono); font-size:.9em; }}
.mx {{ font-size:12px; color:var(--muted); }}
.mx table {{ border-collapse:collapse; font-variant-numeric:tabular-nums; font-family:var(--mono); font-size:11.5px; }}
.mx th, .mx td {{ padding:3px 7px; border-bottom:1px solid var(--line); text-align:right; }}
.mx td.d {{ color:var(--fg); font-weight:600; }}
.tbl {{ overflow-x:auto; }}
</style>
<div class="wrap">
<header>
<h1>Ashley Voice Search</h1>
<p class="cap">No other account in the Aloe pool has Ashley's voice. We compared her voice with every one of the {SR["n_accounts"]:,} speaker
accounts in the pool (none-crossbleed calls with sam-sep codes). The two nearest voices are below, next to Ashley, so you can check by ear.
Each clip is 25 s of the account's own lane, decoded from the SAM-isolated codes. The text is its own word lane.</p>
</header>
<div class="grid">{"".join(cards)}</div>
<footer class="mx">
<p>How: one 3–6 s clean segment of own speech per call and lane (Ashley: up to 3 per call), {SR["n_segments"]:,} segments in all, embedded with ECAPA.
Each account gets a centroid. Ashley's own segments sit at a median cosine of {SR["ashley_own"]["median"]:.2f} to her centroid.
The nearest other account is Maxine at {SR["top"][0]["centroid_cos"]:.2f}, then Joy at {SR["top"][1]["centroid_cos"]:.2f}; every other account is below 0.50.</p>
<p>On the 25 s clips above (mean cosine over pairs of different clips, diagonal = within one account), Ashley vs Maxine is below both
Ashley's and Maxine's own clip-to-clip similarity. Maxine also goes by Maxine in all 14 of her named calls.</p>
<div class="tbl"><table><tr><th></th>{th}</tr>{tr}</table></div>
</footer>
</div>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok", len(cards))

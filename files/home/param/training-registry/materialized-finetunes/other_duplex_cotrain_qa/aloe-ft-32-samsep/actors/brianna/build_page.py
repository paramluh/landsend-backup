"""clips.json + ecapa.json + shortlist/hours_all.json -> brianna/page/index.html: listen page for the accounts that go by Brianna (+ Brian)."""
import html, json, os, shutil
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/brianna"
C = json.load(open(f"{D}/clips.json")); EC = json.load(open(f"{D}/ecapa.json"))
HA = {r["account"]: r for r in json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist/hours_all.json"))["rows"]}
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
e = html.escape
os.makedirs(f"{D}/page/audio", exist_ok=True)
ORDER = ["fFP3835ka3cA7TE0oUXnRIzDAqu2", "55Wms7uND0TBNXet2gFC2uBFISz2", "DV30iHXgXqM2KiQdeKmy0FLh1CO2", "GvncujyzQkdEJyY4GmYLjMoDkzs2",
         "1VxxKU1CbPbpu4VGHEFqu5KgeRV2", "UCGxWiBJdQYNSZBqFZ1g0D8tvvh2", "gCxosLDbwme6idbTGFmv1t1vUzP2"]
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
page = f'''<title>Brianna Accounts</title>
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
<h1>Brianna Accounts</h1>
<p class="cap">Seven accounts: six go by Brianna in their calls, one by Brian. Each clip is 25 s of that account's own lane, decoded from the
SAM-isolated codes, from the minute of the call where it speaks most. The text is its own word lane.</p>
</header>
<div class="grid">{"".join(cards)}</div>
<footer class="mx">
<p>ECAPA speaker-embedding cosine, mean over pairs of different clips. The diagonal compares an account's clips with each other.</p>
<div class="tbl"><table><tr><th></th>{th}</tr>{tr}</table></div>
</footer>
</div>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok", len(cards))

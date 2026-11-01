"""hours_all.json + clips.json + clips_ecapa.json -> shortlist/page/index.html: single-speaker FT candidates (Daisy-page format)."""
import html, json, os, shutil
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist"
HA = {r["account"]: r for r in json.load(open(f"{D}/hours_all.json"))["rows"]}
C = json.load(open(f"{D}/clips.json")); EC = json.load(open(f"{D}/clips_ecapa.json"))
ST = {a["actor_id"]: a for a in json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/actor_stats_acc.json"))["actors"]}
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
e = html.escape
order = sorted(HA.values(), key=lambda r: -r["own_s"])
rank_all = {r["account"]: i for i, r in enumerate(order, 1)}
os.makedirs(f"{D}/page/audio", exist_ok=True)
def name(a):
    v = NM.get(a) or {}
    if a == "fQ85gun79yZYPSwmn7F4wY1sOd73":
        return "Ada (“Daisy”)"
    return v.get("name") or "name unknown"
cards = []
for acc in C["pick"]:
    h = HA[acc]; ft = (ST.get(acc) or {}).get("train", {})
    ref = acc == "fQ85gun79yZYPSwmn7F4wY1sOd73"
    au = ""
    for x in C["clips"][acc]:
        shutil.copy(f"{D}/audio/{x['file']}", f"{D}/page/audio/{x['file']}")
        au += (f'<audio controls preload="none" src="audio/{x["file"]}"></audio>'
               f'<div class="sm">45 s · {"assistant lane (B)" if x["lane"] == "S00" else "user lane (A)"} · conversation <code>{e(x["conv"][:8])}</code> · speaks {x["own_s"]:.0f} s'
               f'<details><summary>words</summary><p class="tr1">{e(x["transcript"])}</p></details></div>')
    tot = h["as_A"] + h["as_B"]
    flags = f'{h["xbleed_convs"]} more calls in cross-bleed buckets (not counted)' + (f', {h["purged_convs"]} purged' if h["purged_convs"] else ", none purged")
    cards.append(f'''<article class="card{' ref' if ref else ''}">
<div class="ch"><span class="crk">#{rank_all[acc]}</span><span class="who">{e(name(acc))}</span><code class="sm">{e(acc)}</code></div>
<div class="big">{h["own_s"] / 3600:.1f} h <span class="u">own speech, all chunks · {ft.get("speech_s", 0) / 3600:.1f} h in the FT plan</span></div>
<div class="meta">{h["convs"]} calls · {h["partners"]} different partners · {flags}</div>
<div class="split"><div class="sa" style="width:{100 * h["as_B"] / max(1, tot):.0f}%"></div></div>
<div class="sl"><span>as B (assistant lane) {h["as_B"]}</span><span>as A (user lane) {h["as_A"]}</span></div>
{au}
<div class="sm">Both clips match one voice: ECAPA cosine {EC.get(acc) or 0:.2f}.</div>
</article>''')
top3 = order[:3]
head = "; ".join(f"{e(name(r['account']))} {r['own_s'] / 3600:.1f} h" for r in top3)
tot_h = sum(r["own_s"] for r in order) / 3600
page = f'''<title>Single-Speaker Shortlist</title>
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
.big {{ font-size:22px; font-weight:600; font-variant-numeric:tabular-nums; }}
.u {{ font-size:13px; font-weight:400; color:var(--muted); }}
.split {{ height:6px; border-radius:3px; background:var(--muted); overflow:hidden; opacity:.85; }}
.sa {{ height:100%; background:var(--a); }}
.sl {{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.ref {{ border-style:dashed; }}
.head3 {{ font-size:19px; max-width:64ch; margin:8px 0 0; }}
</style>
<div class="wrap">
<header>
<h1>Single-Speaker Shortlist</h1>
<p class="head3">The best three candidates by their own speech: <b>{head}</b>. These are hours of their own voice over every chunk of every
none-crossbleed Aloe call with sam-sep codes ({len(order):,} speakers, {tot_h:,.0f} h in all).</p>
<p class="cap">The top 12 speakers, then Ada (“Daisy”) for reference. Each card has two 45 s clips of the speaker's own lane, decoded from the
SAM-isolated codes, so the partner is removed.</p>
</header>
<div class="list">{"".join(cards)}</div>
<footer class="notes">
<p><b>Own speech</b> = time inside the speaker's word spans on their lane (words.json, _separated_speaker), over the full length of each call.
<b>FT plan</b> = the same measure over the current Aloe FT's first-chunk train rows.</p>
<p><b>A / B</b>: in the Aloe FT, A is the user lane (S01) and B the assistant lane (S00). Calls in the cross-bleed buckets are flagged and not
counted. Purged = calls upstream dropped from sam-sep (no face).</p>
<p><b>Names</b> are the names speakers go by in their own calls.</p>
</footer>
</div>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok", head)

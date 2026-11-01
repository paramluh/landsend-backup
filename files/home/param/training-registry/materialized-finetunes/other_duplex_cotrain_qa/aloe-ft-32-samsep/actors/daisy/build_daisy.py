"""daisy/daisy.json -> daisy/page/index.html: who 'Daisy' is in the training data (PPM podcast personas), per podcast."""
import collections, html, json, os, re
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
J = json.load(open(f"{D}/daisy.json"))
R = J["rows"]
os.makedirs(f"{D}/page/audio", exist_ok=True)
import glob, shutil  # noqa: E402
EX = []
for fj in sorted(glob.glob(f"{D}/audio/*.json")):
    e = json.load(open(fj))
    if os.path.exists(f"{D}/audio/{e['file']}"):
        shutil.copy(f"{D}/audio/{e['file']}", f"{D}/page/audio/{e['file']}")
        EX.append(e)
EX.sort(key=lambda e: e["ek"].split("/")[1])
by = collections.defaultdict(list)
for r in R:
    by[r["ek"].split("/")[1]].append(r)
pods = sorted(by.items(), key=lambda kv: -sum(r["sec"] for r in kv[1]))
tot_h = sum(r["sec"] for r in R) / 3600
eps = len({r["ek"] for r in R})
sides = collections.Counter(r["side"] for r in R)
trees = collections.Counter(r["ek"][0] for r in R)
TREE = {"A": "SAM 700k conversational", "B": "SAM intelligent conversations"}
def sent(t):
    t = t.replace("\n", " ")
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    return m.group(1) if m else t[:200]
cards = []
for i, (pod, rs) in enumerate(pods, 1):
    h = sum(r["sec"] for r in rs) / 3600
    sd = collections.Counter(r["side"] for r in rs)
    partners = collections.Counter(r["partner_name"] for r in rs if r["partner_name"])
    who = collections.Counter(sent(r["who"]) for r in rs).most_common(1)[0][0]
    full = max(rs, key=lambda r: r["tokens"] or 0)["prompt"]
    side_txt = " · ".join(f"{k} side {v}" for k, v in sorted(sd.items()))
    ptxt = ", ".join(n for n, _ in partners.most_common(3)) or "not named"
    cards.append(f'''<article class="card">
<div class="ch"><span class="crk">#{i}</span><span class="who">{html.escape(who)}</span></div>
<div class="meta">{h:.2f} h · {len(rs)} chunk{"s" if len(rs) != 1 else ""} · {len({r["ek"] for r in rs})} episode{"s" if len({r["ek"] for r in rs}) != 1 else ""} · {side_txt} · partner: {html.escape(ptxt)}</div>
<div class="sm">{TREE[rs[0]["ek"][0]]} · podcast <code>{pod[:8]}</code></div>
<details><summary>Full persona prompt ({max(r["tokens"] or 0 for r in rs)} tokens)</summary><pre>{html.escape(full)}</pre></details>
</article>''')
def turns(e):
    pn = e["partner_name"] or "Partner"
    out = []
    for w, t in e["turns"]:
        t = t if len(t) < 260 else t[:257] + "…"
        cls, who = ("d", "Daisy") if w == "D" else ("p", html.escape(pn))
        out.append(f'<p class="{cls}"><b>{who}</b> {html.escape(t)}</p>')
    return "".join(out)


ex_html = "".join(f'''<article class="card">
<div class="who">{html.escape(sent(e["who"]))}</div>
<div class="meta">Daisy is the {e["side"]} side · partner: {html.escape(e["partner_name"] or "not named")} · {TREE[e["ek"][0]]} · from {e["start_s"]:.0f} s into the chunk</div>
<audio controls preload="none" src="audio/{e["file"]}"></audio>
<div class="sm">40 s, both sides. <b>Daisy is the left channel</b>, the partner the right. Daisy talks {e["daisy_s"]:.0f} s, the partner {e["partner_s"]:.0f} s.</div>
<details><summary>Transcript of the excerpt</summary><div class="tr">{turns(e)}</div></details>
</article>''' for e in EX)
page = f'''<title>Daisy in Training</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; the answer first, then one card per podcast whose persona is named Daisy */
:root {{ --bg:#f7f6f3; --panel:#ffffff; --fg:#1d1c1a; --muted:#625f58; --line:#e2dfd8; --a:#8a6a12; --track:#efece5;
  --body:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#151513; --panel:#1d1c1a; --fg:#ebe8e2; --muted:#a29e95; --line:#2f2d29;
  --a:#e0b84f; --track:#26241f; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#151513; --panel:#1d1c1a; --fg:#ebe8e2; --muted:#a29e95; --line:#2f2d29; --a:#e0b84f; --track:#26241f; color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; }}
.wrap {{ max-width:900px; margin:0 auto; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:32px 56px; display:flex; flex-direction:column; gap:24px; }}
h1 {{ font-size:30px; font-weight:700; margin:0; text-wrap:balance; }}
h2 {{ font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 12px; }}
.lede {{ font-size:19px; max-width:62ch; margin:8px 0 0; }}
.lede b {{ font-weight:600; }}
.sub {{ color:var(--muted); margin:8px 0 0; max-width:72ch; }}
.facts {{ display:flex; flex-wrap:wrap; gap:6px 22px; font-variant-numeric:tabular-nums; color:var(--muted); font-size:14px; }}
.facts b {{ color:var(--fg); font-weight:600; }}
.list {{ display:flex; flex-direction:column; gap:12px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; display:flex; flex-direction:column; gap:6px; min-width:0; }}
.ch {{ display:flex; gap:10px; align-items:baseline; }}
.crk {{ font-weight:700; color:var(--a); font-variant-numeric:tabular-nums; }}
.who {{ font-weight:500; }}
.meta {{ font-size:13.5px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.sm {{ font-size:12.5px; color:var(--muted); }}
code {{ font-family:var(--mono); font-size:.92em; }}
details summary {{ cursor:pointer; font-size:13px; color:var(--a); }}
details summary:focus-visible {{ outline:2px solid var(--a); outline-offset:2px; }}
audio {{ width:100%; height:36px; }}
.tr {{ display:flex; flex-direction:column; gap:4px; font-size:13.5px; margin-top:8px; }}
.tr p {{ margin:0; }} .tr .d b {{ color:var(--a); }} .tr .p b {{ color:var(--muted); }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font-family:var(--body); font-size:13.5px; background:var(--track); padding:12px; border-radius:6px; margin:8px 0 0; }}
.notes {{ font-size:13px; color:var(--muted); max-width:78ch; border-top:1px solid var(--line); padding-top:14px; display:flex; flex-direction:column; gap:6px; }}
.notes p {{ margin:0; }}
</style>
<div class="wrap">
<header>
<h1>Daisy in Training</h1>
<p class="lede">Daisy is a <b>persona name</b>, not one person. In PPM's podcast training data, <b>{len(pods)} different podcast speakers</b>
have a persona prompt that names them Daisy, across <b>{len(R)} training chunks</b> ({tot_h:.1f} h of conversation) in {eps} episodes.</p>
<p class="sub">No speaker or persona is called Daisy in the Aloe data, which covers all 12,681 handover conversations and the FT set. Daisy is the
assistant-side persona in {sides.get("assistant", 0)} of the chunks and the user-side persona in {sides.get("user", 0)}.</p>
</header>
<div class="facts"><span><b>{len(pods)}</b> podcasts</span><span><b>{eps}</b> episodes</span><span><b>{len(R)}</b> chunks</span><span><b>{tot_h:.1f} h</b></span>
<span><b>{trees.get("A", 0)}</b> chunks in the 700k tree</span><span><b>{trees.get("B", 0)}</b> in the intelligent tree</span></div>
<section>
<h2>Listen: {len(EX)} Daisy conversations</h2>
<div class="list">{ex_html}</div>
</section>
<section>
<h2>Each Daisy, by hours</h2>
<div class="list">{"".join(cards)}</div>
</section>
<footer class="notes">
<p><b>Source</b>: the PPM persona sidecars the PPM and PPM-Voxtral runs train with (<code>qwen-1ch-data/persona/processed</code>, status ok).
A chunk counts when its own persona says "Your name is Daisy", "You are Daisy" or "You go by Daisy". Another {J["mentions_not_name"]:,} persona prompts
mention a Daisy who is not the speaker, such as a partner, a pet, Gatsby's Daisy or Driving Miss Daisy; those are not counted.</p>
<p><b>Hours</b> = the length of those chunks, both speakers included.</p>
<p><b>Audio</b>: decoded from the Mimi codes of the training rows themselves, copied read-only from the training hosts' row stores, one chunk per podcast.</p>
</footer>
</div>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok", len(pods), len(R), round(tot_h, 1))

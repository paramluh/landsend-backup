"""data.json -> index.html (Aloe FT Personas): the persona prompts of the Aloe FT train rows, their structure, and 15 labelled examples.
Layout and styles = the Training Personas page (actors/personas/build_page.py)."""
import html, json, os
D = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{D}/data.json"))
S, EX = J["stats"], J["examples"]
e = html.escape
f0 = lambda x: f"{x:,.0f}"
pc = lambda x: f"{100 * x:.0f}%"
FIELDS = [f["field"] for f in S["fields"]]
HUE = {f: (i * 137.5 + 20) % 360 for i, f in enumerate(FIELDS)}
def hs(f):
    return f'style="--h: {HUE.get(f, 0):.0f}"'
T = S["tokens"]["all"]
ok_share = (S["status"].get("user|ok", 0) + S["status"].get("assistant|ok", 0)) / (2 * S["rows"])
per = S["sections"]["persona"]
kinds = S["kinds"]; kt = sum(kinds.values())
headline = (f"All {f0(S['rows'])} train conversations of the Aloe FT carry a prompt on both sides ({pc(ok_share)} user, {pc(ok_share)} assistant). "
            f"{pc(per['share'])} have a Persona part; the rest carry only a Starting context. Median length {f0(T['median'])} tokens, p90 {f0(T['p90'])}.")
NAMES = {"persona": "Persona", "starting_context": "Starting context", "memory": "Earlier conversation memory"}
sec_rows = "\n".join(f'<tr><th scope="row">{NAMES[k]}</th><td>{pc(S["sections"].get(k, {}).get("share", 0))}</td>'
                      f'<td>{f0(S["sections"].get(k, {}).get("median", 0))}</td><td>{f0(S["sections"].get(k, {}).get("p90", 0))}</td></tr>' for k in NAMES)
def kind_bar(kd):
    t = sum(kd.values()) or 1
    return "".join(f'<span class="bar" title="{k} {kd.get(k, 0)}" style="width:{60 * kd.get(k, 0) / t:.1f}px;background:var({c})"></span>'
                   for k, c in (("real", "--user"), ("variant", "--muted"), ("invented", "--inv")))
def kind_txt(kd):
    t = sum(kd.values()) or 1
    return " · ".join(f"{k} {100 * kd.get(k, 0) / t:.0f}%" for k in ("real", "variant", "invented") if kd.get(k))
field_rows = "\n".join(f'<tr><th scope="row"><span class="sw" {hs(f["field"])}></span>{e(f["field"])}</th><td>{pc(f["conv_share"])}</td>'
                        f'<td>{f["per_prompt"]:.2f}</td><td>{f0(f["median_words"])}</td><td class="l">{kind_bar(f["kinds"])} <span class="sub" style="display:inline">{kind_txt(f["kinds"])}</span></td></tr>'
                        for f in S["fields"])
modes = S["modes"]; mt = sum(modes.values())
KL = {"invented": "invented", "variant": "reworded", "real": "said"}
def card(i, x):
    nm = x["name"] or "name unknown"
    its = "".join(f'<div class="it" {hs(it["field"])}><span class="fc">{e(it["field"] or "?")}</span><span class="tx">{e(it["text"] or "")}'
                  f'<span class="k k-{it["kind"]}">{KL.get(it["kind"], it["kind"])}</span></span></div>' for it in x["items"])
    per_block = f'<p class="sec">Persona · {len(x["items"])} items · {f0(x["tokens"].get("persona", 0))} tokens</p><div class="fl">{its}</div>' if x["items"] else '<p class="sec">Persona · none (context only)</p>'
    return f'''<article class="card" id="ex{i + 1}">
<header class="ch"><span class="n">{i + 1:02d}</span><span class="ds">{e(nm)}</span><span class="side side-{x["side"]}">{x["side"]}</span>
<span class="meta">{f0(x["tokens"].get("system_prompt", 0))} tokens · items drawn from {"the whole call" if x["mode"] == "whole_episode" else "before this chunk only"}</span></header>
<p class="key">conversation {e(x["conv"])} · {x["side"]} side · speaker {e(x["account"][:8])}</p>
{per_block}
<p class="sec">Starting context · {f0(x["tokens"].get("starting_context", 0))} tokens</p><p class="ctx">{e(x["starting_context"])}</p>
</article>'''
cards = "\n".join(card(i, x) for i, x in enumerate(EX))
withp = sorted([x for x in EX if x["items"]], key=lambda x: x["tokens"].get("system_prompt", 0))
pick = [withp[i] for i in sorted({round(j * (len(withp) - 1) / 5) for j in range(6)})]
def share(x):
    t = len(x["items"]) or 1
    c = {k: sum(1 for it in x["items"] if it["kind"] == k) for k in ("real", "variant", "invented")}
    return f'{len(x["items"])} item{"" if len(x["items"]) == 1 else "s"}: {100 * c["real"] / t:.0f}% said, {100 * c["variant"] / t:.0f}% reworded, {100 * c["invented"] / t:.0f}% invented'
prose = "\n".join(f'''<article class="card">
<header class="ch"><span class="n">{i + 1:02d}</span><span class="side side-{x["side"]}">{x["side"]}</span><span class="meta">conversation {e(x["conv"])} · {f0(x["tokens"].get("system_prompt", 0))} tokens</span></header>
<p class="cap1">{share(x)}</p>
<pre class="txt">{e(x["prompt"])}</pre>
</article>''' for i, x in enumerate(pick))
H = f'''<title>Aloe FT Personas</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; headline, one stats block, then example cards grouped by dataset */
:root {{
  --bg: #f4f6f8; --panel: #ffffff; --ink: #18202a; --muted: #5a6573; --rule: #dde2e8; --accent: #1f5f8b;
  --user: #0f7a6c; --user-bg: #e2f3f0; --asst: #9a5a00; --asst-bg: #fbeedb; --sec: #1f5f8b; --pre-bg: #f8fafb; --fs: 52%; --fl: 40%; --inv: #8a3fb0; --inv-bg: #f3e8f9;
  --f-display: "Instrument Sans", "Helvetica Neue", Arial, sans-serif;
  --f-body: "Instrument Sans", "Helvetica Neue", Arial, sans-serif;
  --f-read: "Newsreader", Georgia, "Times New Roman", serif;
  --f-mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1318; --panel: #171c23; --ink: #e4e9ef; --muted: #98a3b1; --rule: #2a323d; --accent: #7cb6e0;
  --user: #5fd0bf; --user-bg: #12302c; --asst: #f0b260; --asst-bg: #382811; --sec: #8cc0e6; --pre-bg: #12171d; --fs: 60%; --fl: 68%; --inv: #d39af0; --inv-bg: #2e1d3a; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0f1318; --panel: #171c23; --ink: #e4e9ef; --muted: #98a3b1; --rule: #2a323d; --accent: #7cb6e0;
  --user: #5fd0bf; --user-bg: #12302c; --asst: #f0b260; --asst-bg: #382811; --sec: #8cc0e6; --pre-bg: #12171d; --fs: 60%; --fl: 68%; --inv: #d39af0; --inv-bg: #2e1d3a; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 15px; line-height: 1.55; }}
.wrap {{ max-width: 880px; margin: 0 auto; padding-block: 40px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; gap: 36px; }}
.eyebrow {{ font: 500 12px/1.4 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }}
h1 {{ font: 700 clamp(30px, 6vw, 42px)/1.1 var(--f-display); letter-spacing: -.01em; margin: 0 0 14px; text-wrap: balance; }}
h2 {{ font: 600 20px/1.3 var(--f-display); margin: 0 0 12px; text-wrap: balance; }}
h3 {{ font: 600 12px/1.4 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }}
.headline {{ font-size: 18px; line-height: 1.5; max-width: 62ch; margin: 0; }}
.note {{ color: var(--muted); max-width: 68ch; margin: 0; }}
.note code, .key, .who {{ font-family: var(--f-mono); font-size: 12.5px; }}
.stats {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 20px; display: grid; gap: 22px; }}
.tbl {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px; }}
th, td {{ text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; white-space: nowrap; }}
th[scope="row"], thead th:first-child {{ text-align: left; font-weight: 600; }}
thead th {{ font: 500 11.5px/1.3 var(--f-mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }}
.sub {{ display: block; font-size: 11.5px; color: var(--muted); font-weight: 400; }}
.lists {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; }}
.lists > div {{ min-width: 0; }}
ol.rank {{ list-style: none; margin: 0; padding: 0; display: grid; gap: 3px; font-variant-numeric: tabular-nums; }}
ol.rank li {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px dotted var(--rule); padding: 2px 0; }}
ol.rank .ct {{ color: var(--muted); font-family: var(--f-mono); font-size: 12.5px; }}
.lists .cap {{ font-size: 12.5px; color: var(--muted); margin: 6px 0 0; }}
.group {{ display: grid; gap: 18px; }}
.card {{ background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px 18px; display: grid; gap: 8px; min-width: 0; }}
.ch {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
.n {{ font: 500 12px var(--f-mono); color: var(--muted); }}
.ds {{ font-weight: 600; font-size: 14px; }}
.side {{ font: 500 11.5px/1 var(--f-mono); text-transform: uppercase; letter-spacing: .05em; padding: 4px 7px; border-radius: 4px; }}
.side-user {{ color: var(--user); background: var(--user-bg); }}
.side-assistant {{ color: var(--asst); background: var(--asst-bg); }}
.meta {{ color: var(--muted); font-size: 13px; }}
.key {{ margin: 0; color: var(--muted); overflow-wrap: anywhere; }}
.who {{ margin: 0; color: var(--ink); overflow-wrap: anywhere; }}
.lbl {{ color: var(--muted); }}
pre.txt {{ margin: 4px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--f-read); font-size: 16.5px; line-height: 1.6;
  background: var(--pre-bg); border-radius: 6px; padding: 14px 16px; color: var(--ink); }}
.sec {{ font: 600 11.5px/2 var(--f-mono); letter-spacing: .06em; text-transform: uppercase; color: var(--sec); }}
nav.jump {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 14px; }}
a {{ color: var(--accent); }}
a:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
footer {{ color: var(--muted); font-size: 13px; }}

.fl {{ display: grid; gap: 0; }}
.it {{ display: grid; grid-template-columns: 8.5em 1fr; gap: 10px; padding: 5px 0 5px 10px; border-left: 4px solid hsl(var(--h) var(--fs) var(--fl)); align-items: baseline; }}
.it + .it {{ border-top: 1px dotted var(--rule); }}
.fc {{ font: 500 11px/1.3 var(--f-mono); letter-spacing: .03em; text-transform: uppercase; color: hsl(var(--h) var(--fs) var(--fl)); overflow-wrap: anywhere; }}
.it .tx {{ font-family: var(--f-read); font-size: 16px; line-height: 1.5; min-width: 0; }}
.k {{ font: 500 10.5px/1 var(--f-mono); text-transform: uppercase; letter-spacing: .04em; padding: 3px 5px; border-radius: 3px; margin-left: 6px; white-space: nowrap; vertical-align: 1px; }}
.k-invented {{ color: var(--inv); background: var(--inv-bg); }}
.k-variant {{ color: var(--muted); background: var(--pre-bg); border: 1px solid var(--rule); }}
.k-real {{ color: var(--user); background: var(--user-bg); }}
.ctx {{ font-family: var(--f-read); font-size: 16px; line-height: 1.5; background: var(--pre-bg); border-radius: 6px; padding: 10px 14px; margin: 0; }}
.sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; background: hsl(var(--h) var(--fs) var(--fl)); margin-right: 7px; vertical-align: 0; }}
td.l {{ text-align: left; }}
.bar {{ display: inline-block; height: 8px; border-radius: 2px; vertical-align: 1px; }}
@media (max-width: 520px) {{ .it {{ grid-template-columns: 1fr; gap: 2px; }} }}
.cap1 {{ margin: 0; font-size: 13.5px; color: var(--muted); }}
details.more > summary {{ cursor: pointer; font: 600 17px/1.4 var(--f-display); color: var(--accent); padding: 4px 0; }}
details.more > summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
.moreb {{ display: grid; gap: 36px; margin-top: 20px; }}
</style>
<div class="wrap">
<header>
<p class="eyebrow">Aloe FT · train set · system prompts</p>
<h1>Aloe FT Personas</h1>
<p class="headline">{headline}</p>
</header>

<section class="group" id="prose" aria-labelledby="h-prose">
<h2 id="h-prose">{len(pick)} personas as the model reads them</h2>
<p class="note">The exact prompt text from the training rows, shortest to longest, each from a different speaker.</p>
{prose}
</section>

<details class="more"><summary>Structure, field breakdown and {len(EX)} labelled examples</summary>
<div class="moreb">
<section class="stats" aria-labelledby="st">
<h2 id="st">What a prompt is made of</h2>
<div class="tbl"><table>
<thead><tr><th>Section</th><th>In prompts</th><th>Median tokens</th><th>p90 tokens</th></tr></thead>
<tbody>
{sec_rows}
</tbody></table></div>
<p class="note">Every train row is the first 10 minutes of a call, so no prompt has an Earlier conversation memory. Each Persona is a list of short items.
Each item has a <b>field</b> (what it is about) and a <b>kind</b>: <b>said</b> = a fact the speaker states in the call, <b>reworded</b> = such a fact paraphrased or
with less detail, <b>invented</b> = a plausible filler fact the generator added that the call never states. Of {f0(kt)} persona items, {100 * kinds.get("real", 0) / kt:.0f}% are said,
{100 * kinds.get("variant", 0) / kt:.0f}% reworded and {100 * kinds.get("invented", 0) / kt:.0f}% invented. A persona has {S["items_per_prompt"]["median"]:.0f} items at the median (p90 {S["items_per_prompt"]["p90"]:.0f}).
For {100 * modes.get("earlier_only", 0) / mt:.0f}% of sides the items come only from what was said before the chunk, which for a first chunk means invented items only or no Persona at all;
for the other {100 * modes.get("whole_episode", 0) / mt:.0f}% they come from the whole call.</p>
<h3>Persona fields</h3>
<div class="tbl"><table>
<thead><tr><th>Field</th><th>In prompts</th><th>Items per prompt</th><th>Median words</th><th style="text-align:left">Said · reworded · invented</th></tr></thead>
<tbody>
{field_rows}
</tbody></table></div>
<p class="note">Tokens are the Qwen3-32B counts stored with each prompt. The prompt text in the store's persona sidecar equals the handover chunk prompt it came from in all {f0(int(S["same_prompt"].get("True", 0)))} sides.</p>
</section>

<section class="group" id="ex" aria-labelledby="h-ex">
<h2 id="h-ex">{len(EX)} example prompts</h2>
<p class="note">One per speaker, alternating sides. The first 15 have a Persona part and run from shortest to longest; the last one is a context-only prompt. Item colours match the field list above.
Badges give the kind of each item.</p>
{cards}
</section>
</div>
</details>

<footer>Source: the sam-sep Aloe FT stores (persona sidecars), first-chunk train rows, and the handover chunk prompts (chunk_00.json, lean-v12.18-aloe). Names: the speaker-account names on the Aloe FT Actors page. Built 2026-10-24.</footer>
</div>
'''
open(f"{D}/index.html", "w").write(H)
print("wrote", len(H))

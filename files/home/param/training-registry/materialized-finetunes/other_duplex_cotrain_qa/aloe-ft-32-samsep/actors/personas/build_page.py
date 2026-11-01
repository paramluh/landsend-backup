"""data.json (+ sam_check.json) -> index.html (Training Personas page)."""
import html, json, os
D = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{D}/data.json"))
SC = json.load(open(f"{D}/sam_check.json")) if os.path.exists(f"{D}/sam_check.json") else None
S, R, A, EX = J["ppm"], J["ppm_report"], J["aloe"], J["examples"]
e = html.escape
f0 = lambda x: f"{x:,.0f}"
pct = lambda a, b: ((f"{100 * a / b:.2f}%" if 0.995 < a / b < 1 else f"{100 * a / b:.1f}%") if b else "–")

# PPM numbers (full counts from REPORT.json; lengths from the parquet sample)
all_ = R["by_dataset"]["ALL"]; rows = all_["rows"]
pu, pa_, pb = all_["user_ok"], all_["assistant_ok"], all_["both_ok"]
pt, pw = S["tokens"], S["words"]
# Aloe numbers
ac = A["counts"]; segs = A["segments"]
au, aa = ac.get("user_seg_with_prompt", 0), ac.get("assistant_seg_with_prompt", 0)
at, aw = A["tokens"], A["words"]
nn_u, nn_a = ac.get("user_seg_with_prompt_none", 0), ac.get("assistant_seg_with_prompt_none", 0)
nn_seg = ac.get("user_segments_none", 0)

def mean(d, k):
    return f0(d[k]["mean"]) if d.get(k) else "–"

headline = (f"PPM podcast: {pct(pu, rows)} of {f0(rows)} rows carry a user persona and {pct(pa_, rows)} an assistant persona, "
            f"averaging {mean(pt, 'all')} tokens. Aloe: {pct(au, segs)} / {pct(aa, segs)} of {f0(segs)} segments, averaging {mean(at, 'all')} tokens.")

def chips(pairs, n=10):
    return "".join(f'<li><span class="nm">{e("CEO" if k == "ceo" else str(k))}</span><span class="ct">{f0(v)}</span></li>' for k, v in pairs[:n])

ppm_names = [p for p in S["names"] if p[0] != "(unnamed)"]
aloe_names = [p for p in A["names"] if p[0] != "(unnamed)"]
roles = S["roles"]; occ = S["occupations"]; rel = A["relationship"]
eps = S["eps_sides"]

sam_line = ""
if SC:
    c = SC["counts"]
    sam_line = (f"SAM test slice (lean-v12.16, {c.get('sam_prompts', 0)} chunk prompts from 11 episodes): "
                f"{c.get('identical', 0)} match the PPM podcast prompt for the same episode, side and chunk byte for byte, "
                f"{c.get('different', 0)} differ, {c.get('sam_not_in_ppm', 0)} have no PPM row. It is the same data as PPM podcast, so it gets no cards of its own.")
else:
    sam_line = "SAM test slice (lean-v12.16): same version string and the same episodes as the PPM podcast rows (1:1 per its README), so it gets no cards of its own."

def stat_rows():
    out = []
    for lab, sub, n, u, a, tk, wd in [
        ("PPM podcast", "A + B trees, rows", rows, pu, pa_, pt, pw),
        ("&nbsp;&nbsp;A · SAM 700k", "rows", R["by_dataset"]["A"]["rows"], R["by_dataset"]["A"]["user_ok"], R["by_dataset"]["A"]["assistant_ok"], None, None),
        ("&nbsp;&nbsp;B · SAM intelligent", "rows", R["by_dataset"]["B"]["rows"], R["by_dataset"]["B"]["user_ok"], R["by_dataset"]["B"]["assistant_ok"], None, None),
        ("Aloe handover", "all buckets, segments", segs, au, aa, at, aw),
        ("&nbsp;&nbsp;none-crossbleed", "segments the Aloe FTs use", nn_seg, nn_u, nn_a, None, None)]:
        tkc = f'{mean(tk, "user")} / {mean(tk, "assistant")}' if tk else ""
        wdc = f'{mean(wd, "user")} / {mean(wd, "assistant")}' if wd else ""
        out.append(f'<tr><th scope="row">{lab}<span class="sub">{sub}</span></th><td>{f0(n)}</td>'
                   f'<td>{f0(u)}<span class="sub">{pct(u, n)}</span></td><td>{f0(a)}<span class="sub">{pct(a, n)}</span></td>'
                   f'<td>{tkc}</td><td>{wdc}</td></tr>')
    return "\n".join(out)

SECT = {"Persona", "Starting context", "Earlier conversation memory"}

def body(t):
    lines = t.split("\n")
    return "\n".join(f'<span class="sec">{e(l)}</span>' if l.strip() in SECT else e(l) for l in lines)

def card(i, x):
    ds = "ppm" if x["dataset"].startswith("PPM") else "aloe"
    sub = (f'tree {x["tree"]} · {"SAM 700k" if x["tree"] == "A" else "SAM intelligent"}' if ds == "ppm" else f'bucket {x.get("bucket")}')
    who = f'<p class="who"><span class="lbl">Speaker label</span> {e(x["who"])}</p>' if x.get("who") else ""
    return f'''<article class="card" id="ex{i + 1}">
<header class="ch"><span class="n">{i + 1:02d}</span><span class="ds ds-{ds}">{e(x["dataset"].replace(" A", "").replace(" B", ""))}</span>
<span class="side side-{x["side"]}">{x["side"]}</span><span class="meta">{sub} · chunk {x["chunk"]} · {f0(x["tokens"])} tokens · {f0(x["words"])} words</span></header>
<p class="key">{e(x["key"])}</p>{who}
<pre class="txt">{body(x["text"])}</pre></article>'''

ppm_ex = [x for x in EX if x["dataset"].startswith("PPM")]
aloe_ex = [x for x in EX if not x["dataset"].startswith("PPM")]
cards_ppm = "\n".join(card(i, x) for i, x in enumerate(ppm_ex))
cards_aloe = "\n".join(card(i + len(ppm_ex), x) for i, x in enumerate(aloe_ex))
st_p, st_a = S["section_tokens"], A["section_tokens"]
vers = ", ".join(f"{e(str(v))}" for v, _ in A["versions"][:2])

H = f'''<title>Training Personas</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; headline, one stats block, then example cards grouped by dataset */
:root {{
  --bg: #f4f6f8; --panel: #ffffff; --ink: #18202a; --muted: #5a6573; --rule: #dde2e8; --accent: #1f5f8b;
  --user: #0f7a6c; --user-bg: #e2f3f0; --asst: #9a5a00; --asst-bg: #fbeedb; --sec: #1f5f8b; --pre-bg: #f8fafb;
  --f-display: "Instrument Sans", "Helvetica Neue", Arial, sans-serif;
  --f-body: "Instrument Sans", "Helvetica Neue", Arial, sans-serif;
  --f-read: "Newsreader", Georgia, "Times New Roman", serif;
  --f-mono: "JetBrains Mono", ui-monospace, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #0f1318; --panel: #171c23; --ink: #e4e9ef; --muted: #98a3b1; --rule: #2a323d; --accent: #7cb6e0;
  --user: #5fd0bf; --user-bg: #12302c; --asst: #f0b260; --asst-bg: #382811; --sec: #8cc0e6; --pre-bg: #12171d; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #0f1318; --panel: #171c23; --ink: #e4e9ef; --muted: #98a3b1; --rule: #2a323d; --accent: #7cb6e0;
  --user: #5fd0bf; --user-bg: #12302c; --asst: #f0b260; --asst-bg: #382811; --sec: #8cc0e6; --pre-bg: #12171d; color-scheme: dark; }}
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
</style>
<div class="wrap">
<header>
<p class="eyebrow">Speech training data · system prompts</p>
<h1>Training Personas</h1>
<p class="headline">{headline}</p>
</header>

<section class="stats" aria-labelledby="st">
<h2 id="st">How many rows carry a persona</h2>
<div class="tbl"><table>
<thead><tr><th>Dataset</th><th>Total</th><th>User persona</th><th>Assistant persona</th><th>Avg tokens u / a</th><th>Avg words u / a</th></tr></thead>
<tbody>
{stat_rows()}
</tbody></table></div>
<p class="note">Both sides: {f0(pb)} PPM rows ({pct(pb, rows)}). A prompt has up to three parts: <b>Persona</b> (facts about the speaker, some real, some invented),
<b>Starting context</b> (who they are talking to and where), and from the second 10-minute chunk on, <b>Earlier conversation memory</b>.
Mean tokens per part, PPM: persona {mean(st_p, "persona")}, context {mean(st_p, "starting_context")}, memory {mean(st_p, "memory(chunk>0)")} (chunk ≥ 1).
Aloe: persona {mean(st_a, "persona")}, context {mean(st_a, "starting_context")}, memory {mean(st_a, "memory(chunk>0)")}.
Tokens are Qwen3-32B counts stored with each prompt. Median PPM length {f0(pt["all"]["median"])} tokens (p10 {f0(pt["all"]["p10"])}, p90 {f0(pt["all"]["p90"])}); chunk 0 averages {mean(pt, "chunk0")}, later chunks {mean(pt, "chunk>0")}.</p>
<div class="lists">
<div><h3>Top names · PPM podcast</h3><ol class="rank">{chips(ppm_names)}</ol><p class="cap">first name, per episode side, {f0(eps)} sampled</p></div>
<div><h3>Top roles · PPM podcast</h3><ol class="rank">{chips([r for r in roles if r[0] != "other"] + occ, 10)}</ol><p class="cap">host / co-host / guest from the speaker label and starting context, then occupation words in the label</p></div>
<div><h3>Top names · Aloe</h3><ol class="rank">{chips(aloe_names)}</ol><p class="cap">first name, per segment side</p></div>

</div>
<p class="note">Aloe has no host or guest: {pct(dict(rel).get("just met", 0), sum(v for _, v in rel))} of its starting contexts say the two speakers have just met on a call app. {pct(ac.get("persona_empty", 0), ac.get("persona_empty", 0) + ac.get("persona_nonempty", 0))} of Aloe chunk prompts have an empty Persona part and carry only context and memory.</p>
<p class="note">{e(sam_line)}</p>
</section>

<nav class="jump" aria-label="Examples"><a href="#ppm">PPM podcast examples ({len(ppm_ex)})</a><a href="#aloe">Aloe examples ({len(aloe_ex)})</a></nav>

<section class="group" id="ppm" aria-labelledby="h-ppm">
<h2 id="h-ppm">PPM podcast · the prompts PPM and PPM-Voxtral train with</h2>
<p class="note">Random status-ok rows from the persona parquets (lean-v12.16), half tree A (SAM 700k), half tree B (SAM intelligent), one per episode. The speaker label is the identity the persona was written for.</p>
{cards_ppm}
</section>

<section class="group" id="aloe" aria-labelledby="h-aloe">
<h2 id="h-aloe">Aloe handover · call-app conversations</h2>
<p class="note">Random chunk prompts ({vers}), one per segment, mostly from the none-crossbleed bucket the Aloe fine-tunes use. SPEAKER_00 is the assistant, SPEAKER_01 the user.</p>
{cards_aloe}
</section>

<footer>PPM row counts: full sidecar report ({f0(rows)} rows). PPM lengths, names and roles: {S["files_sampled"]} of the persona parquet shards ({f0(S["counts"]["user_ok"] + S["counts"]["assistant_ok"])} ok prompts). Aloe: every manifest segment ({f0(segs)}). Built 2026-10-24.</footer>
</div>
'''
open(f"{D}/index.html", "w").write(H)
print("wrote", len(H))

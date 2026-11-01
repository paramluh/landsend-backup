"""actors/actor_stats.json (+ actors/audio/samples.json if present) -> actors/page/index.html (+ audio files to publish alongside)."""
import html
import json
import os
import shutil

D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
TOP = 30
TAG = os.environ.get("ACT_TAG", "")
st = json.load(open(f"{D}/actor_stats{TAG}.json"))
A = st["actors"]
import glob  # noqa: E402
longc = {}
NM = json.load(open(f"{D}/name_map.json")) if os.path.exists(f"{D}/name_map.json") else {}


def nm(aid):
    v = NM.get(aid) or {}
    return v.get("name")


def heard(aid):
    v = NM.get(aid) or {}
    return ", ".join(n for n, _ in (v.get("alts") or [])[:2])


CL = f"{D}/audio_v3{TAG}"
if os.path.exists(f"{CL}/clips.json"):
    for s in json.load(open(f"{CL}/clips.json"))["clips"]:
        longc[s["rank"]] = s
os.makedirs(f"{D}/page/audio", exist_ok=True)

H = lambda s: s / 3600  # noqa: E731
sp = [H(a["train"].get("speech_s", 0)) for a in A]
tot = sum(sp)
lane_tot = sum(H(a["train"].get("lane_s", 0)) for a in A)
share = lambda k: 100 * sum(sp[:k]) / tot  # noqa: E731
n1h = sum(1 for x in sp if x >= 1)
ev_rows = st["rows"]["eval"]
ev_actors = sum(1 for a in A if a["eval"])
ev_h = sum(H(a["eval"].get("speech_s", 0)) for a in A)
step = 2 if sp[0] <= 10 else 10
mx = (int(sp[0] / step) + 1) * step

bars = []
for i, a in enumerate(A[:TOP], 1):
    h = sp[i - 1]
    bars.append(f'<div class="bl"><span class="rk">{i}</span><span class="bn">{html.escape(nm(a["actor_id"]) or "unknown")}</span><span class="id">{html.escape(a["actor_id"][:8])}</span></div>'
                f'<div class="bt"><div class="bf" style="width:{100 * h / mx:.2f}%"></div></div><div class="bv">{h:.1f}</div>')
ticks = "".join(f'<span style="left:{100 * t / mx:.2f}%">{t}</span>' for t in range(0, mx + 1, step))
p10, p30 = share(10), share(TOP)

cards = []
for i, a in enumerate(A[:TOP], 1):
    t, e = a["train"], a["eval"]
    ra, ru = t.get("rows_assistant", 0), t.get("rows_user", 0)
    nmi = nm(a["actor_id"])
    s = longc.get(i)
    if s and s["file"] and os.path.exists(f'{CL}/{s["file"]}'):
        assert s["actor_id"] == a["actor_id"]
        shutil.copy(f'{CL}/{s["file"]}', f'{D}/page/audio/{s["file"]}')
        lc = s["lane_convs"]
        kmin = min(x["sim"] for x in s["segments"])
        au = (f'<div class="voice">Voice = {html.escape(nmi or "this speaker")} (<code>{html.escape(a["actor_id"][:8])}</code>) only, from their own lane: '
              f'S00 (assistant) in {lc["SPEAKER_00"]} conversations, S01 (user) in {lc["SPEAKER_01"]}</div>'
              f'<div class="smp"><audio controls preload="none" src="audio/{s["file"]}"></audio>'
              f'<span class="sm">{s["speech_s"]:.0f} s · {s["n_segments"]} segments · all match one voice (ECAPA cosine ≥ {kmin:.2f} to their centroid)</span></div>')
    else:
        au = '<div class="sm pend">60 s clip decoding, added in a few minutes.</div>'
    evs = f' · eval {e.get("rows", 0)} rows' if e else ""
    nml = (f'<span class="cn">{html.escape(nmi)}</span>' if nmi else
           f'<span class="cn unk">name unknown</span>' + (f'<span class="sm"> heard as {html.escape(heard(a["actor_id"]))}</span>' if heard(a["actor_id"]) else ""))
    cards.append(f'''<article class="card" data-q="{html.escape(((nmi or "") + " " + heard(a["actor_id"]) + " " + a["actor_id"]).lower())}">
<div class="ch"><span class="crk">#{i}</span>{nml}</div>
<code class="cid">{html.escape(a["actor_id"])}</code>
<div class="big">{sp[i - 1]:.1f} h <span class="u">own speech</span></div>
<div class="meta">{t.get("rows", 0)} conversations · {H(t.get("lane_s", 0)):.1f} h on their lane{evs}</div>
<div class="split" title="assistant lane {ra} · user lane {ru}"><div class="sa" style="width:{100 * ra / max(1, ra + ru):.1f}%"></div></div>
<div class="sl"><span><i class="dot a"></i>assistant lane {ra}</span><span><i class="dot b"></i>user lane {ru}</span></div>
{au}
</article>''')

all_rows = [[i, nm(a["actor_id"]) or "", "" if nm(a["actor_id"]) else heard(a["actor_id"]), a["actor_id"], round(H(a["train"].get("speech_s", 0)), 3),
             a["train"].get("rows", 0), round(H(a["eval"].get("speech_s", 0)), 3), a["eval"].get("rows", 0)] for i, a in enumerate(A, 1)]
all_json = json.dumps(all_rows, separators=(",", ":")).replace("</", "<\\/")
n_named = sum(1 for r in all_rows if r[1])
page = f'''<title>Aloe FT Actors</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* layout: one reading column; headline, concentration strip, top-30 bars, then a grid of actor cards with their own-lane clips */
:root {{ --bg:#f6f7f5; --panel:#ffffff; --fg:#1b201d; --muted:#5b655f; --line:#dde2de; --a:#2f6f5e; --b:#c07a2e; --track:#e8ece9;
  --body:"Instrument Sans",system-ui,-apple-system,"Segoe UI",sans-serif; --mono:"JetBrains Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#121614; --panel:#1a1f1c; --fg:#e6ebe8; --muted:#9aa49e; --line:#2b322e;
  --a:#62b8a0; --b:#e2a45e; --track:#252b28; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#121614; --panel:#1a1f1c; --fg:#e6ebe8; --muted:#9aa49e; --line:#2b322e; --a:#62b8a0; --b:#e2a45e; --track:#252b28;
  color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--body); font-size:15px; line-height:1.5; }}
.wrap {{ max-width:1080px; margin:0 auto; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:32px 56px; display:flex; flex-direction:column; gap:28px; }}
h1 {{ font-size:30px; font-weight:700; letter-spacing:-.01em; margin:0; text-wrap:balance; }}
h2 {{ font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin:0 0 12px; }}
.lede {{ font-size:19px; max-width:62ch; margin:8px 0 0; text-wrap:pretty; }}
.lede b {{ font-weight:600; }}
.sub {{ color:var(--muted); margin:6px 0 0; max-width:70ch; }}
.strip {{ display:flex; height:22px; border-radius:4px; overflow:hidden; background:var(--track); }}
.strip div {{ height:100%; }}
.s1 {{ background:var(--a); }} .s2 {{ background:color-mix(in srgb,var(--a) 45%,var(--track)); }}
.sk {{ display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:8px; color:var(--muted); font-size:13px; font-variant-numeric:tabular-nums; }}
.sk i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }}
.chart {{ display:grid; grid-template-columns:auto 1fr 3.2em; gap:4px 10px; align-items:center; font-variant-numeric:tabular-nums; }}
.bl {{ display:flex; gap:8px; font-size:12px; white-space:nowrap; }}
.rk {{ color:var(--muted); width:1.6em; text-align:right; }}
.id {{ font-family:var(--mono); }}
.bt {{ height:12px; background:var(--track); border-radius:2px; min-width:0; }}
.bf {{ height:100%; background:var(--a); border-radius:2px; }}
.bv {{ font-size:12px; text-align:right; }}
.axis {{ grid-column:2; position:relative; height:16px; font-size:11px; color:var(--muted); }}
.axis span {{ position:absolute; transform:translateX(-50%); }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; display:flex; flex-direction:column; gap:6px;
  min-width:0; }}
.ch {{ display:flex; gap:10px; align-items:baseline; min-width:0; }}
.crk {{ font-weight:700; color:var(--a); font-variant-numeric:tabular-nums; }}
.cid {{ font-family:var(--mono); font-size:12px; color:var(--muted); overflow-wrap:anywhere; }}
.big {{ font-size:24px; font-weight:600; font-variant-numeric:tabular-nums; }}
.u {{ font-size:13px; font-weight:400; color:var(--muted); }}
.meta {{ font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.split {{ height:6px; border-radius:3px; background:var(--b); overflow:hidden; margin-top:2px; }}
.sa {{ height:100%; background:var(--a); }}
.sl {{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }}
.dot.a {{ background:var(--a); }} .dot.b {{ background:var(--b); }}
.smp {{ display:flex; flex-direction:column; gap:2px; margin-top:4px; }}
.smp audio {{ width:100%; height:32px; }}
.sm {{ font-size:11.5px; color:var(--muted); }}
.pend {{ font-style:italic; margin-top:4px; }}
.cn {{ font-size:18px; font-weight:600; }} .cn.unk {{ font-weight:400; font-style:italic; color:var(--muted); font-size:15px; }}
.bn {{ min-width:5.5em; }}
.search {{ display:flex; flex-direction:column; gap:8px; }}
.search input {{ font:inherit; font-size:16px; padding:9px 12px; border:1px solid var(--line); border-radius:6px; background:var(--panel); color:var(--fg); max-width:420px; }}
.search input:focus-visible {{ outline:2px solid var(--a); outline-offset:1px; }}
.res {{ display:flex; flex-direction:column; gap:2px; font-size:13.5px; font-variant-numeric:tabular-nums; max-height:320px; overflow:auto; }}
.res div {{ display:grid; grid-template-columns:3.5em minmax(6em,10em) 1fr; gap:10px; padding:3px 0; border-bottom:1px solid var(--line); }}
.res .u8 {{ font-family:var(--mono); font-size:12px; color:var(--muted); overflow-wrap:anywhere; }}
.voice {{ font-size:12.5px; margin-top:6px; padding-top:8px; border-top:1px solid var(--line); }}
.notes {{ font-size:13px; color:var(--muted); max-width:78ch; border-top:1px solid var(--line); padding-top:14px; display:flex; flex-direction:column; gap:6px; }}
.notes p {{ margin:0; }}
code {{ font-family:var(--mono); font-size:.92em; }}
</style>
<div class="wrap">
<header>
<h1>Aloe FT Actors</h1>
<p class="lede"><b>{len(A):,} actors</b> speak in the {st["rows"]["train"]:,} training conversations, for <b>{tot:,.0f} h</b> of their own speech.
The top 10 actors hold <b>{p10:.0f}%</b> of those hours and the top {TOP} hold {p30:.0f}%. The top actor alone has <b>{sp[0]:.1f} h</b>.</p>
<p class="sub">{n1h} actors have at least 1 h; the other {len(A) - n1h:,} share the long tail.
Eval split: {ev_rows} conversations, {ev_actors} actors, {ev_h:.1f} h (not counted above).</p>
</header>
<section>
<h2>Share of speech hours</h2>
<div class="strip"><div class="s1" style="width:{p10:.2f}%"></div><div class="s2" style="width:{p30 - p10:.2f}%"></div></div>
<div class="sk"><span><i class="s1"></i>top 10 · {sum(sp[:10]):.0f} h</span><span><i class="s2"></i>ranks 11–{TOP} · {sum(sp[10:TOP]):.0f} h</span>
<span><i style="background:var(--track);border:1px solid var(--line)"></i>{len(A) - TOP:,} other actors · {sum(sp[TOP:]):.0f} h</span></div>
</section>
<section>
<h2>Own-speech hours, top {TOP}</h2>
<div class="chart">{"".join(bars)}<div class="axis">{ticks}</div></div>
</section>
<section class="search">
<h2>Find a speaker</h2>
<input id="q" type="search" placeholder="Name or id, e.g. Kate" autocomplete="off" aria-label="Find a speaker by name or id">
<div class="res" id="res"></div>
</section>
<section>
<h2>Top {TOP} actors</h2>
<div class="grid">{"".join(cards)}</div>
</section>
<footer class="notes">
<p><b>Actor</b> = <code>speaker_00_account_uid</code> / <code>speaker_01_account_uid</code> from the Aloe handover manifest, joined to each training row by its
conversation key. This is the field that tracks one voice. The manifest's <code>actor_id</code> field does not: its 12 largest ids each cover 20–27
different accounts whose voices do not match (ECAPA cosine 0.13 across accounts vs 0.6–0.7 within one), so they are shared role ids, not people. Lane SPEAKER_00 is the assistant lane and SPEAKER_01 the user lane. The FT trains each row in both orientations, so the lane is
the data's label, not who the model plays.</p>
<p><b>Own speech</b> = time inside the actor's word spans on their lane (store word timings), summed over the first-chunk rows the FT trains on
(each row is the first 10 min of a conversation). "On their lane" = full row length ({lane_tot:,.0f} h across all actors).</p>
<p><b>Clip</b> = about 60 s of the actor's own lane, decoded from the SAM-isolated Mimi codes the FT trains on. Segments are cut at the actor's own word timings
(2.4–10 s), are skipped when the partner has any word within 1 s, and come at most 2 per conversation. Every segment is checked with an ECAPA speaker
embedding against the actor's voice centroid; segments under cosine 0.5 are dropped.</p>
<p><b>Names</b> = the name each person goes by in their own conversations, from the per-conversation speaker identification in the Aloe prompt
files (<code>system_prompts/SPEAKER_NN/episode.json</code>, <code>profile.a_speaker</code>). The handover data has no name field per account, so a
name is shown only when one name covers at least 60 % of that account's named conversations (and at least 2 of them); otherwise "name unknown"
with the names heard. {n_named:,} of {len(A):,} speakers have a name.</p>
</footer>
</div>
<script>
const ALL = {all_json};
const q = document.getElementById("q"), res = document.getElementById("res"), cards = [...document.querySelectorAll(".card")];
function esc(x) {{ return String(x).replace(/[&<>"]/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}}[c])); }}
function run() {{
  const v = q.value.trim().toLowerCase();
  cards.forEach(c => {{ c.hidden = !!v && !c.dataset.q.includes(v); }});
  if (!v) {{ res.innerHTML = ""; return; }}
  const hit = ALL.filter(r => (r[1] + " " + r[2] + " " + r[3]).toLowerCase().includes(v));
  const h = hit.reduce((s, r) => s + r[4], 0);
  res.innerHTML = `<p class="sm">${{hit.length}} speaker${{hit.length === 1 ? "" : "s"}} · ${{h.toFixed(2)}} h of own speech in train</p>` +
    hit.slice(0, 60).map(r => `<div><span>#${{r[0]}}</span><span>${{esc(r[1] || "name unknown")}}${{r[1] ? "" : (r[2] ? " (" + esc(r[2]) + ")" : "")}}</span>` +
      `<span>${{r[4].toFixed(2)}} h · ${{r[5]}} conv${{r[6] ? " · eval " + r[6].toFixed(2) + " h / " + r[7] : ""}} <span class="u8">${{esc(r[3])}}</span></span></div>`).join("");
}}
q.addEventListener("input", run);
</script>
'''
open(f"{D}/page/index.html", "w").write(page)
print("ok", len(A), round(tot, 1), round(p10, 1), round(sp[0], 2), len(longc))

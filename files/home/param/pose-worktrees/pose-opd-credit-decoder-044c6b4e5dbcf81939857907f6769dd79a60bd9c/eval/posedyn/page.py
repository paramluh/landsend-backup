"""10-27 pose-dynamics test page: per actor, the REAL pose (the onestep run: real pose forced on every frame) vs a FREE rollout at the
playground's pose top-k (cb0 30, light 151) vs a FREE rollout with pose top-k OFF; all CFG off, T 1, real speech + words forced, seed 101.
Headline numbers = means over every finished run in out/results.json. Videos ride as base64 and are turned into blob URLs on load.
  python page.py -> page.html"""
import base64
import html
import json
import statistics as st

D = "/data/home/amartyavarma/ppm/posedyn"
R = json.load(open(f"{D}/out/results.json"))
ACTORS = [("01_ab63d745", "Actor 01"), ("03_b437950e", "Actor 03"), ("00_14b594aa", "Actor 00"), ("06_74807fe0", "Kate")]
SEED = 101


def mean(rs, k, side="m"):
    return st.mean(r[side][k] for r in rs) if rs else float("nan")


done = sorted(set(r["person"] for r in R))
full = [p for p in done if len([r for r in R if r["person"] == p]) == 8]
RR = [r for r in R if r["person"] in full]
fd = [r for r in RR if r["mode"] == "free" and r["topk"] == "default"]
fo = [r for r in RR if r["mode"] == "free" and r["topk"] == "off"]
re_ = [r for r in RR if r["mode"] == "onestep" and r["topk"] == "default"]
H = {"real": (mean(re_, "per_min", "real"), mean(re_, "moving", "real")), "def": (mean(fd, "per_min"), mean(fd, "moving")),
     "off": (mean(fo, "per_min"), mean(fo, "moving"))}

blobs, cards = [], []
for pid, name in ACTORS:
    if pid not in full:
        continue
    cells = []
    for key, run, lab, sub in (("real", f"{pid}_default_{SEED}_onestep", "Real pose", "tracked from the video"),
                               ("def", f"{pid}_default_{SEED}_free", "Generated, playground top-k", "cb0 top-k 30, light 151"),
                               ("off", f"{pid}_off_{SEED}_free", "Generated, top-k off", "plain T 1 sampling")):
        rs = [r for r in RR if r["person"] == pid and r["mode"] == ("onestep" if key == "real" else "free")
              and r["topk"] == ("off" if key == "off" else "default")]
        g = mean(rs, "per_min", "real" if key == "real" else "m")
        bid = f"v{len(blobs)}"
        blobs.append((bid, base64.b64encode(open(f"{D}/out/runs/{run}/stick.mp4", "rb").read()).decode()))
        cells.append(f'<figure class="cell {key}"><video id="{bid}" controls playsinline preload="metadata"></video>'
                     f'<figcaption><b>{lab}</b><span>{sub}</span><span class="num">{g:.0f} gestures/min</span></figcaption></figure>')
    cards.append(f'<section class="card"><h2>{html.escape(name)}</h2><div class="row">{"".join(cells)}</div></section>')

page = f"""<title>Pose Top-k Test</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* layout: one headline block, then one card per actor with three stick-figure videos side by side (stacked on phones) */
:root {{ --bg:#f5f6f8; --panel:#ffffff; --fg:#1b1f27; --mute:#5d6573; --line:#dfe3ea; --real:#3d7be0; --def:#b06a1c; --off:#1e9a6a;
        --f-display:"Archivo",system-ui,sans-serif; --f-body:"IBM Plex Sans",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,monospace }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#12151b; --panel:#1a1e26; --fg:#e7eaf0; --mute:#98a1b0; --line:#2a303b;
        --real:#6fa0f0; --def:#e09a4c; --off:#4cc795; color-scheme:dark }} }}
:root[data-theme="dark"] {{ --bg:#12151b; --panel:#1a1e26; --fg:#e7eaf0; --mute:#98a1b0; --line:#2a303b; --real:#6fa0f0; --def:#e09a4c; --off:#4cc795;
        color-scheme:dark }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--f-body); line-height:1.5;
       padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); padding-block:24px 48px }}
main {{ max-width:1080px; margin:0 auto; display:grid; gap:20px }}
h1 {{ font-family:var(--f-display); font-size:1.7rem; line-height:1.2; margin:0; text-wrap:balance }}
.lede {{ max-width:68ch; margin:0; color:var(--mute) }}
.lede b {{ color:var(--fg) }}
.nums {{ display:flex; flex-wrap:wrap; gap:10px 28px; font-family:var(--f-num); font-variant-numeric:tabular-nums }}
.nums span {{ display:flex; gap:8px; align-items:baseline }}
.nums i {{ font-style:normal; font-size:1.5rem; font-weight:600 }}
.k-real {{ color:var(--real) }} .k-def {{ color:var(--def) }} .k-off {{ color:var(--off) }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; display:grid; gap:12px }}
.card h2 {{ font-family:var(--f-display); font-size:1.05rem; margin:0 }}
.row {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px }}
@media (max-width:640px) {{ .row {{ grid-template-columns:1fr }} }}
figure {{ margin:0; display:grid; gap:6px; min-width:0 }}
video {{ width:100%; max-width:100%; aspect-ratio:270/480; max-height:440px; background:#181818; border-radius:6px; border-top:3px solid var(--c) }}
.real {{ --c:var(--real) }} .def {{ --c:var(--def) }} .off {{ --c:var(--off) }}
figcaption {{ display:grid; font-size:.85rem }}
figcaption span {{ color:var(--mute) }}
figcaption .num {{ font-family:var(--f-num); color:var(--c) }}
.foot {{ font-size:.8rem; color:var(--mute); max-width:80ch }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;600&display=swap">
<main>
<h1>Top-k truncation is what makes the generated pose stiff</h1>
<p class="lede">Same model, no CFG, temperature 1, real speech and words. With the playground's pose top-k the hands barely move.
With top-k off, free generation moves about as much as the real tracked pose. <b>All {len(full)} actors with a voice, 2 seeds each.</b> Kate is the exception and stays stiff either way.</p>
<div class="nums">
<span class="k-real"><i>{H['real'][0]:.0f}</i>real gestures/min</span>
<span class="k-def"><i>{H['def'][0]:.0f}</i>playground top-k</span>
<span class="k-off"><i>{H['off'][0]:.0f}</i>top-k off</span>
</div>
{"".join(cards)}
<p class="foot">Gestures = wrist-speed strokes, the same detector used on the real data. Per-actor numbers average both seeds; the videos show seed {SEED}.
Real pose comes from the run that forces the tracked pose on every frame. The first 1.9 s of each generated clip is the real-pose prefill.</p>
</main>
{"".join(f'<script type="text/plain" id="b{b}">{d}</script>' for b, d in blobs)}
<script>
for (const v of document.querySelectorAll('video')) {{
  try {{
    const s = atob(document.getElementById('b' + v.id).textContent.trim());
    const a = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
    v.src = URL.createObjectURL(new Blob([a], {{type: 'video/mp4'}}));
  }} catch (e) {{ v.replaceWith(Object.assign(document.createElement('p'), {{textContent: 'Video failed to load.'}})); }}
}}
</script>
"""
open(f"{D}/page.html", "w").write(page)
print("full actors", full, {k: [round(x, 2) for x in v] for k, v in H.items()}, "bytes", len(page))

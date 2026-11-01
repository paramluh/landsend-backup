"""Rewrites klweb/ppm_aloe_posemono_candidates/index.html from candidates.json + the renders present on disk (idempotent; run under flock
after each render lands). Cards carry ONLY non-distilled 9-sinks renders (user, 10-19): r/cNN.jpg = the rendered starting image (top),
r/cNN.mp4 = the rendered clip (real pose + audio, first 5 s, studio identity STUDIO[(NN - 1) % 10]) once ready. No stick figures. URL unchanged.
   python3 cand_page.py"""
import html
import json
import os

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
OUT = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates"
STUDIO = ["03_b437950e", "05_e25c6859", "06_74807fe0", "09_d9ef3c94", "10_27286bdd", "12_72cb45d3", "13_956c09af", "15_d1e1322c",
          "18_1ffa9083", "19_47473a46"]
E = html.escape
cards = json.load(open(f"{W}/candidates.json"))
ex = lambda p: os.path.exists(f"{OUT}/{p}")  # noqa: E731
n_s = n_r = n_f = 0
items = []
for c in cards:
    k = c["num"]
    nn = f"c{k:02d}"
    rk = c["row_key"].split("/")
    idn = STUDIO[(k - 1) % 10]
    vis = []
    # FIRST on every card (user, 10-19): the REAL camera recording of the card's speaker (realsrc_fetch.py: the original Aloe app footage,
    # src/cNN.mp4 = the card's 20 s audio window with its audio, poster src/cNN.jpg = its first frame, shown without pressing play).
    real = ""
    if ex(f"src/{nn}.mp4") and ex(f"src/{nn}.jpg"):
        n_f += 1
        real = (f'<div class="real"><p class="rl">Real recording (original camera)</p><video controls preload="none" playsinline poster="src/{nn}.jpg" '
                f'src="src/{nn}.mp4"></video><p class="vc">{E(c["side"])} camera · first 20 s of the row (same window as the audio below), with its audio</p></div>')
    else:
        real = '<div class="real"><p class="rl">Real recording (original camera)</p><div class="wait">real recording pending</div></div>'
    # Then the video-model renders of the real pose (non-distilled 9-sinks). No stick figures, no s/ links, no other visuals.
    if ex(f"r/{nn}.jpg"):
        n_r += 1
        vis.append(f'<img class="start" src="r/{nn}.jpg" alt="video-model starting image"><p class="vc">Video-model render of the real pose · starting image · identity {E(idn)}</p>')
        if ex(f"r/{nn}.mp4"):
            n_s += 1
            vis.append(f'<video controls preload="none" playsinline poster="r/{nn}.jpg" src="r/{nn}.mp4"></video><p class="vc">Video-model render of the real pose · 0–5 s with audio</p>')
    else:
        vis.append(f'<div class="wait">video-model render pending · identity {E(idn)}</div>')
    items.append(f'''<article class="card"><div class="top"><span class="num">{k}</span><div class="id"><b>{E(rk[2][:8])}/{E(rk[3])}/{E(rk[4])}</b>
<span>{E(c["side"])} ({E(c["speaker"])}) · track {c["track_s"] // 60} min {c["track_s"] % 60:02d} s · window: first 60 s · persona: none</span>
<span>video identity if picked: studio prompt {E(idn)} (studio-studio)</span></div></div>
{real}
<p class="rl">Video-model render of the real pose</p>
<div class="vis">{"".join(vis)}</div>
<audio controls preload="none" src="{c["audio"]}"></audio><p class="tx">{E(c["snippet"]) or "<i>(no words in the first 20 s)</i>"}</p></article>''')
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pose Eval Candidates</title>
<style>
:root{{--bg:#f5f6f8;--panel:#fff;--fg:#17202c;--muted:#5f6b7a;--line:#dde2e9;--b:#2563c9;--vbg:#000}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#10141a;--panel:#171c24;--fg:#e6eaf0;--muted:#97a3b3;--line:#2a323d;--b:#6b9cf0;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#10141a;--panel:#171c24;--fg:#e6eaf0;--muted:#97a3b3;--line:#2a323d;--b:#6b9cf0;color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:24px 16px 48px;display:grid;gap:14px}}h1{{margin:0;font-size:24px}}
.cap{{margin:0;color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;display:grid;gap:8px;min-width:0}}
.top{{display:flex;gap:10px;align-items:flex-start}}.num{{flex:none;width:34px;height:34px;border-radius:50%;background:var(--b);color:#fff;display:grid;place-items:center;font-weight:700}}
.id{{display:grid;font-size:13px;min-width:0}}.id b{{font-family:ui-monospace,monospace;font-size:13px;overflow-wrap:anywhere}}.id span{{color:var(--muted)}}
.vis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;align-items:start}}
.vis img,.vis video,.vis .wait{{width:100%;aspect-ratio:256/480;object-fit:contain;background:var(--vbg);border-radius:6px;border:1px solid var(--line)}}
.vis .wait{{display:grid;place-items:center;color:#97a3b3;font-size:12px;text-align:center;padding:8px}}
.vis .vc{{margin:0;font-size:11.5px;color:var(--muted);grid-column:auto}}
.real{{display:grid;gap:4px}}.real video,.real .wait{{width:100%;max-height:640px;aspect-ratio:9/16;object-fit:contain;background:var(--vbg);border-radius:6px;border:1px solid var(--line)}}
.real .wait{{display:grid;place-items:center;color:#97a3b3;font-size:12px}}.real .vc{{margin:0;font-size:11.5px;color:var(--muted)}}
.rl{{margin:0;font-size:13px;font-weight:650}}
audio{{width:100%;height:34px}}.tx{{margin:0;font-size:13.5px;max-height:7.5em;overflow-y:auto;overflow-wrap:anywhere}}
</style></head><body><div class="wrap">
<h1>Pose eval candidates: pick 5</h1>
<p class="cap">20 held-out Aloe handover pose tracks from the eval split of the PPM 32B monoplex pose fine-tune (aloe-pose-mono-8020). Stratified by
speaker side (10 assistant, 10 user), one track per conversation, all with a voice clip available. Each card: first, the real recording from the
speaker's original camera (the row's first 20 s, with its audio); below it, the video-model render of the real pose: the starting image and a short
rendered clip (first 5 s) of the row's real pose and audio through the non-distilled 9-sinks video model, with the studio identity the row would get,
then the first 20 s of audio and its words. Generation uses no persona
text (persona: none on every row); every row gets the speaker's voice clip. The eval window is the first 60 s. Reply with 5 numbers.</p>
<div class="grid">
{chr(10).join(items)}
</div></div></body></html>'''
open(f"{OUT}/index.html.tmp", "w").write(page)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
print(f"cand page: real recordings {n_f}/20, rendered images {n_r}/20, rendered clips {n_s}/20", flush=True)

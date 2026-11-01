#!/usr/bin/env python3
"""Aloe FT comparison: ONE training-loss curve per run (train normalised word CE on B, smoothed), all v2-recipe Aloe FTs.
Reads the two live page JSONs (ppm_aloe_v2init_live.json, ppm_aloe_v2vox_live.json); writes aloe_ft_compare.html next to them."""
import json, os, time, html

K = "/data/home/amartyavarma/setup-logs/klweb"


def load(p):
    try:
        return json.load(open(os.path.join(K, p)))
    except Exception:
        return {}


def nword(train):
    return [(int(r[0]), float(r[2])) for r in (train or []) if r and r[2] is not None]


runs = []
ti = load("ppm_aloe_v2init_live.json")
tv = load("ppm_aloe_v2vox_live.json")
COL = {"12829": "#0e7490", "11488": "#2563c9", "12529": "#0b7a6e", "10351": "#9333ea", "8869": "#d97706", "6242": "#dc2626"}
if ti.get("ref_train"):
    runs.append(("PPM · w-11488 · text A (v2, 22383)", "11488", False, nword(ti["ref_train"]), False))
for r in ti.get("runs", []):
    init = str(r.get("init"))
    if init == "11488" and ti.get("ref_train"):
        continue                                                     # = the v2 reference (22383), already added above
    fam = "INTL" if init == "6242" else "PPM"
    runs.append((f"{fam} · w-{init} · text A", init, False, nword(r.get("train")), r.get("state") == "RUNNING"))
for key, init in (("v", "11488"), ("r", "11488"), ("n", "6242")):
    b = tv.get(key) or {}
    if b.get("train"):
        name = b.get("run") or key
        lab = {"v": "PPM · w-11488 + ASR w-1972 · Voxtral A (loudness-norm, 22420)",
               "r": "PPM · w-11488 + ASR w-1972 · Voxtral A raw (23081, live demo)",
               "n": "INTL · w-6242 + ASR w-480 · Voxtral A"}[key]
        runs.append((lab, init, True, nword(b["train"]), b.get("state") == "RUNNING" or b.get("live") is True))


# si20k-vox (10-18 PT): the spoken-instruct 20k TTS fine-tunes (Voxtral A, B clip 0.8 / persona 0.2), read from each run's newest metrics.jsonl
import glob
PODEV = []                                                           # si20k runs' podcast eval (pretraining eval_speech rows, native A)
for rdir, init, lab in (("/data/home/amartyavarma/ppm/si20k-vox-ppm/tests/pagetest/ppm", "12829", "PPM · w-12829 + ASR w-1972 · si20k + podcast 1:1"),
                        ("/data/home/amartyavarma/ppm/si20k-vox-ppm/tests/pagetest/intl", "6242", "INTL · w-6242 + ASR w-480 · si20k + podcast 1:1")):
    ms = sorted(glob.glob(f"{rdir}/runs/*-j*/metrics.jsonl"), key=os.path.getmtime)
    if not ms:
        continue
    pts, pe = [], []
    for ln in open(ms[-1]):
        try:
            d = json.loads(ln)
        except Exception:
            continue
        if d.get("train_speech/word_ce_norm") is not None:
            pts.append((int(d["step"]), float(d["train_speech/word_ce_norm"])))
        if d.get("eval_pod_speech/word_ce_norm") is not None:
            pe.append((int(d["step"]), float(d["eval_pod_speech/word_ce_norm"])))
    if pe:
        PODEV.append((lab, init, pe))
    if pts:
        runs.append((lab, init, "si20k", pts, time.time() - os.path.getmtime(ms[-1]) < 600))


def smooth(pts, w=15):
    out = []
    for i in range(len(pts)):
        lo = max(0, i - w + 1)
        seg = [p[1] for p in pts[lo:i + 1]]
        out.append((pts[i][0], sum(seg) / len(seg)))
    return out


W, H, L, R, T, B = 960, 520, 64, 16, 20, 44
allv = [v for _, _, _, pts, _ in runs for _, v in smooth(pts)]
ymin, ymax = (min(allv), max(allv)) if allv else (1.7, 2.1)
pad = (ymax - ymin) * 0.06
ymin, ymax = ymin - pad, ymax + pad
xmax = max([253] + [p[-1][0] for _, _, _, p, _ in runs if p])
X = lambda s: L + (W - L - R) * (s - 1) / (xmax - 1)
Y = lambda v: T + (H - T - B) * (ymax - v) / (ymax - ymin)
svg = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="training normalised word CE per run">']
for i in range(6):
    v = ymin + (ymax - ymin) * i / 5
    svg.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g"/><text x="{L-6}" y="{Y(v)+4:.1f}" class="ax" text-anchor="end">{v:.3f}</text>')
for s in sorted(set([1] + list(range(50, xmax + 1, 50)) + [253, xmax])):
    svg.append(f'<text x="{X(s):.1f}" y="{H-B+18}" class="ax" text-anchor="middle">{s}</text>')
svg.append(f'<text x="{(L+W-R)/2}" y="{H-6}" class="ax" text-anchor="middle">step</text>')
rows = []
for lab, init, vox, pts, live in runs:
    sp = smooth(pts)
    if not sp:
        continue
    d = " ".join(f"{'M' if j == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for j, (s, v) in enumerate(sp))
    c = COL.get(init, "#555")
    dash = 'stroke-dasharray="1.5 3.5" stroke-linecap="round"' if vox == "si20k" else ("" if vox else 'stroke-dasharray="6 4"')
    sw = 2.6 if live else 1.8
    svg.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="{sw}" {dash}><title>{html.escape(lab)}</title></path>')
    rows.append((lab, c, vox, live, sp[-1][0], sp[-1][1]))
svg.append("</svg>")
DASH = 'stroke-dasharray="6 4"'
DOT = 'stroke-dasharray="1.5 3.5" stroke-linecap="round"'
STAR = "★ "
leg = "".join(f'<tr><td><svg width="34" height="10"><line x1="0" x2="34" y1="5" y2="5" stroke="{c}" stroke-width="2.5" {DOT if vox == "si20k" else ("" if vox else DASH)}/></svg></td>'
              f'<td>{STAR if live else ""}{html.escape(lab)}</td><td class="n">step {s}</td><td class="n">{v:.4f}</td></tr>' for lab, c, vox, live, s, v in rows)
now = time.strftime("%a %m-%d %H:%M PT", time.localtime())
page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>Aloe FT Comparison</title><style>
:root{{--bg:#f4f5f7;--fg:#151b23;--mu:#5b6673;--card:#fff;--line:#d8dde3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#10141a;--fg:#e4e8ee;--mu:#9aa4b1;--card:#171c24;--line:#2a313c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}.w{{max-width:1000px;margin:0 auto;padding:20px 16px}}
.c{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}}.g{{stroke:var(--line)}}.ax{{fill:var(--mu);font-size:12px}}
table{{border-collapse:collapse;width:100%;margin-top:10px}}td{{padding:4px 6px;border-bottom:1px solid var(--line)}}.n{{text-align:right;font-variant-numeric:tabular-nums;color:var(--mu)}}
</style></head><body><div class="w"><h2 style="margin:0 0 4px">Aloe fine-tunes: training loss, one curve per run</h2>
<div style="color:var(--mu)">Train normalised word CE on the assistant side (B), {15}-step moving average · Aloe runs: same v2 data, schedule and targets · solid = Voxtral latents on A, dashed = text (words) on A · dotted = si20k + podcast 1:1 runs (spoken-instruct 20k TTS rows with Voxtral A, B voice clip 0.8 / persona 0.2, co-trained 1:1 with unseen pretraining podcast rows; curve = the si20k rows; one pass in their own step count) · colour = starting checkpoint · ★ = running · updated {now}, reloads every 60 s</div>
<div class="c" style="margin-top:12px">{''.join(svg)}<table>{leg}</table></div></div></body></html>"""
if PODEV:
    pv = [v for _, _, p_ in PODEV for _, v in p_]
    y0, y1 = min(pv), max(pv); pd_ = max((y1 - y0) * 0.1, 0.01); y0, y1 = y0 - pd_, y1 + pd_
    H2 = 300; xm2 = max([xmax] + [p_[-1][0] for _, _, p_ in PODEV])
    X2 = lambda s_: L + (W - L - R) * s_ / max(xm2, 1)  # noqa: E731
    Y2 = lambda v: T + (H2 - T - B) * (y1 - v) / (y1 - y0)  # noqa: E731
    sv2 = [f'<svg viewBox="0 0 {W} {H2}" width="100%" role="img" aria-label="podcast eval normalised word CE per run">']
    for i in range(5):
        v = y0 + (y1 - y0) * i / 4
        sv2.append(f'<line x1="{L}" x2="{W-R}" y1="{Y2(v):.1f}" y2="{Y2(v):.1f}" class="g"/><text x="{L-6}" y="{Y2(v)+4:.1f}" class="ax" text-anchor="end">{v:.3f}</text>')
    for s_ in sorted(set([0] + list(range(50, xm2 + 1, 50)) + [xm2])):
        sv2.append(f'<text x="{X2(s_):.1f}" y="{H2-B+18}" class="ax" text-anchor="middle">{s_}</text>')
    leg2 = ""
    for lab, init, p_ in PODEV:
        c = COL.get(init, "#555")
        d_ = " ".join(f"{'M' if j == 0 else 'L'}{X2(s_):.1f},{Y2(v):.1f}" for j, (s_, v) in enumerate(p_))
        sv2.append(f'<path d="{d_}" fill="none" stroke="{c}" stroke-width="2.2"><title>{html.escape(lab)}</title></path>')
        sv2 += [f'<circle cx="{X2(s_):.1f}" cy="{Y2(v):.1f}" r="3" fill="{c}"/>' for s_, v in p_]
        leg2 += f'<tr><td><svg width="34" height="10"><line x1="0" x2="34" y1="5" y2="5" stroke="{c}" stroke-width="2.5"/></svg></td><td>{html.escape(lab)}</td><td class="n">step {p_[-1][0]}</td><td class="n">{p_[-1][1]:.4f}</td></tr>'
    sv2.append("</svg>")
    page = page.replace("</div></body></html>", f"""<h3 style="margin:18px 0 4px">Podcast eval (pretraining eval_speech rows, native A): normalised word CE</h3>
<div style="color:var(--mu)">si20k + podcast co-training runs only; step 0 = the init; lower = pretraining ability kept</div>
<div class="c" style="margin-top:8px">{''.join(sv2)}<table>{leg2}</table></div></div></body></html>""")
tmp = "/data/home/amartyavarma/ppm/si20k-vox-ppm/tests/pagetest/.t.tmp"
open(tmp, "w").write(page)
os.replace(tmp, "/data/home/amartyavarma/ppm/si20k-vox-ppm/tests/pagetest/aloe_ft_compare_test.html")
print("wrote", len(rows), "runs")

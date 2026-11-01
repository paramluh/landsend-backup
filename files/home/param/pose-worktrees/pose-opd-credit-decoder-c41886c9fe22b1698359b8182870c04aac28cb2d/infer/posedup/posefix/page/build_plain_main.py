#!/usr/bin/env python3
"""main's PLAIN comparison page (10-25): data.json -> index.html. Direct <video src="a/...">, no fetch/blob (viewer-safe), white flash +
beep at the prefill -> generated switch (user: "flash bang when switch"). usage: python3 build_plain_main.py data.json index.html"""
import html, json, re, sys
d = json.load(open(sys.argv[1])); m = d['meta']; E = html.escape
def sc(x): return f"cb0 repeat {x['repeat']}% · out of frame {x['oof']}% · conf {x['conf']:.2f}" if x else ''
ref = m.get('ref', {}); groups = m['groups']
def key(gi):
    l = groups[gi]['label']
    return (0 if 'full prefill' in l.lower() or 'words' in l.lower() else 1 if ('151' in l and 'CFG' not in l) else 2 if l == 'Pose prefill 2 s' else 3, gi)
order = sorted(range(len(groups)), key=key)
out = []
for gi in order:
    g = groups[gi]; cs = [c for c in d['convs'] if c['group'] == gi]
    if not cs: continue
    h = g.get('head', {})
    out.append(f'<section id="g{gi}"><h2>{E(g["label"])}</h2><p class="hl">'
               + (f'<span><b>Monoplex</b> {E(sc(h.get("mono")))}</span>' if h.get('mono') else '')
               + (f'<span><b>Duplex</b> {E(sc(h.get("dupM")))}</span>' if h.get('dupM') else '') + '</p>')
    for c in cs:
        ck = c['ck'][sorted(c['ck'])[-1]]
        cells = []
        for v in ck['versions']:
            vid = (v.get('video') or {}).get('B', {}).get('src')
            b = v['tracks'][1]['src'] if len(v['tracks']) > 1 else None
            lab = v['label'].split(' (')[0]
            np_s = float(v.get('np_s') or 0)
            s = sc(v.get('score')) if v.get('score') else ('real recording' if v['kind'] == 'gt' else '')
            media = (f'<div class="vw"><video controls playsinline preload="metadata" data-np="{np_s:.2f}" src="{E(vid)}"></video><div class="flash"></div></div>'
                     if vid else '<div class="nov">no avatar video</div>')
            aud = f'<audio controls preload="none" data-np="{np_s:.2f}" src="{E(b)}"></audio>' if b else ''
            sw = f'<br><span class="s">switch to generated at {np_s:.2f} s</span>' if np_s > 0 else ''
            cells.append(f'<figure class="{E(v["kind"])}">{media}<figcaption><b>{E(lab)}</b><br><span class="s">{E(s)}</span>{sw}</figcaption>{aud}</figure>')
        pf = c.get('prefill_label') or (f'prefill {c["np_frames"]/12.5:.2f} s' if c.get('np_frames') else 'no prefill')
        extra = (' · ' + E(c['arm'])) if c.get('arm') else ''
        out.append(f'<article><h3>{E(c["title"])} <span class="s">· {E(pf)}{extra}</span></h3><div class="row">{"".join(cells)}</div></article>')
    out.append('</section>')
nav = ' '.join(f'<a href="#g{gi}">{E(groups[gi]["label"])}</a>' for gi in order if any(c['group'] == gi for c in d['convs']))
CSS = open(__file__.replace('build_plain_main.py', 'plain_main.css')).read()
JS = open(__file__.replace('build_plain_main.py', 'plain_main.js')).read()
page = f'''<title>Pose Monoplex vs Duplex</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
{CSS}
</style>
<div class="wrap">
<header><h1>Pose Monoplex vs Duplex</h1>
<p class="s">Same prefill for both models, then generated. A white flash and a beep mark the switch from prefill to generated. Avatar renders with no post-processing. Real reference: held-out Aloe {E(sc(ref.get("aloe")))}; Person 01's own clip {E(sc(ref.get("p01")))}.</p>
<nav>{nav}</nav></header>
{"".join(out)}
</div>
<script>
{JS}
</script>
'''
open(sys.argv[2], 'w').write(page); print(len(page))

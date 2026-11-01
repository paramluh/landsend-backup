"""xabi-view page (10-27): Xabi's data (Aloe pose release v2 = v2pure stores) as GT cards = the pose samples pages' GT cell
(s20 build_page.py / sam8b-eval-cfg format: same CSS tokens, figure cards, .vv video, playback-synced .tr transcript, .pp list) + the
klweb dark-default theme snippet + one raw-row-layout panel. Reads OUT/data.json (build.py), writes OUT/page/index.html + media links.
   python build_page.py OUTDIR"""
import html
import json
import os
import sys

OUT = sys.argv[1]
d = json.load(open(f"{OUT}/data.json"))
os.makedirs(f"{OUT}/page", exist_ok=True)
THEME = open("/data/home/amartyavarma/setup-logs/klweb/theme_snip.html").read()
E = html.escape

CSS = """<style>
/* layout: the pose samples pages' GT-cell cards in a grid (4 / 2 / 1 columns), header box on top, one wide row-layout panel below */
:root{--bg:#f4f5f7;--fg:#151b23;--mu:#5b6673;--card:#fff;--line:#d8dde3;--ac:#2563c9;--chip:#eef1f5;--a:#8a6d1f;--b:#2563c9;--p:#2e8b57;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#10141a;--fg:#e4e8ee;--mu:#9aa4b1;--card:#171c24;--line:#2a313c;--ac:#6b9cf0;--chip:#1f2630;--a:#d6b45a;--b:#7fa8f2;--p:#6cc596;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#10141a;--fg:#e4e8ee;--mu:#9aa4b1;--card:#171c24;--line:#2a313c;--ac:#6b9cf0;--chip:#1f2630;--a:#d6b45a;--b:#7fa8f2;--p:#6cc596;color-scheme:dark}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
.w{max-width:1500px;margin:0 auto;padding-block:20px 40px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.hdr{display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap}.hdr h2{margin:0;text-wrap:balance}
.m{color:var(--mu);font-size:13.5px}
.cfgbox{font-size:13.5px;line-height:1.55;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 12px;margin:10px 0}
.cfgbox b{font-weight:600}.cfgbox div{margin:3px 0}
.q{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:12px}
@media (max-width:1200px){.q{grid-template-columns:repeat(2,minmax(0,1fr))}}@media (max-width:640px){.q{grid-template-columns:1fr}}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;min-width:0;display:flex;flex-direction:column;gap:6px}
figcaption{font-size:13.5px;font-weight:600;display:flex;gap:6px;align-items:baseline;flex-wrap:wrap}
.sp{font-size:11.5px;font-weight:700;padding:1px 7px;border-radius:5px;border:1px solid var(--line);color:var(--mu)}.sp.ev{color:var(--ac);border-color:var(--ac)}
.vv{display:flex;justify-content:center}.vv video{width:100%;max-width:100%;height:auto;max-height:62vh;object-fit:contain;border-radius:6px;background:#000}
.meta{font-size:12px;color:var(--mu);font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.meta code{font-family:var(--mono);font-size:11.5px}
.tr{font-size:12.5px;line-height:1.55;max-height:8.5em;overflow:auto;padding:4px 7px;border:1px solid var(--line);border-radius:6px}
.tr .th{font-size:11.5px;color:var(--mu);margin-bottom:2px}.tr .w{border-radius:3px;padding:0 1px;transition:background .12s;cursor:pointer}
.tr .w.past{opacity:.55}.tr .w.cur{background:var(--ac);color:#fff;opacity:1}
.pp{font-size:12px;color:var(--mu);margin:0;padding:0;list-style:none;line-height:1.5}.pp li{border-bottom:1px solid var(--line);padding:1px 0;display:flex;justify-content:space-between;gap:10px}
.pp li b{font-weight:500;text-align:right;color:var(--fg);font-variant-numeric:tabular-nums;overflow-wrap:anywhere;min-width:0}
details.ip{margin:16px 0 0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 12px}details.ip summary{cursor:pointer;font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0;font-family:var(--mono);font-size:12px}
.chips span{background:var(--chip);border:1px solid var(--line);border-radius:4px;padding:1px 5px;white-space:nowrap}.chips span.vc{border-color:var(--b);color:var(--b)}
.lx{overflow-x:auto;border:1px solid var(--line);border-radius:6px;margin:6px 0}
.lx table{border-collapse:collapse;font-family:var(--mono);font-size:11.5px;font-variant-numeric:tabular-nums}
.lx th,.lx td{padding:2px 5px;border-right:1px solid var(--line);text-align:center;white-space:nowrap}
.lx th{position:sticky;left:0;background:var(--card);text-align:left;font-weight:600;font-family:system-ui,sans-serif;font-size:11.5px;z-index:1}
.lx tr.hd td{color:var(--mu)}.lx tr.ra td{color:var(--a)}.lx tr.rb td{color:var(--b)}.lx tr.rp td{color:var(--p)}.lx tr.gap td,.lx tr.gap th{border-top:2px solid var(--line)}
.lx td.nw{font-weight:700}.lx td.e{opacity:.35}
.fn{margin:4px 0 0;font-size:12px;color:var(--mu)}
</style>"""

JS = """<script>
(function(){function sync(v){var fg=v.closest('figure');if(!fg)return;var tr=fg.querySelector('.tr[data-sync="1"]');if(!tr)return;var t=v.currentTime;
tr.querySelectorAll('.w[data-s]').forEach(function(w){var s=+w.dataset.s,e=+w.dataset.e;w.classList.toggle('cur',s<=t&&t<Math.max(e,s+0.12));
w.classList.toggle('past',t>=Math.max(e,s+0.12))});var c=tr.querySelector('.w.cur');if(c&&!v.paused){var r=c.offsetTop-tr.offsetTop;
if(r<tr.scrollTop||r>tr.scrollTop+tr.clientHeight-18)tr.scrollTop=r-18}}
function loop(v){if(v.paused||v.ended){sync(v);return}sync(v);requestAnimationFrame(function(){loop(v)})}
document.addEventListener('play',function(e){if(e.target.tagName==='VIDEO')loop(e.target)},true);
['seeked','timeupdate','pause'].forEach(function(n){document.addEventListener(n,function(e){if(e.target.tagName==='VIDEO')sync(e.target)},true)});
document.addEventListener('click',function(e){var w=e.target.closest('.tr .w[data-s]');if(!w)return;var v=w.closest('figure').querySelector('video');if(v){v.currentTime=+w.dataset.s;sync(v)}});
/* blob-load each mp4 (click-to-seek on hosts without HTTP Range), as the samples pages do */
document.querySelectorAll('video[data-src]').forEach(function(v){var u=v.dataset.src;fetch(u).then(function(r){if(!r.ok)throw 0;return r.blob()})
.then(function(b){v.src=URL.createObjectURL(b)}).catch(function(){v.src=u})});
document.addEventListener('play',function(e){if(e.target.tagName!=='VIDEO')return;document.querySelectorAll('video').forEach(function(o){if(o!==e.target)o.pause()})},true);})();
</script>"""


def fmt_s(x):
    return f"{x:.1f} s" if x < 60 else f"{int(x // 60)} min {x % 60:.0f} s"


def card(c):
    ev = c["split"] == "eval"
    words = c["words"]
    sp = []
    for k, w in enumerate(words):
        e = min(w["e"], words[k + 1]["s"] if k + 1 < len(words) else w["e"]) if w["e"] > w["s"] else w["s"] + 0.3
        sp.append(f'<span class="w" data-s="{w["s"]:.2f}" data-e="{max(e, w["s"] + 0.08):.2f}" title="{w["s"]:.2f}–{w["e"]:.2f} s · frame {w["f"]}">{E(w["t"])}</span>')
    tr = (f'<div class="tr" data-sync="1"><div class="th">B word lane source: {len(words)} words, clip-local timing '
          f'(highlighted with playback; click a word to seek)</div>{" ".join(sp) or "<i>(no words)</i>"}</div>')
    vc = c["voice"]
    vtxt = (f'yes: {vc[0]["frames"][1] - vc[0]["frames"][0]} frames ({(vc[0]["frames"][1] - vc[0]["frames"][0]) / 12.5:.1f} s) of '
            f'{E(vc[0]["src_key"].rsplit("/", 1)[-1])}' if vc else ("drawn, none found" if c["voice_drawn"] else "no (2 of 3 rows)"))
    items = [("store row", f'p{c["store"]} · row {c["row"]}'), ("active speaker", f'{c["speaker"]} (data side {c["side"]})'),
             ("position in recording", f'clip {int(c["row_key"][-5:]) + 1} of {c["n_clips_rec"]} · {c["clip_start_sec"]:.2f}–{c["clip_end_sec"]:.2f} s'),
             ("voice clip in system block", vtxt), ("trainer positions", f'P {c["P"]} + {c["n_frames"]} frames + CD 2 = {c["T"]}')]
    pp = '<ul class="pp">' + "".join(f"<li><span>{E(k)}</span><b>{E(str(v))}</b></li>" for k, v in items) + "</ul>"
    meta = (f'<div class="meta">rec <code>{E(c["rid"])}</code> · {fmt_s(c["dur_s"])} ({c["n_frames"]} frames) · '
            f'{"held-out" if ev else "train"} · source {int(c["orig_w"])}×{int(c["orig_h"])}</div>')
    return (f'<figure><figcaption>Ground truth <span class="sp{" ev" if ev else ""}">{"held-out" if ev else "train"}</span>'
            f'<span class="m">{fmt_s(c["dur_s"])}</span></figcaption>{meta}'
            f'<div class="vv"><video controls preload="metadata" playsinline data-src="media/{c["tag"]}.mp4" '
            f'style="aspect-ratio:{c["canvas"][0]}/{c["canvas"][1]}"></video></div>{tr}{pp}</figure>')


def layout_panel(L, clip):
    P, conv = L["P"], L["conv"]
    ch = []
    k = 0
    sysb = L["sys"]
    while k < len(sysb):
        s = sysb[k]
        if s["mimi"]:
            j = k
            while j < len(sysb) and sysb[j]["mimi"]:
                j += 1
            v = L["voice"][0] if L["voice"] else None
            src = f' of {E(v["src_key"].rsplit("/", 1)[-1])} frames {v["frames"][0]}–{v["frames"][1]}' if v else ""
            ch.append(f'<span class="vc">[{j - k} voice-clip positions: lane PAD + B Mimi codes{src}]</span>')
            k = j
            continue
        ch.append(f'<span title="pos {s["pos"]} · id {s["id"]}">{E(s["tok"])}</span>')
        k += 1
    head = ("<tr class='hd'><th>position</th>" + "".join(f"<td>{c['pos']}</td>" for c in conv) + "</tr>"
            "<tr class='hd'><th>frame · time</th>" + "".join(f"<td>{c['f']} · {c['t']:.2f}</td>" for c in conv) + "</tr>")

    def cell(v):
        return '<td class="e">·</td>' if v < 0 else f"<td>{v}</td>"
    ra = ("<tr class='ra gap'><th>A lane (empty)</th>" + "".join(f"<td>{'PAD' if c['a_id'] == 151670 else c['a_id']}</td>" for c in conv) + "</tr>"
          "<tr class='ra'><th>A Mimi / pose</th>" + "".join('<td class="e">·</td>' for _ in conv) + "</tr>")
    rb = "<tr class='rb gap'><th>B word lane</th>" + "".join(
        f"<td class='{'nw' if c['b_tok'] == 'NW' else ('e' if c['b_tok'] == 'PAD' else '')}'>{E(c['b_tok'])}</td>" for c in conv) + "</tr>"
    rows = []
    for nm, key, cls in (("B Mimi", "mimi", "rb"), ("B pose", "pose", "rp")):
        for cb in (0, 1, 2):
            rows.append(f"<tr class='{cls}{' gap' if cb == 0 else ''}'><th>{nm} cb{cb}</th>" + "".join(cell(c[key][cb]) for c in conv) + "</tr>")
        rows.append(f"<tr class='{cls}'><th>⋮ cb3–30</th>" + "".join('<td class="e">⋮</td>' for _ in conv) + "</tr>")
        rows.append(f"<tr class='{cls}'><th>{nm} cb31</th>" + "".join(cell(c[key][31]) for c in conv) + "</tr>")
    tbl = f'<div class="lx"><table>{head}{ra}{rb}{"".join(rows)}</table></div>'
    return (f'<details class="ip" open><summary>Raw row layout: one clip as the trainer sees it ({E(clip["rid"][:8])}… '
            f'{fmt_s(clip["dur_s"])}, {"held-out" if clip["split"] == "eval" else "train"}, first {len(conv)} conversation positions ≈ 2.1 s)</summary>'
            f'<p class="m" style="margin:6px 0 2px">System block, positions 0–{P - 1} (lane B; channel A is PAD throughout):</p>'
            f'<div class="chips">{"".join(ch)}</div>'
            f'<p class="m" style="margin:8px 0 2px">Conversation, positions {P}–{P + len(conv) - 1} (one position per 80 ms frame). '
            f'Codes shown local (0–2047), · = none:</p>{tbl}'
            f'<p class="fn">The word lane is at position P + f for frame f; each word is a NW (NEW_WORD) token on the frame before its start, '
            f'then its Qwen3-32B pieces. A word that starts at frame 0 has no NW, since its marker would fall on frame −1; that is the first word of {eq["first_piece_at_frame0_no_NW"]} of {eq["n"]} held-out clips (clips are cut at a word onset), as here. Mimi and pose frame f sit at P + 2 + f (delay CD = 2), so the first 2 positions carry text only. '
            f'Loss is on B only: the lane (pieces, NW, PAD, EOB), Mimi cb0 + cb1–31 (speech decoder), pose cb0 + cb1–31 (pose decoder). '
            f'This row has {L["T"]} positions in total (P {P} + {L["n"]} frames + 2).</p></details>')


cn = d["counts"]
clips = d["clips"]
L = d["layout"]
lclip = next(c for c in clips if c["tag"] == L["tag"]) if L else None
eq = d["eval_quirks"]
hdr = f"""<div class="hdr"><h2>Xabi's data: Aloe pose release v2 (v2pure)</h2></div>
<p class="m" style="margin:4px 0 0">What the monoplex pose fine-tunes train on. 8 real store rows ({sum(c['split'] == 'eval' for c in clips)} held-out, {sum(c['split'] == 'train' for c in clips)} train; one recording each), decoded from the stored codes as the samples pages' ground-truth cell does: Mimi speech codes → audio, SAM pose codes → stick figure, muxed.</p>
<div class="cfgbox">
<div><b>Clips.</b> One store row per active-speaker clip. Train {cn['by_split']['train']:,} clips · {cn['frames']['train'] / 1e6:.2f}M frames · 2,047 h · {cn['recs']['train']:,} recordings. Held-out {cn['by_split']['eval']:,} clips (3,481 used in eval) · {cn['frames']['eval']:,} frames · 19.3 h · {cn['recs']['eval']} recordings (recording-level split: blake2b8 rule). Clip length median {cn['median_s']:.1f} s (10th–90th percentile {cn['p10_s']:.1f}–{cn['p90_s']:.1f} s, longest {fmt_s(cn['max_s'])}). 12.5 frames/s.</div>
<div><b>System block.</b> <code>&lt;|im_start|&gt;system\\n … &lt;|im_end|&gt;\\n</code>. On 1 row in 3 (hash rule) it holds a 3–10 s voice clip of the same speaker, cut from another clip of the same recording (Mimi codes on lane PAD; of {eq['n']} held-out rows, {eq['voice_clip'] + eq['voice_drawn_no_clip']} drew one, {eq['voice_clip']} got one, {eq['voice_drawn_no_clip']} found none). No persona on any row.</div>
<div><b>Channel B</b> = the clip's active speaker: word lane (Qwen3-32B pieces, each word led by a NEW_WORD timing token on the frame before it, except a word at frame 0), 32-codebook Mimi speech codes, 32-codebook SAM pose codes (tokenizer pose-tokenizer-32cb-200k-noshift, frame-0 root + shoulder-width normalisation).</div>
<div><b>Channel A</b> is empty: no partner in this data (lane PAD, no audio, no pose, present 0).</div>
</div>"""
page = (f'<title>Xabi pose data v2pure</title><meta name="claude:viewport-fit-sides" content="cover">{THEME}{CSS}<div class="w">{hdr}'
        f'<div class="q">{"".join(card(c) for c in clips)}</div>'
        f'<p class="fn">Stick figure: the GT cell\'s drawing (body blue, left hand green, right hand orange, face grey; line alpha = joint confidence), '
        f'25 fps (2 pose frames per 12.5 Hz step), canvas at the source frame\'s aspect.</p>'
        f'{layout_panel(L, lclip) if L else ""}</div>{JS}')
open(f"{OUT}/page/index.html", "w").write(page)
os.makedirs(f"{OUT}/page/media", exist_ok=True)
files = {f"media/{c['tag']}.mp4": f"{OUT}/media/{c['tag']}.mp4" for c in clips}
json.dump(files, open(f"{OUT}/page/files.json", "w"), indent=1)
print("page", len(page), "bytes; media", sum(c["mp4_bytes"] for c in clips), "bytes")

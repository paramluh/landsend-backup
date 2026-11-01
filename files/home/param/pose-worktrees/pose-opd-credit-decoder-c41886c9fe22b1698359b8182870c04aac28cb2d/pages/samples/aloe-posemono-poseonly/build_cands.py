"""aloe-posemono-poseonly CANDIDATES (10-19): 20 held-out Aloe handover pose tracks for the user to pick a fixed 5-row pose eval set.
1-10 = rows R-000..R-009 (rows/rows.json, already generated); 11-20 = the next 10 tracks of the SAME stratified order (build_rows.py:
pool = eval items whose trained clip rule finds B a voice clip; strata = B side, blake2b8 order, interleaved, one track per session),
built here as trainer-parity micros R-010..R-019 (60 s windows, every row B's voice clip, persona on even slots) into rows2/ for pre-generation.
Page klweb/ppm_aloe_posemono_candidates: per candidate number, id, track length, speaker side, persona yes/no, a 20 s audio preview (Mimi
decode of the track's real codes, frames 0-249) and the transcript of those 20 s (the store's word timings; no ASR).
   python build_cands.py      (source aloe-pose-mono-8020/aloe_mono8020.env first; compute node, CPU)"""
import hashlib
import html
import json
import os
import pickle
import subprocess
import sys

import numpy as np

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
OUT = "/data/home/amartyavarma/setup-logs/klweb/ppm_aloe_posemono_candidates"
R2 = f"{W}/rows2"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code")
import data_aloe_pose as A  # noqa: E402
D = A.D
sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402
import imageio_ffmpeg  # noqa: E402
FF = imageio_ffmpeg.get_ffmpeg_exe()
HAND = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files"
N60, NPREV = 750, 250
os.makedirs(R2, exist_ok=True)
os.makedirs(f"{OUT}/a", exist_ok=True)
assert not os.path.exists(f"{R2}/rows.json"), "refusing to overwrite rows2/rows.json"
cfg = A.aloe_cfg()
CFG = {True: dict(cfg, voice_p10=10, persona_p10=10), False: dict(cfg, voice_p10=10, persona_p10=0)}
scan = json.load(open(f"{W}/rows/scan.json"))
rows1 = json.load(open(f"{W}/rows/rows.json"))
cand = [x for x in scan if x["clip"] and x["n"] >= N60]


def h8(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


strata = {s: sorted([x for x in cand if x["side"] == s], key=lambda x: h8(f"posemono-poseonly|{x['row_key']}|{x['side']}")) for s in ("assistant", "user")}
order = []
for i in range(max(len(v) for v in strata.values())):
    for s in ("assistant", "user"):
        if i < len(strata[s]):
            order.append(strata[s][i])
used, sel = set(), []
for x in order:
    if x["session"] in used:
        continue
    used.add(x["session"])
    sel.append(x)
    if len(sel) == 20:
        break
assert [x["row_key"] for x in sel[:10]] == [r["row_key"] for r in rows1], "slots 1-10 must be R-000..R-009"
st = A.Stores(A.STORES)
off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
mimi = I.Mimi("cpu")
cards, rows2 = [], []
for j, it in enumerate(sel):
    node, rows_, sib, ps, pz, b_ok = st.open(it["store"])
    row = rows_[it["node_row"]]
    side = it["side"]
    codes = np.asarray(row[f"codes_{side}"])[:, :NPREV].astype(np.int64)
    wav = mimi.decode(codes)
    num = j + 1
    fa = f"{OUT}/a/c{num:02d}.mp3"
    subprocess.run([FF, "-y", "-hide_banner", "-loglevel", "error", "-f", "f32le", "-ar", "24000", "-ac", "1", "-i", "-", "-c:a", "libmp3lame",
                    "-b:a", "96k", fa + ".tmp.mp3"], input=np.clip(wav, -1, 1).astype(np.float32).tobytes(), check=True)
    os.replace(fa + ".tmp.mp3", fa)
    ws = row[f"words_{side}"]
    tk = "word" if "word" in ws[0] else "text"
    snip = " ".join(str(w[tk]).strip() for w in ws if float(w["start"]) < NPREV / 12.5)
    with_p = j % 2 == 0
    if j < 10:
        r1 = rows1[j]
        assert r1["with_persona"] == with_p
        rid = r1["row"]
    else:
        rid = f"R-{j:03d}"
        m = st.conv((it["store"], it["node_row"], it["swap"]), CFG[with_p], 0)
        pv = json.loads(m["prov"])
        P = int(m["P"])
        assert pv["chan_A"] == "empty" and len(pv["clips"]) == 1 and (len(pv["persona"]) > 0) == with_p, pv
        L = P + N60 + D.CD
        w = {"P": P, "ids": m["ids"][:, :L].copy(), "depth": m["depth"][:, :L].copy(), "present": m["present"][:, :L].copy(),
             "pdepth": m["pdepth"][:L].copy(), "pid": m["pid"][:L].copy(), "row_key": m["row_key"], "item": m["item"], "prov": m["prov"],
             "swap": m["swap"], "kind": m["kind"]}
        sds = [f"{HAND}/{bk}/{it['session']}/{it['seg']}" for bk in ("none", "SPEAKER_00", "SPEAKER_01", "both", "unknown")]
        sds = [d for d in sds if os.path.exists(f"{d}/w00.{it['speaker']}.pose_tokens.npy")]
        assert len(sds) == 1, sds
        hp = np.load(f"{sds[0]}/w00.{it['speaker']}.pose_tokens.npy")[:, :N60].astype(np.int64).T
        hc = np.load(f"{sds[0]}/w00.{it['speaker']}.codes.npy")[:, :N60].astype(np.int64).T
        assert it["frame_start"] == 0 and np.array_equal(w["pdepth"][P + D.CD:P + D.CD + N60] - off, hp) and \
            np.array_equal(w["depth"][1, P + D.CD:P + D.CD + N60] - off, hc), it["row_key"]
        ptext = None
        if with_p:
            pers = ps.get(it["node_row"], it["row_key"])
            D._tok_init()
            av = D.PT.avail(pers.get(D.PER.SIDES[1 ^ it["swap"]]))
            ptext = D.PT.block_text(av, [q for q in D.PT.SECTIONS if q in av])
        cpos = np.flatnonzero(w["depth"][1, :P, 0] >= 0)
        pickle.dump(w, open(f"{R2}/{rid}.pkl.tmp", "wb"))
        os.replace(f"{R2}/{rid}.pkl.tmp", f"{R2}/{rid}.pkl")
        pj = json.load(open(f"{sds[0]}/w00.{it['speaker']}.pose.json"))
        nm = pj["normalization"]
        rows2.append(dict(row=rid, item=m["item"], row_key=it["row_key"], side=side, speaker=it["speaker"], session=it["session"], seg=it["seg"],
                          chunk=it["chunk"], frame_start=0, n=N60, n_seg=it["n"], P=P, T=L, dur_s=N60 / 12.5,
                          prompt="voice clip + persona" if with_p else "voice clip only", with_persona=with_p, persona_text=ptext,
                          persona_tokens=sum(x["tokens"] for x in pv["persona"]), persona_cut=pv["persona_cut"],
                          clip=dict(pv["clips"][0], pos=[int(cpos[0]), int(cpos[-1]) + 1]),
                          window=f"{it['session'][:8]}/{it['seg']} {it['speaker']} chunk {it['chunk']}: frames 0-{N60 - 1} of {it['n']}",
                          norm={"s1": {"frame0_root": nm["root0_yx_normalized"], "frame0_width": nm["shoulder_width0_normalized"],
                                       "orig_w": nm["orig_w"], "orig_h": nm["orig_h"], "lip_lag_ms": pj["alignment"]["lip_sync_best_lag_ms"]}}))
    cards.append(dict(num=num, row=rid, row_key=it["row_key"], side=side, speaker=it["speaker"], track_s=round(it["n"] / 12.5),
                      persona=with_p, snippet=snip, audio=f"a/c{num:02d}.mp3"))
    print(num, rid, it["row_key"], side, with_p, snip[:100], flush=True)
json.dump(rows2, open(f"{R2}/rows.json.tmp", "w"), indent=1)
os.replace(f"{R2}/rows.json.tmp", f"{R2}/rows.json")
json.dump(cards, open(f"{W}/candidates.json", "w"), indent=1)

E = html.escape
items = "\n".join(
    f'''<article class="card"><div class="top"><span class="num">{c["num"]}</span><div class="id"><b>{E(c["row_key"].split("/")[2][:8])}/{E(c["row_key"].split("/")[3])}/{E(c["row_key"].split("/")[4])}</b>
<span>{E(c["side"])} ({E(c["speaker"])}) · track {c["track_s"] // 60} min {c["track_s"] % 60:02d} s · window: first 60 s · persona: {"yes" if c["persona"] else "no"}</span></div></div>
<audio controls preload="none" src="{c["audio"]}"></audio><p class="tx">{E(c["snippet"]) or "<i>(no words in the first 20 s)</i>"}</p></article>'''
    for c in cards)
page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pose Eval Candidates</title>
<style>
:root{{--bg:#f5f6f8;--panel:#fff;--fg:#17202c;--muted:#5f6b7a;--line:#dde2e9;--b:#2563c9}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#10141a;--panel:#171c24;--fg:#e6eaf0;--muted:#97a3b3;--line:#2a323d;--b:#6b9cf0;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#10141a;--panel:#171c24;--fg:#e6eaf0;--muted:#97a3b3;--line:#2a323d;--b:#6b9cf0;color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:24px 16px 48px;display:grid;gap:14px}}h1{{margin:0;font-size:24px}}
.cap{{margin:0;color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;display:grid;gap:8px;min-width:0}}
.top{{display:flex;gap:10px;align-items:flex-start}}.num{{flex:none;width:34px;height:34px;border-radius:50%;background:var(--b);color:#fff;display:grid;place-items:center;font-weight:700}}
.id{{display:grid;font-size:13px;min-width:0}}.id b{{font-family:ui-monospace,monospace;font-size:13px;overflow-wrap:anywhere}}.id span{{color:var(--muted)}}
audio{{width:100%;height:34px}}.tx{{margin:0;font-size:13.5px;max-height:7.5em;overflow-y:auto;overflow-wrap:anywhere}}
</style></head><body><div class="wrap">
<h1>Pose eval candidates: pick 5</h1>
<p class="cap">20 held-out Aloe handover pose tracks from the eval split of the PPM 32B monoplex pose fine-tune (aloe-pose-mono-8020). Stratified by
speaker side (10 assistant, 10 user), one track per conversation, all with a voice clip available. Each card has the first 20 s of the real audio and
its words. "Persona" says whether the eval row gets persona text; every row gets the speaker's voice clip. The eval window is the first 60 s.
Reply with 5 numbers.</p>
<div class="grid">
{items}
</div></div></body></html>'''
open(f"{OUT}/index.html.tmp", "w").write(page)
os.replace(f"{OUT}/index.html.tmp", f"{OUT}/index.html")
print("page ->", OUT, len(cards), "candidates")

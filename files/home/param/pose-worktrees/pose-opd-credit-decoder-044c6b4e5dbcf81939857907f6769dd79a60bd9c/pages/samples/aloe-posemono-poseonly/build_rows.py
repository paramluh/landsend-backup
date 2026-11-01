"""aloe-posemono-poseonly ROWS (10-19): held-out Aloe handover pose tracks of job 24272's eval split (aloe-pose-mono-8020: 531 eval items =
22091 / 22317's qualifying eval tracks) as trainer-parity micros (the run's own code + env: data_aloe_pose.Stores.conv, channel A empty,
B = one speaker's words + Mimi + pose), cut to the window's first NW frames: positions [0, P + NW + CD) (ids / depth / present / pdepth / pid;
training-target arrays dropped). NW = 750 (60 s) for R-000..R-007, 1500 (120 s) for R-008..R-009.
PROMPT (standing inference rule): EVERY row gets B's voice clip; persona text on half the rows. Built with the run's own prefix builder
(podcast_prefix_a_empty) under the run's cfg with only the two draws forced: voice_p10 = 10 (clip always drawn; the clip itself = the
trained data_ppm.voice_clips rule, unchanged) and persona_p10 = 10 (persona row) or 0 (clip-only row).
POOL: eval items with n >= NW for which the trained clip rule finds a voice clip for B (the video identity comes from studio prompts, not
from these app tracks, so no identity constraint). Asserted: the row's B pose / Mimi codes of the window == the handover cut's
w00.<speaker>.pose_tokens / codes at the row's frame_start (7,500 x chunk index).
SELECTION (deterministic, stratified random, independent of any model output): strata = B's data side (assistant SPEAKER_00 / user
SPEAKER_01), order within a stratum by blake2b8('posemono-poseonly|' + row_key + '|' + side); the two strata are interleaved, one track per
session; slots 0..7 = 60 s, slots 8..9 = 120 s (n >= 1500); persona on even slots (needs B persona text), clip only on odd slots.
   python build_rows.py <out dir>      (source aloe-pose-mono-8020/aloe_mono8020.env first; compute node, CPU)"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np

OUT = sys.argv[1]
assert os.environ.get("ALOE_DATA") == "aloe-mono" and os.environ.get("QQA_A_EMPTY") == "1" and os.environ.get("QQA_POSE_B") == "1", \
    "source aloe-pose-mono-8020/aloe_mono8020.env first"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code")
import data_aloe_pose as A  # noqa: E402

D = A.D
HAND = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files/none"
N60, N120 = 750, 1500
SLOTS = [N60] * 8 + [N120] * 2
os.makedirs(OUT, exist_ok=True)
assert not os.path.exists(f"{OUT}/rows.json"), "refusing to overwrite rows.json"
cfg = A.aloe_cfg()
assert A.MONO and cfg["voice_rule"] == "p10" and cfg["persona_rule"] == "p10" and cfg["draw_key"] == "item", cfg
CFG = {True: dict(cfg, voice_p10=10, persona_p10=10), False: dict(cfg, voice_p10=10, persona_p10=0)}
sp = A.splits_v2(A.STORES)
ev = sp["eval"]
assert len(ev) == 531, len(ev)


def h8(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


st = A.Stores(A.STORES)
scan, cand = [], []
for k, r, sw, n, key in ev:
    node, rows_, sib, ps, pz, b_ok = st.open(int(k))
    row = rows_[int(r)]
    side = "user" if sw else "assistant"
    spk = "SPEAKER_01" if sw else "SPEAKER_00"
    _, _, session, seg, ch = key.split("/")
    try:
        pers = ps.get(int(r), row["row_key"])
    except (FileNotFoundError, OSError):
        pers = None
    _, pv = A.podcast_prefix_a_empty(row, int(sw), sib, pers, CFG[True])
    it = dict(store=int(k), node_row=int(r), swap=int(sw), n=int(n), row_key=key, side=side, speaker=spk, session=session, seg=seg, chunk=ch,
              chunk_index=int(row["chunk_index"]), frame_start=int(row["frame_start"]), clip=len(pv["clips"]) > 0, persona=len(pv["persona"]) > 0)
    assert ch == f"c{it['chunk_index']:02d}" and it["frame_start"] == 7500 * it["chunk_index"], (key, it["chunk_index"], it["frame_start"])
    scan.append(it)
    if it["clip"] and n >= N60:
        cand.append(it)
tot = len(scan)
print(json.dumps({"eval_items": tot, "clip_found": sum(x["clip"] for x in scan), "persona_found": sum(x["persona"] for x in scan),
                  "pool": len(cand),
                  "pool_n_ge_1500": sum(x["n"] >= N120 for x in cand), "pool_persona": sum(x["persona"] for x in cand)}), flush=True)
json.dump(scan, open(f"{OUT}/scan.json", "w"), indent=1)

strata = {s: sorted([x for x in cand if x["side"] == s], key=lambda x: h8(f"posemono-poseonly|{x['row_key']}|{x['side']}"))
          for s in ("assistant", "user")}
order = []
for i in range(max(len(v) for v in strata.values())):
    for s in ("assistant", "user"):
        if i < len(strata[s]):
            order.append(strata[s][i])
used, sel = set(), []
for j, nw in enumerate(SLOTS):
    want_p = j % 2 == 0
    pick = next((x for x in order if x["session"] not in used and x["n"] >= nw and (x["persona"] or not want_p)), None)
    assert pick is not None, ("no candidate left for slot", j, nw, want_p)
    used.add(pick["session"])
    sel.append(dict(pick, nw=nw, with_persona=want_p))


def norm_of(sd, spk):
    pj = json.load(open(f"{sd}/w00.{spk}.pose.json"))
    nm = pj["normalization"]
    assert pj["status"] == "ok" and nm["rotation"] == 0
    return {"frame0_root": nm["root0_yx_normalized"], "frame0_width": nm["shoulder_width0_normalized"], "orig_w": nm["orig_w"],
            "orig_h": nm["orig_h"], "lip_lag_ms": pj["alignment"]["lip_sync_best_lag_ms"]}


out = []
off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
for j, it in enumerate(sel):
    m = st.conv((it["store"], it["node_row"], it["swap"]), CFG[it["with_persona"]], 0)
    pv = json.loads(m["prov"])
    P, NW = int(m["P"]), int(it["nw"])
    assert pv["chan_A"] == "empty" and pv["voice_drawn"] == 1 and len(pv["clips"]) == 1, pv
    assert (len(pv["persona"]) > 0) == it["with_persona"] and pv["persona_drawn"] == int(it["with_persona"]), pv
    assert (m["ids"][0] == D.PAD_ID).all() and (m["depth"][0] < 0).all() and (m["present"][0] == 0).all(), "channel A not empty"
    assert m["ids"].shape[1] == P + it["n"] + D.CD, (m["ids"].shape, P, it["n"])
    L = P + NW + D.CD
    w = {"P": P, "ids": m["ids"][:, :L].copy(), "depth": m["depth"][:, :L].copy(), "present": m["present"][:, :L].copy(),
         "pdepth": m["pdepth"][:L].copy(), "pid": m["pid"][:L].copy(), "row_key": m["row_key"], "item": m["item"], "prov": m["prov"],
         "swap": m["swap"], "kind": m["kind"]}
    # the window = the row's first NW frames of B's track = handover cut frames [frame_start, frame_start + NW)
    sds = [f"{HAND[:-5]}/{bk}/{it['session']}/{it['seg']}" for bk in ("none", "SPEAKER_00", "SPEAKER_01", "both", "unknown")]
    sds = [d for d in sds if os.path.exists(f"{d}/w00.{it['speaker']}.pose_tokens.npy")]
    assert len(sds) == 1, ("handover dir of the session (one crossbleed bucket)", it["row_key"], sds)
    sd = sds[0]
    fs = it["frame_start"]
    hp = np.load(f"{sd}/w00.{it['speaker']}.pose_tokens.npy")[:, fs:fs + NW].astype(np.int64).T
    hc = np.load(f"{sd}/w00.{it['speaker']}.codes.npy")[:, fs:fs + NW].astype(np.int64).T
    assert np.array_equal(w["pdepth"][P + D.CD:P + D.CD + NW] - off, hp), ("window pose != handover w00 pose tokens", it["row_key"])
    assert np.array_equal(w["depth"][1, P + D.CD:P + D.CD + NW] - off, hc), ("window Mimi != handover w00 codes", it["row_key"])
    assert (w["pdepth"][:P + D.CD] < 0).all() and (w["depth"][1, P:P + D.CD] < 0).all()
    # B's persona text exactly as the prefix builder renders it (card text) + the clip's positions in the system block
    node, rows_, sib, ps, pz, b_ok = st.open(it["store"])
    ptext = None
    if it["with_persona"]:
        pers = ps.get(it["node_row"], it["row_key"])
        D._tok_init()
        av = D.PT.avail(pers.get(D.PER.SIDES[1 ^ it["swap"]]))
        ptext = D.PT.block_text(av, [q for q in D.PT.SECTIONS if q in av])
    cpos = np.flatnonzero(w["depth"][1, :P, 0] >= 0)
    assert len(cpos) == pv["clips"][0]["frames"][1] - pv["clips"][0]["frames"][0] and (np.diff(cpos) == 1).all(), (len(cpos), pv["clips"])
    row = f"R-{j:03d}"
    pickle.dump(w, open(f"{OUT}/{row}.pkl.tmp", "wb"))
    os.replace(f"{OUT}/{row}.pkl.tmp", f"{OUT}/{row}.pkl")
    rec = dict(row=row, item=m["item"], row_key=it["row_key"], side=it["side"], speaker=it["speaker"], session=it["session"], seg=it["seg"],
               chunk=it["chunk"], frame_start=fs, n=NW, n_seg=it["n"], P=P, T=L, dur_s=round(NW / 12.5, 2),
               prompt="voice clip + persona" if it["with_persona"] else "voice clip only", with_persona=it["with_persona"],
               persona_text=ptext, persona_tokens=sum(x["tokens"] for x in pv["persona"]), persona_cut=pv["persona_cut"],
               clip=dict(pv["clips"][0], pos=[int(cpos[0]), int(cpos[-1]) + 1]),
               window=f"{it['session'][:8]}/{it['seg']} {it['speaker']} chunk {it['chunk']}: frames 0-{NW - 1} of {it['n']}",
               norm={"s1": norm_of(sd, it["speaker"])})
    out.append(rec)
    print(row, {k_: v for k_, v in rec.items() if k_ not in ("persona_text", "norm")}, flush=True)
json.dump(out, open(f"{OUT}/rows.json.tmp", "w"), indent=1)
os.replace(f"{OUT}/rows.json.tmp", f"{OUT}/rows.json")
print("done", len(out), "rows")

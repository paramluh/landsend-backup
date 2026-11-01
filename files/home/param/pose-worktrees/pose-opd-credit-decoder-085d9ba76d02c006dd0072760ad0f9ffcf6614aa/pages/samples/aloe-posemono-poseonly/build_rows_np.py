"""aloe-posemono-poseonly NO-PERSONA rows (10-19, user: "do inferences with no personas"): the 20 candidates (candidates.json, the same
tracks as rows/ R-000..R-009 + rows2/ R-010..R-019) rebuilt as trainer-parity micros with persona text OFF on every row and B's voice clip
ON on every row (run cfg with voice_p10 = 10, persona_p10 = 0; the clip = the trained data_ppm.voice_clips rule, unchanged), all cut to the
first 60 s (750 frames) -> rows-np/R-000..R-019.pkl + rows.json. Asserted: no persona segment, exactly one voice clip, channel A empty, the
window's B pose / Mimi == the handover cut's w00 tokens, and the clip identical to the earlier rows' clip.
   python build_rows_np.py      (source aloe-pose-mono-8020/aloe_mono8020.env first; compute node, CPU)"""
import json
import os
import pickle
import sys

import numpy as np

W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
OUT = f"{W}/rows-np"
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/code")
import data_aloe_pose as A  # noqa: E402
D = A.D
HAND = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files"
N60 = 750
os.makedirs(OUT, exist_ok=True)
assert not os.path.exists(f"{OUT}/rows.json"), "refusing to overwrite rows-np/rows.json"
cfg = dict(A.aloe_cfg(), voice_p10=10, persona_p10=0)
scan = {(x["row_key"], x["side"]): x for x in json.load(open(f"{W}/rows/scan.json"))}   # both sides of a conversation share row_key
old = {r["row"]: r for r in json.load(open(f"{W}/rows/rows.json")) + json.load(open(f"{W}/rows2/rows.json"))}
cards = json.load(open(f"{W}/candidates.json"))
st = A.Stores(A.STORES)
off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
out = []
for c in cards:
    it = scan[(c["row_key"], c["side"])]
    rid = c["row"]
    m = st.conv((it["store"], it["node_row"], it["swap"]), cfg, 0)
    pv = json.loads(m["prov"])
    P = int(m["P"])
    assert pv["chan_A"] == "empty" and pv["voice_drawn"] == 1 and pv["persona_drawn"] == 0 and len(pv["clips"]) == 1 and not pv["persona"], pv
    assert pv["clips"][0]["src_key"] == old[rid]["clip"]["src_key"] and pv["clips"][0]["frames"] == old[rid]["clip"]["frames"], (rid, pv["clips"])
    assert (m["ids"][0] == D.PAD_ID).all() and (m["depth"][0] < 0).all() and (m["present"][0] == 0).all()
    L = P + N60 + D.CD
    w = {"P": P, "ids": m["ids"][:, :L].copy(), "depth": m["depth"][:, :L].copy(), "present": m["present"][:, :L].copy(),
         "pdepth": m["pdepth"][:L].copy(), "pid": m["pid"][:L].copy(), "row_key": m["row_key"], "item": m["item"], "prov": m["prov"],
         "swap": m["swap"], "kind": m["kind"]}
    sds = [f"{HAND}/{bk}/{it['session']}/{it['seg']}" for bk in ("none", "SPEAKER_00", "SPEAKER_01", "both", "unknown")]
    sds = [d for d in sds if os.path.exists(f"{d}/w00.{it['speaker']}.pose_tokens.npy")]
    assert len(sds) == 1 and it["frame_start"] == 0, sds
    hp = np.load(f"{sds[0]}/w00.{it['speaker']}.pose_tokens.npy")[:, :N60].astype(np.int64).T
    hc = np.load(f"{sds[0]}/w00.{it['speaker']}.codes.npy")[:, :N60].astype(np.int64).T
    assert np.array_equal(w["pdepth"][P + D.CD:P + D.CD + N60] - off, hp) and np.array_equal(w["depth"][1, P + D.CD:P + D.CD + N60] - off, hc), rid
    cpos = np.flatnonzero(w["depth"][1, :P, 0] >= 0)
    if not old[rid]["with_persona"]:              # clip-only rows: identical to the earlier micro over the first 60 s
        rd = f"{W}/rows" if int(rid[2:]) < 10 else f"{W}/rows2"
        xo = pickle.load(open(f"{rd}/{rid}.pkl", "rb"))
        assert P == old[rid]["P"] and np.array_equal(w["ids"], xo["ids"][:, :L]) and np.array_equal(w["depth"], xo["depth"][:, :L]) and \
            np.array_equal(w["pdepth"], xo["pdepth"][:L]), ("clip-only row differs from the earlier micro", rid)
    else:
        assert P < old[rid]["P"], (rid, P, old[rid]["P"])
    pickle.dump(w, open(f"{OUT}/{rid}.pkl.tmp", "wb"))
    os.replace(f"{OUT}/{rid}.pkl.tmp", f"{OUT}/{rid}.pkl")
    r = dict(old[rid], P=P, T=L, n=N60, dur_s=N60 / 12.5, prompt="voice clip only (no persona)", with_persona=False, persona_text=None,
             persona_tokens=0, persona_cut=0, clip=dict(pv["clips"][0], pos=[int(cpos[0]), int(cpos[-1]) + 1]), candidate=c["num"],
             window=f"{it['session'][:8]}/{it['seg']} {it['speaker']} chunk {it['chunk']}: frames 0-{N60 - 1} of {it['n']}")
    out.append(r)
    print(c["num"], rid, it["row_key"], "P", P, "clip", len(cpos), flush=True)
json.dump(out, open(f"{OUT}/rows.json.tmp", "w"), indent=1)
os.replace(f"{OUT}/rows.json.tmp", f"{OUT}/rows.json")
print("done", len(out), "no-persona rows")

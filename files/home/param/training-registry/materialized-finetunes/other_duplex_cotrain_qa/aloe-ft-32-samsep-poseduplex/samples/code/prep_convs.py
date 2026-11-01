"""pdsamples PREP (CPU, compute node): held-out conversations for the pose-duplex duo samples page. For each conversation (an EVAL row of the
sam-sep plan = data_aloe_v2.splits 'eval', first chunk, never trained on) and BOTH orientations (swap 0: B = data 'assistant' = speaker 1;
swap 1: B = data 'user' = speaker 2) build the EXACT poseduplex training micro (data_aloe_posedup.Stores.conv, env = run_poseduplex.env)
with ONE inference change: B's voice clip ALWAYS (data_aloe_v2.VOICE_NUM = VOICE_DEN, i.e. the 30 % draw forced on; the clip itself = the
unchanged PPM voice_clips rule, a LATER chunk of the same speaker); no persona (v2 rows have none); channel A = Voxtral of the partner audio.
Only conversations where BOTH sides have pose (POSE_INDEX ok) and both clips exist, n_frames >= NEED + 2.
Per conversation -> <out>/c<i>.npz, direction d = swap d (0 = speaker 1 is B, 1 = speaker 2 is B):
  pre_ids_d [2, P], pre_depth_d [2, P, 32], pre_present_d [2, P], pre_pdepth_d [P, 32]     (system block: B's voice clip)
  laneB_d [NEED], depB_d [NEED, 32], pdepB_d [NEED, 32]   (B lane / Mimi / pose at positions P .. P + NEED - 1, global ids, -1 none)
  stem_d  = speaker d's own audio (24 kHz, raw gain = the A input the OTHER direction trained on) over row frames 0 .. NEED (+ 1 frame)
  gtpose_d [32, NEED] = speaker d's real pose tokens (cb-local), row frame f
plus <out>/c<i>.json (row_key, item, prov per direction, real words per side).
usage: python prep_convs.py <out dir> <n convs> <need frames>"""
import json
import os
import sys

OUT, NCONV, NEED = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
HERE = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/code"
sys.path.insert(0, HERE)
os.environ.setdefault("RANK", "0"); os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np  # noqa: E402
import data_aloe_posedup as DP  # noqa: E402

assert DP.POSE, "POSEDUP_POSE must be 1 (run_poseduplex.env)"
A2, D = DP.A2, DP.D
A2.VOICE_NUM = A2.VOICE_DEN                                       # INFERENCE RULE: B's voice clip on every row (draw forced on)
cfg = DP.aloe_cfg()
sp = DP.splits()
ev = sorted(sp["eval"], key=lambda kr: D.h64(f"pdsamples|{kr[0]}|{kr[1]}"))
print(f"eval first-chunk rows: {len(ev)}; train rows {len(sp['train'])}", flush=True)
trk = set()
st = DP.Stores()
os.makedirs(OUT, exist_ok=True)
off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
got = 0
for k, r in ev:
    if got >= NCONV:
        break
    items = [(k, r, 0), (k, r, 1)]
    if not all(DP.pose_ok_item(it) for it in items):
        continue
    node, rows, sib = st.open(int(k))
    row = rows[int(r)]
    if int(row["n_frames"]) < NEED + 2:
        continue
    arrs, info = {}, {"row_key": row["row_key"], "item": [int(k), int(r)], "dirs": []}
    ok = True
    for d, it in enumerate(items):
        m = st.conv(it, cfg, 0)
        pv = json.loads(m["prov"])
        P = int(m["P"])
        if not m["voice_clip"] or int(m["pose_ok"]) != 1:
            print(f"skip {row['row_key']}: dir {d} voice_clip {m['voice_clip']} pose_ok {m['pose_ok']}", flush=True); ok = False; break
        assert not pv.get("persona"), pv
        assert pv["chan_B_data_side"] == ("user" if d else "assistant"), pv
        arrs[f"pre_ids_{d}"] = m["ids"][:, :P]; arrs[f"pre_depth_{d}"] = m["depth"][:, :P]; arrs[f"pre_present_{d}"] = m["present"][:, :P]
        arrs[f"pre_pdepth_{d}"] = m["pdepth"][:P]
        assert (m["pdepth"][:P + D.CD] < 0).all()
        arrs[f"laneB_{d}"] = m["ids"][1, P:P + NEED]; arrs[f"depB_{d}"] = m["depth"][1, P:P + NEED]; arrs[f"pdepB_{d}"] = m["pdepth"][P:P + NEED]
        gp = m["pdepth"][P + D.CD:P + D.CD + NEED]
        assert (gp >= 0).all(), "B pose missing inside the window"
        arrs[f"gtpose_{d}"] = (gp - off[None]).T.astype(np.int16)
        assert m["wav"].shape[0] >= (NEED + 1) * 1920, (m["wav"].shape, NEED)
        arrs[f"stem_{1 - d}"] = m["wav"][:(NEED + 1) * 1920].astype(np.float32)   # A of direction d = the OTHER speaker's (1 - d) audio
        pv["aref"] = m["aref"]; pv["pose_src"] = m["pose_src"]
        info["dirs"].append(pv)
    if not ok:
        continue
    for side in ("assistant", "user"):
        info[f"words_{side}"] = [[round(float(w["start"]), 2), w["text"]] for w in (row.get(f"words_{side}") or []) if float(w["start"]) < NEED / 12.5]
    np.savez(f"{OUT}/c{got}.npz", **arrs)
    json.dump(info, open(f"{OUT}/c{got}.json", "w"), default=str)
    print(f"c{got}: {row['row_key']} P {[arrs[f'pre_ids_{d}'].shape[1] for d in (0, 1)]} clips {[p['clips'] for p in info['dirs']]} "
          f"n_frames {row['n_frames']}", flush=True)
    got += 1
print(f"DONE {got} conversations -> {OUT}", flush=True)

"""D33 test data: dump the REAL micros test_d32.py builds (same rows / order: pose rows plan[:N], speech rows plan[:2N], text packs
j * WORLD + RANK) for ranks 0..7 of a one-node world, from the node-local stores of the node it runs on, into <out>/rank<R>.pkl, so the
D33 GPU tests can run on a node without the stores (slinky-15). CPU only.
usage: python tests/dump_d33_micros.py <out dir> <steps>   (env = prod_ppm.env + the test_d32.sh overrides)"""
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    sys.path.insert(0, p)
import numpy as np  # noqa: E402
import data_ppm as D  # noqa: E402

E = os.environ.get
OUT, NS = sys.argv[1], int(sys.argv[2])
LW = WORLD = 8
SEQ = [x for x in E("PPM_GA_SEQ", "pose,speech,speech,text").split(",") if x]
dcfg = D.default_cfg(seed=int(E("PPM_DATA_SEED", "20261011")), sp_cap=int(E("PPM_SP_CAP", "2000")), persona_sides=E("PPM_PERSONA_SIDES", "both"),
                     light_frac=float(E("PPM_LIGHT_FRAC", "1.0")), pose_light_frac=float(E("PPM_POSE_LIGHT_FRAC", "1.0")),
                     txt_cap=int(E("PPM_TXT_CAP", "7500")), crop_conv=E("PPM_CROP_CONV", "1") == "1",
                     chat_side=E("PPM_CHAT_SIDE", "assistant"), chat_eob=E("PPM_CHAT_EOB", "1") == "1", pack_attn=E("PPM_PACK_ATTN", "causal"))
pose_root, speech_root, text_root = E("PPM_POSE_ROOT"), E("PPM_SPEECH_ROOT"), E("PPM_TEXT_ROOT")
seeds = {k: int(E(f"PPM_{k.upper()}_SEED", "20261011")) for k in ("pose", "speech")}
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
pnode = D.PS.load_node(pose_root)
snode = D.R3.load_node(speech_root)
sidx = D.DS.load_index(speech_root)
npm, nsm, ntm = (SEQ.count(k) for k in ("pose", "speech", "text"))
for r in range(WORLD):
    pplan = D.PS.plan(pnode, seeds["pose"], LW, r, 1, LW, r, 1)
    splan = D.speech_plan(snode, sidx, seeds["speech"], LW, r, float(E("PPM_SPEECH_EVAL_FRAC", "0.0005")), E("PPM_SPEECH_EVAL_SALT", "ppm-v1"), 1)
    pds = D.PoseDS(pnode, pplan["rows"][:NS * npm], dcfg, r)
    sds = D.SpeechDS(speech_root, splan["rows"][:NS * nsm], dcfg, r)
    tds = D.TextDS(text_root, "train", [j * WORLD + r for j in range(NS * ntm)], dcfg)
    rec = {"dcfg": dcfg, "seq": SEQ, "node": pnode["node"], "pose": [pds[i] for i in range(NS * npm)], "speech": [sds[i] for i in range(NS * nsm)],
           "text": [tds[i] for i in range(NS * ntm)], "pose_rows": [int(x) for x in pplan["rows"][:NS * npm]]}
    prs = D.PS.PoseRows(pnode)                                    # raw pose rows of the first 3 pose micros (alignment test: labels)
    rec["raw_pose"] = []
    for i in range(min(3, NS * npm)):
        row = prs[int(pplan["rows"][i])]
        rec["raw_pose"].append({"pose": np.asarray(row["pose"]), "segs": [tuple(int(v) for v in x) for x in D.PS.row_segments(row)]})
    pickle.dump(rec, open(f"{OUT}/rank{r}.pkl.tmp", "wb"), protocol=5)
    os.replace(f"{OUT}/rank{r}.pkl.tmp", f"{OUT}/rank{r}.pkl")
    print(f"rank {r}: {NS} steps of micros ({time.time() - t0:.0f}s)", flush=True)
open(f"{OUT}/DONE", "w").write(str(NS))
print("DONE", OUT, flush=True)

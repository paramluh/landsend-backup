"""aloe-vox-ft CPU check of the data path on a test node (stores + some A FLACs present):
  1. plan: per-node items / steps (world 64) = 430 steps; test plan (world 16, ALOE_TEST_CONVS) non-empty.
  2. for N items (eval + test-train) of the local stores: the vox micro vs the pose FT micro of the SAME row / orientation (pose FT stores on Weka):
     channel B (lane, Mimi, pose, targets, cb0, light, pose light) identical on the conversation part (offset by the prefix lengths), B's prefix
     identical whenever the row drew no A voice clip; channel A: no text / codes / present in the conversation; A's prefix codes all absent.
  3. A audio: decode (ffmpeg 24 kHz mono) time, length = n x 1920, FLAC duration vs the conversation's Mimi T (shortfall <= 0.25 s), RMS of the
     gained waveform (-23 LUFS target -> ~0.05-0.1), the speech-energy correlation with A's word timings (frames inside A's words louder).
usage: python -I cpu_check.py --stores 0,2,4,6 [--n 12]"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-vox-ft/code")
import data_aloe_vox as V  # noqa: E402
import data_aloe_pose as AP  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--stores", required=True); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
ks = [int(x) for x in a.stores.split(",")]
pl = V.make_plan(64)
print("[plan] world 64: steps", pl["steps"], "items per node", [len(n["items"]) for n in pl["nodes"]], "eval per node", [len(n["eval"]) for n in pl["nodes"]],
      "train", pl["n_train"], "eval", pl["n_eval"], flush=True)
assert pl["steps"] == 430 and pl["n_train"] == 27517 and pl["n_eval"] == 531
os.environ["ALOE_TEST_CONVS"] = "/data/home/amartyavarma/ppm/aloe-vox-ft/tests/test_convs.json"
pt = V.make_plan(16)
print("[plan] test world 16: steps", pt["steps"], "items per node", [len(n["items"]) for n in pt["nodes"]], flush=True)
del os.environ["ALOE_TEST_CONVS"]
cfg = V.aloe_cfg()
st = V.Stores(ks)
# pose FT store index by row_key
import pyarrow.parquet as pq  # noqa: E402
pidx = {}
for k in range(8):
    t = pq.read_table(f"{AP.SRC_STORES[k]}/rows.parquet", columns=["row", "row_key"]).to_pydict()
    for r, rk in zip(t["row"], t["row_key"]):
        pidx[rk] = (k, r)
pst = AP.Stores(AP.SRC_STORES)
tc = set(json.load(open("/data/home/amartyavarma/ppm/aloe-vox-ft/tests/test_convs.json")))
items = []
sm = V.store_meta()
for k in ks:
    items += [(k, r, sw) for r, sw in sm[f"p{k}"]["items_eval"][:3]]
    items += [(k, r, sw) for r, sw in sm[f"p{k}"]["items_train"] if sm[f"p{k}"]["row_convs"][r] in tc][:max(1, a.n // len(ks) - 3)]
nclip, tdec = 0, []
for it in items:
    t0 = time.time()
    m = st.conv(it, cfg, 0)
    tdec.append(time.time() - t0)
    row = st.open(it[0])[1][it[1]]
    pk, pr = pidx[row["row_key"]]
    mp = pst.conv((pk, pr, it[2]), cfg, 0)
    P, Pp, n = int(m["P"]), int(mp["P"]), int(row["n_frames"])
    pv = json.loads(m["prov"])
    # channel B identical on the conversation part
    for key in ("ids", "depth", "present"):
        x, y = np.asarray(m[key])[1, P:], np.asarray(mp[key])[1, Pp:]
        assert x.shape == y.shape and np.array_equal(x, y), (key, it)
    assert np.array_equal(np.asarray(m["pdepth"])[P:], np.asarray(mp["pdepth"])[Pp:]) and np.array_equal(m["pcb0"][P:], mp["pcb0"][Pp:])
    assert np.array_equal(m["sel"] - P, mp["sel"] - Pp) and np.array_equal(m["y"], mp["y"]) and np.array_equal(m["word"], mp["word"])
    assert np.array_equal(m["cb0"][P:], mp["cb0"][Pp:]) and np.array_equal(m["light"] - P, mp["light"] - Pp)
    assert np.array_equal(m["plight"] - P, mp["plight"] - Pp)
    if not pv["a_clip_dropped"]:
        assert P == Pp and np.array_equal(np.asarray(m["ids"])[1], np.asarray(mp["ids"])[1]) and np.array_equal(np.asarray(m["depth"])[1], np.asarray(mp["depth"])[1]), it
    else:
        nclip += 1
    # channel A empty except the prefix text
    assert (np.asarray(m["depth"])[0] < 0).all() and (np.asarray(m["ids"])[0, P:] == V.D.PAD_ID).all() and (np.asarray(m["present"])[0, P:] == 0).all()
    assert tuple(m["vin"]) == (P, n)
    w = m["wav"]
    assert w.shape == (n * 1920,), w.shape
    # energy inside vs outside A's words (A's word timings from the row; not used by the model, alignment sanity only)
    aside = m["aref"]["side"]
    fr = np.sqrt((w.reshape(n, 1920) ** 2).mean(1) + 1e-12)
    inw = np.zeros(n, bool)
    for wd in row[f"words_{aside}"]:
        s0, s1 = int(float(wd["start"]) * 12.5), int(float(wd["end"]) * 12.5) + 1
        inw[max(s0, 0):min(s1, n)] = True
    e_in, e_out = (float(np.median(fr[inw])) if inw.any() else float("nan")), (float(np.median(fr[~inw])) if (~inw).any() else float("nan"))
    print(f"[item] {it} {row['row_key']} side A={aside} P={P} (pose FT P={Pp}, A clip dropped {pv['a_clip_dropped']}) n={n} | wav rms {float(np.sqrt((w ** 2).mean())):.4f} "
          f"gain {m['aref']['gain_db']:+.2f} dB | frame RMS median inside A words {e_in:.4f} outside {e_out:.4f} (x{e_in / max(e_out, 1e-9):.1f}) | "
          f"load+decode {tdec[-1]:.2f}s | B identical to the pose FT micro", flush=True)
    assert e_in > 2 * e_out, ("A audio not aligned with A's words?", it, e_in, e_out)
print(f"[ok] {len(items)} items: channel B == pose FT micro on every one ({nclip} with A's voice clip dropped -> prefix shorter); A empty; audio aligned; "
      f"decode mean {np.mean(tdec):.2f}s max {max(tdec):.2f}s", flush=True)

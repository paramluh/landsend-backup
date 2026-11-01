"""pose-hands-cmp: per TRACK (conversation i, speaker s1 = B-iii.pkl channel B = SPEAKER_00, s2 = A-iii.pkl channel B = SPEAKER_01) the
real 60 s window exactly as the 32B samplers see it (samples_fast.post_duo: frames f = 0 .. 749 of the row = positions P + CD + f):
  <out>/C-iii-sK.npz : audio [32, 750] int16 (real Mimi), pose [32, 750] int16 (real pose tokens)
  <out>/C-iii-sK.json: words (infer_ppm.lane_words of the real B lane: text / t / prompt), text (all words of the 60 s window, for the
                       8B's upfront text), norm (frame-0 root / width / orig_w / orig_h of that speaker), row_key
Asserts the existing duo runs' gt_s* / gt_pose_s* (aloe-pose-ft-430/runs-pose) equal these where they exist.
   python extract_gt.py <rows-duo dir> <out dir>     (qwen-s2 venv, env_fast.sh)"""
import json
import os
import pickle
import sys

sys.path.insert(0, "/data/home/amartyavarma/ppm/fastinfer")
import fast_ppm as FP  # noqa: E402
import numpy as np  # noqa: E402

I = FP.IP
CD, FPS, OFF = FP.CD, FP.FPS, FP.OFF
RD, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
Npc, Ng = int(round(15.0 * FPS)), int(round(45.0 * FPS))
Ntot = Npc + Ng
REF = "/data/home/amartyavarma/ppm/samples/aloe-pose-ft-430/runs-pose"
rows = json.load(open(f"{RD}/rows.json"))
for r in rows:
    i = int(r["row"].split("-")[1])
    for spk, pre in (("s1", "B"), ("s2", "A")):
        m = pickle.load(open(f"{RD}/{pre}-{i:03d}.pkl", "rb"))
        P = int(m["P"])
        c = (m["depth"][1, P + CD:P + CD + Ntot] - OFF).T
        gp = (m["pdepth"][P + CD:P + CD + Ntot] - OFF).T
        assert c.shape == (32, Ntot) and gp.shape == (32, Ntot) and c.min() >= 0 and gp.min() >= 0 and c.max() < 2048 and gp.max() < 2048
        words = I.lane_words(m["ids"][1][:P + Ntot], P, P, P + Npc)
        ref = f"{REF}/B-{i:03d}/codes.npz"
        if os.path.isfile(ref):
            z = np.load(ref)
            assert np.array_equal(z[f"gt_{spk}"], c) and np.array_equal(z[f"gt_pose_{spk}"], gp), ("differs from runs-pose", i, spk)
            wj = json.load(open(f"{REF}/B-{i:03d}/words.json"))
            assert [w["text"] for w in wj["gt"][spk]] == [w["text"] for w in words], ("words differ from runs-pose", i, spk)
        np.savez_compressed(f"{OUT}/C-{i:03d}-{spk}.npz", audio=c.astype(np.int16), pose=gp.astype(np.int16))
        json.dump({"track": f"C-{i:03d}-{spk}", "conv": i, "spk": spk, "row_key": r["row_key"], "norm": r["norm"][spk], "Npc": Npc, "Ntot": Ntot,
                   "words": words, "text": " ".join(w["text"] for w in words if w["text"]),
                   "n_words": len(words), "n_words_gen": sum(1 for w in words if not w["prompt"])},
                  open(f"{OUT}/C-{i:03d}-{spk}.json", "w"), indent=1)
        print(i, spk, "P", P, "words", len(words), "checked" if os.path.isfile(ref) else "new", flush=True)
print("done")

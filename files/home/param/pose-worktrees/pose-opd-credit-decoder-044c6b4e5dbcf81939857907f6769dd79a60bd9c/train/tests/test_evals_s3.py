"""CPU check of the extra evals: gemma_s3 micros build (both lanes, swaps = eval-list parity / flipped), their conversations are NOT in PPM's
txtdup-full train split; speech_s3 micros load, lanes flipped, row keys = the Demeter 103."""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PPM_CD", "2")
import data_ppm as D
from extra_digest import CFG
g = D.TXD.Store("/data/home/amartyavarma/data/gemma-txtdup-v1")
f = D.TXD.Store("/scratch/amartyavarma/ppm/txtdup-full")
cfg = dict(CFG, txt_cap=int(g.cap))
conv = []
for p in range(152):
    ids = g.pack("eval", p); recs = [g.get(j) for j in ids]
    conv += [r["conv_id"] for r in recs]
    for fl in (0, 1):
        sw = [((int(g.eval_ptr[p]) + k) % 2) ^ fl for k in range(len(recs))]
        m = D.packed(D.text_micro(recs, cfg, sws=sw), cfg)
        assert m["n_text"] > 0 and [x["swap"] for x in json.loads(m["prov"])] == sw
fe = set()
for p in range(f.n_packs("eval")):
    fe |= {f.get(j)["conv_id"] for j in f.pack("eval", p)}
print("gemma_s3: 152 packs,", len(conv), "conversations; in txtdup-full eval split:", sum(c in fe for c in conv), "; (eval rule => none in train)")
assert all(c in fe for c in conv)
man = json.load(open("/data/home/amartyavarma/ppm/evalsets/speech_s3/MANIFEST.json"))
R = pickle.load(open("/data/home/amartyavarma/setup-logs/s3hestia/ourrows/rows103.pkl", "rb"))
assert man["row_keys"] == [r["row_key"] for r in R] and man["n"] == 103
for i in (0, 50, 102):
    a = pickle.load(open(f"/data/home/amartyavarma/ppm/evalsets/speech_s3/A-{i:03d}.pkl", "rb"))
    b = pickle.load(open(f"/data/home/amartyavarma/ppm/evalsets/speech_s3/B-{i:03d}.pkl", "rb"))
    assert a["swap"] == 1 - b["swap"] and b["swap"] == int(R[i]["dataset"] == "B") and a["ids"].shape == b["ids"].shape == (2, 7502)
print("speech_s3: 103 rows x 2 lanes ok; datasets", {d: sum(r["dataset"] == d for r in R) for d in "AB"})

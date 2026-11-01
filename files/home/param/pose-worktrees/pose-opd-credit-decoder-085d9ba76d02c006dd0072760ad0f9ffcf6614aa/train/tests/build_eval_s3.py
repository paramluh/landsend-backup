"""build the speech_s3 eval micros: the 103 stage-3 eval rows (slinky-5 store, data_s3q.plan salt s3eval-v1) that Demeter 'ourrows'
(klweb/s3ref_demeter_ourrows.json) was scored on, through the PPM row path (system prompt, CD, 7,500 crop), in the stage-3 trainer's
eval orientation: lane B = model channel B with swap = (dataset == 'B') (dataset-A rows unswapped, dataset-B rows swapped); lane A =
the same row flipped (the stage-3 channel-A speaker on PPM's predicted channel B). Run on slinky-5. -> OUT/{A,B}-iii.pkl + MANIFEST.json"""
import json, os, pickle, socket, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PPM_CD", "2")
import data_ppm as D
from extra_digest import CFG
out = sys.argv[1]
os.makedirs(out, exist_ok=True)
R = pickle.load(open("/data/home/amartyavarma/setup-logs/s3hestia/ourrows/rows103.pkl", "rb"))
root = "/scratch/amartyavarma/s3-rows"
node = D.R3.load_node(root)
assert node["node"] == "slinky-5" == socket.gethostname().split(".")[0], node["node"]
idx = D.DS.load_index(root)
ds = D.MixedSpeechDS(root, None, [], CFG, 0)
keys = []
for i, r in enumerate(R):
    row = int(r["row"])
    assert idx["row_key"][row] == r["row_key"], (row, idx["row_key"][row], r["row_key"])
    assert D.DS.is_eval_episode(D.DS.episode_key(r["dataset"], r["podcast_id"], r["episode_id"]), 0.0005, "s3eval-v1")
    swB = int(r["dataset"] == "B")
    for o, sw in (("B", swB), ("A", 1 - swB)):
        m = ds.get(D.code_of(0, row), swap=sw)
        assert m["swap"] == sw
        pickle.dump(m, open(f"{out}/{o}-{i:03d}.pkl", "wb"), protocol=4)
    keys.append(r["row_key"])
json.dump({"n": len(R), "node": "slinky-5", "rows": [int(r["row"]) for r in R], "row_keys": keys, "cfg": CFG, "rank_arg": 0,
           "source": "stage-3 eval rows (salt s3eval-v1) of slinky-5 = Demeter ourrows 103; lane B swap = dataset B, lane A flipped"},
          open(f"{out}/MANIFEST.json", "w"))
print("built", len(R), "rows x 2 lanes ->", out)

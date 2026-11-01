"""CPU check of the zoom data path: ZoomAccDS micros (shapes / dtypes / counts / determinism / dummies / epoch cap), eval micros, and
equality of the zoom micro's text part with PPM's own chat micro path (chat_ppm.chat_micro + data_ppm.flatten_segments)."""
import sys, json, time
sys.path.insert(0, sys.argv[1])
import numpy as np
import data_zoom as Z
tb, meta = Z.load_table()
n0 = len(Z.epoch_packs(tb["train"], 0))
ds = Z.ZoomAccDS(tb["train"], [0, None], 64, 3)
t0 = time.time()
m = ds[0]; m2 = ds[0]; md = ds[1]
print("build s", round(time.time() - t0, 2), "keys", sorted(m))
for k in ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light"):
    print(k, np.asarray(m[k]).shape, np.asarray(m[k]).dtype)
assert all(np.array_equal(np.asarray(m[k]), np.asarray(m2[k])) for k in ("ids", "sel", "y")), "not deterministic"
assert md["dummy"] == 1 and md["own_zoom"] == 0.0 and md["padslot"] == 1
N = m["ids"].shape[1]
assert m["segs"] == [N] and (m["depth"] == -1).all() and (m["cb0"] == -100).all() and m["light"].size == 0 and m["own_zoom"] == len(m["sel"])
assert (m["sel"] == np.arange(0, N - 1)).all(), "every position but the last predicts B's next token"
# PPM's own chat path on the same conversations
import chat_ppm as CH, data_ppm as DP
r = Z.Renderer("train"); pk = json.loads(m["zoom_convs"])
ref = DP.flatten_segments(CH.chat_micro([r(j) for j in pk], [str(j) for j in pk], 7500, "assistant", 0, True))
for k in ("ids", "present", "pid", "sel", "y", "word", "pad", "eob", "nw"):
    assert np.array_equal(np.asarray(ref[k]), np.asarray(m[k])), k
print("== PPM chat micro (chat_ppm + data_ppm.flatten_segments) on the same pack: identical", "B words", int(m["word"].sum()), "EOB", int(m["eob"].sum()), "PAD", int(m["pad"].sum()))
# epoch cap (zoom_epoch mode): stream packs >= n0 are dummies
dsc = Z.ZoomAccDS(tb["train"], [63], 64, 77, max_packs=n0)
last = dsc[76]
print("zoom_epoch: step 76 slot 0 (logical 63) stream pack", 76 * 64 + 63, ">= n0", n0, "-> dummy", last.get("dummy", 0))
assert last.get("dummy", 0) == 1
dsc2 = Z.ZoomAccDS(tb["train"], [35], 64, 77, max_packs=n0); assert not dsc2[76].get("dummy", 0)
dsc3 = Z.ZoomAccDS(tb["train"], [36], 64, 77, max_packs=n0); assert dsc3[76].get("dummy", 0)
ev = Z.eval_micros(tb["eval"], 0, 64)
print("eval micros rank 0:", len(ev), [int(e["own_zoom"]) for e in ev])
# stream epochs: 4900 packs then epoch 1, each a permutation
st = Z.Stream(tb["train"])
e0 = [st.get(s)[2][0] for s in range(4900)]; e1 = [st.get(4900 + s)[2][0] for s in range(4900)]
assert sorted(e0) == list(range(4900)) == sorted(e1) and e0 != e1
print("epochs 0 / 1: each a permutation of all 4,900 conversations; differ:", e0[:5], e1[:5])
print("CPU DATA CHECK PASS")

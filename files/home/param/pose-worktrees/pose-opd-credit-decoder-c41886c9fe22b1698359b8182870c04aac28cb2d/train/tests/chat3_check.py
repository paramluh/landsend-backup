import sys, time
sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import chat_ppm as C, data_ppm as D
t0 = time.time(); packs = C.multi_split_packs([2, 3], 7500, 20261011)
n2 = sum(1 for s, _ in packs if s == 2)
print(f"packs total {len(packs):,} (split2 {n2:,}, split3 {len(packs) - n2:,}) in {time.time() - t0:.1f}s; per rank at world 128: {len(packs) // 128:,}")
ds = C.ChatDS([packs[n2], packs[n2 + 1], packs[0]], D.default_cfg())
for i in range(3):
    m = ds[i]
    assert (m["y"] == m["ids"][1, m["sel"] + 1]).all() and m["segs"] == [len(m["pid"])]
    print("split", m["chat_split"], "N", len(m["pid"]), "B words", int(m["word"].sum()), "eob", int(m["eob"].sum()), m["prov"][:100])
print("CHAT3 OK")

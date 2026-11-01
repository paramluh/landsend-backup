# TikTok-pose-data v2+v3 rows of the overlap clip_ids (and same-video rows) -> their mimi codes + clip/segment meta
import sys, glob, io, pickle, numpy as np, multiprocessing as mp, pyarrow.parquet as pq
M = "/data/home/amartyavarma/ppm/code-811/fresh/mimichk"; SRC = "/data/home/amartyavarma/data/TikTok-pose-data"
fresh = [x for f in sorted(glob.glob(f"{M}/fresh-*.pkl")) for x in pickle.load(open(f, "rb"))]
vids = {x["video_id"] for x in fresh}
def one(f):
    t = pq.read_table(f, columns=["clip_id"]).column("clip_id").to_pylist()
    hit = [i for i, c in enumerate(t) if c.split("_", 1)[0] in vids]
    if not hit: return []
    sch = pq.read_schema(f).names
    cols = [c for c in ("clip_id", "segment_id", "kind", "segment", "start_s", "end_s", "mimi") if c in sch]
    tb = pq.read_table(f, columns=cols)
    out = []
    for i in hit:
        r = {c: tb.column(c)[i].as_py() for c in cols if c != "mimi"}
        m = pq.read_table(io.BytesIO(tb.column("mimi")[i].as_py())).to_pylist()[0]
        r["mimi"] = np.asarray(m["mimi_codes"], np.int32); r["words"] = m.get("word_texts"); r["file"] = f
        out.append(r)
    return out
if __name__ == "__main__":
    files = [f"{SRC}/pose-{ds}/{g}-of-2048.parquet" for ds in ("v3", "v2") for g in range(2048)]
    res = []
    with mp.Pool(int(sys.argv[1])) as p:
        for k, o in enumerate(p.imap_unordered(one, files, chunksize=8)):
            res += o
            if k % 512 == 0: print(k, len(res), flush=True)
    pickle.dump(res, open(f"{M}/tiktok.pkl", "wb")); print("DONE tiktok rows", len(res), "videos", len({r['clip_id'].split('_',1)[0] for r in res}), "of", len(vids), flush=True)

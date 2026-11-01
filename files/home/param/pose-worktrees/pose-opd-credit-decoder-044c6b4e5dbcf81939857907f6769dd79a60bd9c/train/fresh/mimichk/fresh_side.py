# per host: fresh segments whose video_id is in PPM's TikTok v2+v3 set -> their raw Mimi codes (audio.mimi_codes) + pose cb0 + meta
import sys, socket, pickle, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
H = socket.gethostname().split(".")[0]; HL = {"slinky-7": "slinky-10"}.get(H, H)
F = "/data/home/amartyavarma/ppm/code-811/fresh"; M = f"{F}/mimichk"; ROOT = "/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs"
ppm = set(np.load(f"{F}/ppm_tiktok_video_ids.npy").tolist())
out = []
for hn in sorted({H, HL}):
    try: t = pq.read_table(f"{F}/scan/{hn}.parquet", columns=["file", "row", "segment_id", "video_id", "clip_id", "segment_index", "n"])
    except FileNotFoundError: continue
    d = t.to_pydict()
    idx = [i for i, v in enumerate(d["video_id"]) if v.isdigit() and int(v) in ppm]
    byf = {}
    for i in idx: byf.setdefault(d["file"][i], []).append(i)
    for f, ii in byf.items():
        try: pf = pq.read_table(f"{ROOT}/{f}", columns=["segment_id", "clip_id", "audio", "segment_metadata_json"])
        except Exception as e: print(H, "skip", f, e); continue
        segs = pf.column("segment_id").to_pylist()
        for i in ii:
            r = d["row"][i]; assert segs[r] == d["segment_id"][i]
            a = pf.column("audio")[r].as_py()
            out.append({"host": H, "segment_id": segs[r], "clip_id": d["clip_id"][i], "video_id": d["video_id"][i], "segment_index": d["segment_index"][i],
                        "mimi": np.asarray(a["mimi_codes"], np.int32), "words": a.get("word_texts"), "segmeta": pf.column("segment_metadata_json")[r].as_py()})
pickle.dump(out, open(f"{M}/fresh-{H}.pkl", "wb")); print(H, "overlap segments", len(out), flush=True)

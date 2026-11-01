import time, glob, pyarrow as pa, pyarrow.parquet as pq, numpy as np
f = sorted(glob.glob("/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs/0012b2d775a54dc58c814c0a68e239ca/*/part-00000.parquet"))[0]
cols = ["segment_id", "pose.tokens_cb0", "pose.orig_h", "audio.mimi_codes"]
t = pq.read_table(f, columns=cols); print(t.schema.names, t.num_rows)
t0 = time.time(); t = pq.read_table(f); rows = t.select(["segment_id","clip_id","video_id","bucket","pose","audio"]).to_pylist(); print("full read+pylist", time.time() - t0, len(rows))
t3 = time.time()
for r in rows:
    p = np.asarray([r["pose"][f"tokens_cb{i}"] for i in range(32)], np.int16); m = np.asarray(r["audio"]["mimi_codes"], np.int32)
print("np", time.time() - t3, p.shape, m.shape)

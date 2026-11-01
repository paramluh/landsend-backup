import time, glob, pyarrow as pa, pyarrow.parquet as pq, numpy as np
print(pa.__version__)
f = sorted(glob.glob("/scratch/amartyavarma/new-pose-llm-data/new-pose-llm-data/jobs/0012b2d775a54dc58c814c0a68e239ca/*/part-00000.parquet"))[0]
cols = ["segment_id", "clip_id", "video_id", "bucket"] + [f"pose.tokens_cb{i}" for i in range(32)] + ["pose.orig_h", "pose.orig_w", "pose.global_state", "pose.frame0_pose", "pose.n_code_steps", "pose.skipped", "pose.checkpoint_revision", "pose.normalization_recipe", "pose.token_time_anchor", "pose.token_stride_s"] + [f"audio.{k}" for k in ("parent_clip_id", "segment_index", "source_start_s", "source_end_s", "language", "num_frames", "n_codebooks", "mimi_codes", "word_texts", "word_start_seconds", "word_end_seconds", "audio_sr")]
t0 = time.time(); t = pq.read_table(f, columns=cols); t1 = time.time()
rows = t.to_pylist(); t2 = time.time()
print("read", t1 - t0, "pylist", t2 - t1, len(rows), list(rows[0].keys()), list(rows[0]["pose"].keys())[:5], list(rows[0]["audio"].keys()))
t3 = time.time()
for r in rows:
    p = np.asarray([r["pose"][f"tokens_cb{i}"] for i in range(32)], np.int16); m = np.asarray(r["audio"]["mimi_codes"], np.int32)
print("np", time.time() - t3, p.shape, m.shape)
t = pq.read_table(f, columns=["segment_id", "video_id", "audio.num_frames"]); print(t.schema)

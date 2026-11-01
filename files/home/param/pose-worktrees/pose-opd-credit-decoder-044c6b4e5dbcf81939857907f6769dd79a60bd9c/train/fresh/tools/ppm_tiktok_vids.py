"""PPM's pose training videos = every video_id (clip_id prefix, as build_pose_rows_sam.py asserts) in the TikTok-pose-data pose-v2 + pose-v3
slices (all 2 x 2048 slices were built into PPM's samdata stores). -> fresh/ppm_tiktok_video_ids.npy (sorted unique uint64) + non-numeric list."""
import sys, os, json, numpy as np, multiprocessing as mp
import pyarrow.parquet as pq, pyarrow.compute as pc
SRC = "/data/home/amartyavarma/data/TikTok-pose-data"
OUT = "/data/home/amartyavarma/ppm/code-811/fresh"
def one(f):
    t = pq.read_table(f, columns=["clip_id"]).column("clip_id")
    v = pc.unique(pc.list_element(pc.split_pattern(t, "_", max_splits=1), 0)).to_pylist()
    return f, len(t), v
if __name__ == "__main__":
    files = [f"{SRC}/pose-{ds}/{g}-of-2048.parquet" for ds in ("v3", "v2") for g in range(2048)]
    assert all(os.path.exists(f) for f in files)
    num, other, rows = set(), set(), 0
    with mp.Pool(56) as p:
        for k, (f, n, v) in enumerate(p.imap_unordered(one, files, chunksize=4)):
            rows += n
            for x in v:
                (num if x.isdigit() else other).add(x)
            if k % 500 == 0: print(k, rows, len(num), len(other), flush=True)
    a = np.array(sorted(int(x) for x in num), np.uint64)
    np.save(f"{OUT}/ppm_tiktok_video_ids.npy.tmp.npy", a); os.replace(f"{OUT}/ppm_tiktok_video_ids.npy.tmp.npy", f"{OUT}/ppm_tiktok_video_ids.npy")
    json.dump({"rows": rows, "files": len(files), "numeric_videos": len(a), "other": sorted(other)}, open(f"{OUT}/ppm_tiktok_video_ids.json", "w"))
    print("DONE rows", rows, "videos", len(a), "non-numeric", len(other), flush=True)

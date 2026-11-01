import numpy as np, pyarrow.parquet as pq, json, glob
r="/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p0"
t=pq.read_table(f"{r}/rows.parquet").to_pydict(); print(list(t.keys())); print({k:t[k][:2] for k in t if k not in ()})
for f in sorted(glob.glob(f"{r}/rows-00000-of-00001.*")): 
    try:
        a=np.load(f, mmap_mode="r"); print(f.split("/")[-1], a.shape, a.dtype)
    except Exception as e: print(f.split("/")[-1], "n/a")
z=np.load(f"{r}/pose_index.npz"); print({k:z[k].shape for k in z.files})
print(json.load(open(f"{r}/POSE.json")) if True else '')

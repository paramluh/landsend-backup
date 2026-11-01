"""pose code distribution: aloe handover pose (store p0, qualifying sides) vs PPM main's pose data (SAM pose-s3 store on a PPM node).
Same tokenizer string asserted at build; here: cb0 / cb8 / cb31 histogram Jensen-Shannon divergence (same codebook -> small)."""
import glob, sys
import numpy as np
def hist(a):
    a = a[a >= 0]; h = np.bincount(a.astype(np.int64), minlength=2048).astype(np.float64); return h / h.sum()
def js(p, q):
    m = 0.5 * (p + q); kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log(x / np.maximum(y, 1e-30)), 0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
z = np.load("/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p0/pose_index.npz"); idx = np.flatnonzero(z["b_ok"][:, 1])[:300]
al = np.load("/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p0/rows-00000-of-00001.pose_assistant.npy", mmap_mode="r")[idx]
sam_f = sorted(glob.glob("/scratch/amartyavarma/pose-s3-samdata/rows/*.pose_assistant.npy"))
sa = np.load(sam_f[0], mmap_mode="r")[:300]
sb = np.load(sam_f[1], mmap_mode="r")[:300]
for cb in (0, 8, 31):
    A, S1, S2 = hist(np.asarray(al[:, cb])), hist(np.asarray(sa[:, cb])), hist(np.asarray(sb[:, cb]))
    perm = np.random.default_rng(0).permutation(2048)
    print(f"cb{cb}: JS(aloe, SAM) {js(A, S1):.3f}  JS(SAM, SAM other shard) {js(S1, S2):.3f}  JS(aloe, SAM scrambled ids) {js(A, S1[perm]):.3f}")

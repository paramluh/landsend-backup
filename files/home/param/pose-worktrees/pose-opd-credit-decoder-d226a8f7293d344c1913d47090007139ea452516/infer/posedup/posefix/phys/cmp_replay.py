import numpy as np
for key, rid, seed in (("7b5a", "20261026-205615-7b5a", 14871328), ("b62a", "20261026-205651-b62a", 57851)):
    a = np.load(f"/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/phys/runs/{key}_U_s{seed}.npz")["codes"].astype(int)
    t = np.load(f"/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/runs/{rid}/tokens.npz")
    b = t["pose"].astype(int)
    n = min(len(a), len(b))
    eq = (a[:n] == b[:n]).all(1)
    first = int(np.argmin(eq)) if not eq.all() else n
    print(key, "replay", a.shape, "call", b.shape, "equal frames", int(eq.sum()), "of", n, "first diff at", first, "keys", t.files)

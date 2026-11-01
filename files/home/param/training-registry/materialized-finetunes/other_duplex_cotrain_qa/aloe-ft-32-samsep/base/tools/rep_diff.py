import json, sys, torch
m = json.load(open("/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/initctl/manifest-slinky-0.json"))
rk = m["rep_keys"]
d = torch.load("/scratch/amartyavarma/aloe-samsep-17850-init/rep.pt", map_location="cpu", mmap=True, weights_only=True)
print("converted rep keys", len(d), "manifest rep_keys", len(rk))
print("only in converted", sorted(set(d) - set(rk))[:20]); print("only in manifest", sorted(set(rk) - set(d))[:20])
print("order equal", list(d) == rk[: len(d)] if set(d) == set(rk) else None, "sorted manifest?", rk == sorted(rk), "converted sorted?", list(d) == sorted(d))
import glob
raw = glob.glob("/scratch/amartyavarma/aloe-samsep-17850-dl/raw/dcp-17850/slinky-0/rep.pt")
if raw:
    s = torch.load(raw[0], map_location="cpu", mmap=True, weights_only=True)
    print("raw top keys", list(s)[:10], {k: (len(v) if hasattr(v, "__len__") else v) for k, v in s.items() if k != "model"} if isinstance(s, dict) else type(s))
    print("raw model keys", len(s["model"]), "dtypes", {str(t.dtype) for t in s["model"].values()})

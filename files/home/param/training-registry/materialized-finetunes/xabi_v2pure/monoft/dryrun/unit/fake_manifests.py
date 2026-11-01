import json, os, sys, torch
L, out = sys.argv[1], sys.argv[2]
m = json.load(open(f"{L}/meta.json")); hosts = m["hosts"]
phys = {h: ("slinky-7" if h == "slinky-10" else h) for h in hosts}
man = {}
for r, h in enumerate(hosts):
    t = torch.load(f"{L}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
    e = man.setdefault(phys[h], {"phys": phys[h], "host_as": h, "step": m["step"], "world": m["world"], "files": {}})
    e["files"][f"rank{r}.pt"] = {"rank": r, "size": os.path.getsize(f"{L}/rank{r}.pt"), "sha256": "x", "rows": {n: int(v.shape[0]) for n, v in t.items()}}
for f in ("rep.pt", "meta.json"):
    man[phys[hosts[0]]]["files"][f] = {"size": os.path.getsize(f"{L}/{f}"), "sha256": "x"}
for h, e in man.items():
    json.dump(e, open(f"{out}/manifest-{h}.json", "w"))
print(len(man), "manifests")

# launch-day placement per PHYSICAL host (src_fetch / src_verify pick placement/<hostname>.tsv): plan key K's share of W/placement/K.tsv;
# slinky-15 plays plan key slinky-9 (PPM_HOST_ALIAS slinky-15:slinky-9), slinky-7 its own plan key (slot slinky-10). npl5/ = parquet only (all the
# trainer, fresh_pose check and the dry run read); npl5json/ = parquet + release.json (WITH_JSON=1, only for an on-host re-scan).
import json, os, shutil
W = "/data/home/amartyavarma/ppm/pose5/npl5"
PHYS = "slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-11 slinky-12 slinky-13 slinky-14 slinky-15 slinky-16 slinky-17 slinky-19".split()
KEY = {p: ("slinky-9" if p == "slinky-15" else p) for p in PHYS}
assert sorted(KEY.values(), key=lambda h: int(h[7:])) == sorted([f[:-4] for f in os.listdir(f"{W}/placement")], key=lambda h: int(h[7:]))
out = {}
for v in ("npl5", "npl5json"):
    d = f"{W}/launch/{v}"; os.makedirs(f"{d}/placement", exist_ok=True); shutil.copyfile(f"{W}/LISTING.json", f"{d}/LISTING.json")
    for p in PHYS:
        n = b = 0
        with open(f"{d}/placement/{p}.tsv", "w") as o:
            for ln in open(f"{W}/placement/{KEY[p]}.tsv"):
                rel, s, xh = ln.rstrip("\n").split("\t")
                if v == "npl5json" or rel.endswith(".parquet"):
                    o.write(ln); n += 1; b += int(s)
        out.setdefault(v, {})[p] = {"key": KEY[p], "files": n, "GB": round(b / 1e9, 2)}
json.dump({"key_of_phys": KEY, "shares": out}, open(f"{W}/launch/PLACEMENT_PHYS.json", "w"), indent=1)
print(json.dumps({v: {p: x["GB"] for p, x in o.items()} for v, o in out.items()}))

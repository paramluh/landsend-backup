"""resolved-config diff: this run's logged cfg (trainer startup line '... cfg {json}') vs job 25963's (aloe-v2vox-raw-w14503). JSON only.
usage: python -I cfg_diff_25963.py <new trainer log> [out json]"""
import json, sys
def cfg_of(log):
    for ln in open(log, errors="replace"):
        if "logical nodes per physical node" in ln and " cfg {" in ln:
            return json.JSONDecoder().raw_decode(ln[ln.index(" cfg {") + 5:])[0]
    raise SystemExit(f"no cfg line in {log}")
def flat(d, pre=""):
    o = {}
    for k, v in d.items():
        o.update(flat(v, f"{pre}{k}.")) if isinstance(v, dict) else o.__setitem__(f"{pre}{k}", v)
    return o
ref = flat(cfg_of("/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/logs/aloev2vraw-25963.log"))
cur = flat(cfg_of(sys.argv[1]))
EXPECT = ("init", "vadapter.init", "vadapter.sha_expect", "run_tag", "log_root", "control_dir", "export", "local_export_root", "export_wait_min",
          "accum.lnodes", "accum.physical_world", "accum.slots")   # init / adapter / paths / bookkeeping (export wait, 8-node accumulation mapping)
rows = [(k, ref.get(k, "<absent>"), cur.get(k, "<absent>")) for k in sorted(set(ref) | set(cur)) if ref.get(k, "<absent>") != cur.get(k, "<absent>")]
print(f"[cfg] resolved config vs 25963: {len(rows)} of {len(set(ref) | set(cur))} keys differ")
for k, a, b in rows:
    print(f"  {'ok ' if k in EXPECT else 'UNEXPECTED'} {k}: 25963 {a} -> new {b}")
bad = [k for k, _, _ in rows if k not in EXPECT]
print(f"[cfg] VERDICT {'PASS (only init / adapter / paths / bookkeeping differ)' if not bad else 'FAIL ' + str(bad)}")
if len(sys.argv) > 2:
    json.dump({"rows": rows, "bad": bad}, open(sys.argv[2], "w"), indent=1, default=str)

"""resolved-config diff: this run's logged cfg (trainer startup line '... cfg {json}') vs job 26232's (aloe-v2vox-asr-w15973). JSON only.
Only the B-speech switch (b_text_only.*), paths (run dirs) and nodes (accum.* = the logical-rank map on 3 nodes) may differ.
usage: python -I cfg_diff_26232.py <new trainer log> [out json]"""
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
ref = flat(cfg_of("/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/logs/aloev2vasr-26232.log"))
cur = flat(cfg_of(sys.argv[1]))
PATHS = ("run_tag", "log_root", "control_dir", "export", "local_export_root")
NODES = ("accum.lnodes", "accum.physical_world", "accum.slots")
ok_ = lambda k: k in PATHS or k in NODES or k.startswith("b_text_only.")   # noqa: E731
rows = [(k, ref.get(k, "<absent>"), cur.get(k, "<absent>")) for k in sorted(set(ref) | set(cur)) if ref.get(k, "<absent>") != cur.get(k, "<absent>")]
print(f"[cfg] resolved config vs 26232: {len(rows)} of {len(set(ref) | set(cur))} keys differ")
for k, a, b in rows:
    kind = "B-speech" if k.startswith("b_text_only.") else "path" if k in PATHS else "nodes" if k in NODES else "UNEXPECTED"
    print(f"  {'ok ' if ok_(k) else 'BAD'} [{kind}] {k}: 26232 {a} -> textB {b}")
bad = [k for k, _, _ in rows if not ok_(k)]
print(f"[cfg] VERDICT {'PASS (only the B-speech switch, paths and nodes differ)' if not bad else 'FAIL ' + str(bad)}")
if len(sys.argv) > 2:
    json.dump({"rows": rows, "bad": bad}, open(sys.argv[2], "w"), indent=1, default=str)

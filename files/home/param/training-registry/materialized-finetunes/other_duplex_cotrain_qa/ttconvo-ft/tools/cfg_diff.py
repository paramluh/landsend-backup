"""resolved-cfg diff: reference cfg (a train_state.json 'cfg' or the 'cfg {...}' JSON of a trainer log's first line) vs the FT's -> differing keys."""
import json, sys
def cfg_of(p):
    if p.endswith(".json"):
        return json.load(open(p))["cfg"]
    for ln in open(p, errors="replace"):
        if "; cfg {" in ln:
            return json.JSONDecoder().raw_decode(ln[ln.index("; cfg {") + 6:])[0]
    raise SystemExit(f"no cfg line in {p}")
def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = v
    return out
a, b = flat(cfg_of(sys.argv[1])), flat(cfg_of(sys.argv[2]))
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        print(f"{k}: {json.dumps(a.get(k, '<absent>'))} -> {json.dumps(b.get(k, '<absent>'))}")

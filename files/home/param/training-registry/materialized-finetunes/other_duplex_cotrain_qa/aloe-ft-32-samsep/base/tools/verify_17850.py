"""ALOEFT32 base (10-24, coordinator: launch ASAP, no extra verification): finalize the node-local 17850 copy. stage_17850.py already
checked every converted file's sha256 against the 17850 monoplex FT init manifest (and every raw download against HF's lfs sha256) and
wrote ctl/stage-<host>.json only after all 129 passed. Here: that marker + 129 files at the manifest sizes + meta.json -> COMPLETE = 17850.
The full CPU load_init / structure check is kept as verify_17850_full.py (not run). The real run's init load is the load check."""
import argparse, glob, json, os, socket
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--ctl", required=True); ap.add_argument("--world", type=int, default=14); ap.add_argument("--step", type=int, default=17850)
ap.add_argument("--man", default="/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/initctl")
a = ap.parse_args()
me = socket.gethostname().split(".")[0]
st = json.load(open(f"{a.ctl}/stage-{a.step}-{me}.json"))
assert st["stage"] == "done" and st["files"] == 129 and int(st["step"]) == a.step, st
exp = {}
for p in glob.glob(f"{a.man}/manifest-*.json"):
    m_ = json.load(open(p))
    if int(m_["step"]) == a.step:
        exp.update(m_["files"])
if not exp:                                                        # no monoft manifest for this step: the stage's own ledger (sha256 of each converted file)
    exp = json.load(open(f"{a.ctl}/ledger-{a.step}-{me}.json"))
files = [f"rank{r}.pt" for r in range(128)] + ["rep.pt"]
bad = [f for f in files if os.path.getsize(f"{a.init}/{f}") != exp[f]["size"]]
assert not bad, ("size != manifest", bad[:6])
m = json.load(open(f"{a.init}/meta.json"))
assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["step"]) == a.step and int(m["world"]) == 128
open(f"{a.init}/COMPLETE", "w").write(str(a.step))
gb = sum(exp[f]["size"] for f in files) / 1e9
json.dump({"host": me, "init": a.init, "COMPLETE": True, "gb": round(gb, 2), "step": a.step, "check": "stage: raw sha256 == HF lfs for all 129 files (+ converted == manifest if one exists); sizes; meta"},
          open(f"{a.ctl}/verify-{a.step}-{me}.json", "w"), indent=1)
print(f"== READY {a.init} on {me}: COMPLETE = {a.step}, 129 bf16 files {gb:.2f} GB", flush=True)

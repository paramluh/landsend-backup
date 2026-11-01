"""Compare the 2-GPU test runs' metrics (runs-811test/<tag>/metrics.jsonl) and logged rows. usage: python test811/compare_runs.py"""
import json
import re

R = "/data/home/amartyavarma/ppm/runs-811test"
L = "/data/home/amartyavarma/ppm/logs/ppm811test"


def recs(tag):
    out = {"train": {}, "eval": {}}
    for ln in open(f"{R}/ppm811-t{tag}/metrics.jsonl"):
        d = json.loads(ln)
        out["eval" if any(k.startswith("eval_") for k in d) else "train"][d["step"]] = d
    return out


def cmp(a, b, pre, keys_from=None):
    ks = sorted(k for k in a if k in b and k.startswith(pre) and isinstance(a[k], float) and not k.endswith(("time_s", "_s", "step_time", "data_wait")))
    df = sorted(((abs(a[k] - b[k]), k) for k in ks), reverse=True)
    return len(ks), sum(1 for x, _ in df if x == 0.0), df[:4]


def show(name, a, b, pre):
    n, ex, top = cmp(a, b, pre)
    print(f"{name}: {n} metrics, exact {ex}, max |diff| {top[0][0]:.3g} ({top[0][1]}); next " + ", ".join(f"{k} {x:.3g}" for x, k in top[1:]))


r1, r4, r5, r2, r3 = (recs(t) for t in ("R1", "R4", "R5", "R2", "R3"))
show("eval at step 3: R4 (PPM mode resumed) vs R1 (PPM mode, before its save)", r4["eval"][3], r1["eval"][3], "eval_")
show("eval at step 3: R5 (8,1,1 mode, PPM GA all full) vs R1", r5["eval"][3], r1["eval"][3], "eval_")
show("eval at step 3: R2 (8,1,1 schedule, switch at 3) vs R1", r2["eval"][3], r1["eval"][3], "eval_")
show("train step 4: R5 (8,1,1 mode, PPM GA all full) vs R4 (PPM mode), same state + rows", r5["train"][4], r4["train"][4], "train_")
show("train step 5: R5 vs R4 (after one optimizer step; FA3 backward is not bitwise deterministic)", r5["train"][5], r4["train"][5], "train_")
show("train step 6: R3 (resumed from R2's dcp-5) vs R2 (uninterrupted)", r3["train"][6], r2["train"][6], "train_")
show("train step 7: R3 vs R2", r3["train"][7], r2["train"][7], "train_")
rows = {}
for tag in ("R2", "R3"):
    for ln in open(f"{L}/t{tag}.log", errors="replace"):
        m = re.match(r"\[811 rows\] rank (\d+) step (\d+): (.*)$", ln.strip())
        if m:
            rows.setdefault(tag, {})[(int(m[1]), int(m[2]))] = m[3]
common = sorted(set(rows.get("R2", {})) & set(rows.get("R3", {})))
same = all(rows["R2"][k] == rows["R3"][k] for k in common)
print(f"rows (pose / speech / text source) of steps {sorted({s for _, s in common})} on ranks {sorted({r for r, _ in common})}: R3 == R2: {same} "
      f"({len(common)} rank-steps)")
for k in common[:2]:
    print(f"   rank {k[0]} step {k[1]}: {rows['R3'][k]}")

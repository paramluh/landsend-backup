"""Resolved-config diff: the 8,1,1 trainer's CFG under prod_811.env vs the live PPM run's logged cfg (ppm/logs/ppm-<job>.log, rank 0's
"PPM <tag>: world .. cfg {...}" line). usage (env already sourced from prod_811.env, RANK/WORLD_SIZE/LOCAL_RANK set):
  python test811/cfg_diff.py <ppm log> <out.txt>
Expected differences = the schedule knobs only: ga_seq, max_steps, run_tag (job id), the s811 block (+ s811_* test/launch-check keys)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG, OUT = sys.argv[1], sys.argv[2]
TORCHW = re.compile(r"\[[WEI]\d{4} \d\d:\d\d:\d\d\.\d+ [^\]]*\] .*$")
txt, carry = None, ""
for line in open(LOG, errors="replace"):
    m = TORCHW.search(line) if "[" in line else None
    if m:                                               # torch C++ warnings spliced into rank 0's long cfg line (ppm_doc.py's glitchfix)
        carry += line[:m.start()]; continue
    if carry:
        line, carry = carry + line, ""
    if " cfg {" in line and "world " in line:
        txt = line[line.index(" cfg {") + 5:].strip(); break
assert txt, ("no cfg line in", LOG)
ref = json.loads(txt)
import train_ppm as T  # noqa: E402   (module-level CFG from the environment; no process group needed)
new = json.loads(json.dumps(T.CFG))


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict) and k not in ("node_swap",):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


a, b = flat(ref), flat(new)
rows = []
for k in sorted(set(a) | set(b)):
    if a.get(k, "<absent>") != b.get(k, "<absent>"):
        rows.append((k, a.get(k, "<absent>"), b.get(k, "<absent>")))
with open(OUT, "w") as f:
    f.write(f"# resolved cfg: 8,1,1 (code-811/prod_811.env) vs live PPM ({os.path.basename(LOG)}); {len(a)} vs {len(b)} keys; {len(rows)} differ\n")
    f.write("# s811.from_step / prev_ga_seq are filled at launch from the resume checkpoint (the switch step and PPM's GA)\n")
    for k, x, y in rows:
        f.write(f"{k}\n    PPM   : {json.dumps(x)}\n    8,1,1 : {json.dumps(y)}\n")
print(open(OUT).read())

"""speaker FT init (10-24): the staged init = 29345's final node-local export-249 (ppm-local-bf16-shards-v1, world 64) + its EXACT fp32 vgain
theta written into meta.json (export_local wrote no 'vgain_theta_fp32' and the shards hold a bf16 theta). Runs on slinky-1 (the rank-0 host).
theta source: the END full save dcp-249 (safety_ppm.save_local: rank<R>.pt['model'] = this rank's fp32 FSDP2 shard; vgain.theta (1,) lives on
rank 0). Checks: exp(theta) == the logged eval/vox_a_scale of 29345 at step 249 (metrics.jsonl, exact float), bf16(theta) == the export
shard's bf16 theta, the save is step 249 of run ppm-aloe-samsep-vox17127-j29345, the export is step 249 / world 64.
usage: python -I init_meta.py <export dir> <dcp dir> <metrics.jsonl> <out meta.json>"""
import json
import math
import sys

import torch

X, S, MJ, OUT = sys.argv[1:5]
ts = json.load(open(f"{S}/train_state.json"))
assert int(ts["step"]) == 249 and ts["run_tag"] == "ppm-aloe-samsep-vox17127-j29345", (ts["step"], ts["run_tag"])
sd = torch.load(f"{S}/rank0.pt", map_location="cpu", mmap=True, weights_only=True)
t = sd["model"]["vgain.theta"]
assert t.dtype == torch.float32 and tuple(t.shape) == (1,), (t.dtype, t.shape)
th = float(t[0])
s = math.exp(th)
logged = None
for ln in open(MJ):
    d = json.loads(ln)
    if int(d["step"]) == 249 and "eval/vox_a_scale" in d:
        logged = float(d["eval/vox_a_scale"])
assert logged is not None and s == logged, ("exp(theta) != 29345's logged eval/vox_a_scale at step 249", s, logged)
ex = torch.load(f"{X}/rank0.pt", map_location="cpu", mmap=True, weights_only=True)["vgain.theta"]
assert ex.dtype == torch.bfloat16 and float(ex[0]) == float(t.to(torch.bfloat16)[0]), (float(ex[0]), th)
m = json.load(open(f"{X}/meta.json"))
assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["step"]) == 249 and int(m["world"]) == 64 and "vgain_theta_fp32" not in m
m.update({"vgain_theta_fp32": th, "vgain_s_fp32": s, "vgain_theta_dtype": "float32", "vox_a_scale": s,
          "vgain_theta_src": f"{S}/rank0.pt model['vgain.theta'] (fp32 FSDP2 shard of rank 0; END full save of job 29345, step 249)",
          "vgain_theta_check": f"exp(theta) == 29345 metrics.jsonl step 249 eval/vox_a_scale {logged!r} exactly; bf16(theta) == export shard theta {float(ex[0])!r}",
          "vgain_theta_fp32_bits": int(t.view(torch.int32)[0]),
          "staged_by": "aloe-ft-32-samsep/speaker (single-speaker FT init, 10-24): = export-249 meta.json + the vgain_* keys above"})
json.dump(m, open(OUT, "w"), indent=1, default=str)
print(f"theta {th!r} (bits {int(t.view(torch.int32)[0])}) s {s!r} == logged {logged!r}; export bf16 theta {float(ex[0])!r}; -> {OUT}")

"""PPM: node-local training save (dcp-N, per-rank fp32 shards + optimizer states on each training node's /scratch) -> an INFERENCE checkpoint on
/data, WITHOUT touching the training process (read-only on the saved files; run as a CPU-only overlap step, one task per training node).

Why: the trainer exports consolidated bf16 weights (w-N, safety_ppm.export_weights) only on its wall-clock cadence (PPM_EXPORT_EVERY_MIN) or at
a step-count cadence; SAVE_NOW writes the node-local save only and STOP_AFTER_SAVE stops the run. There is no 'export now and keep training'
control. The hourly node-local saves hold the exact weights, so this tool converts one of them.

Each task (one per node): reads train_state.json (hosts = rank -> host map), loads each local rank<R>.pt with mmap (only the 'model' section's
pages are touched; optimizer states are never read), casts to bf16, writes OUT/rank<R>.pt. The host of rank 0 also writes OUT/rep.pt (the
replicated decoders, model part) and OUT/meta.json. infer_ppm.load_weights() concatenates the dim-0 FSDP2 shards in rank order (an empty
shard on the last ranks is fine). COMPLETE is written by `--finalize` once all 128 rank files exist.

  srun --jobid <train job> --overlap -N16 --ntasks-per-node=1 -c4 --cpu-bind=none --mem=24G nice -n 19 python ckpt_local_export.py \
       --src /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-335 --out /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-335
  python ckpt_local_export.py --out <same> --finalize      (anywhere; checks every rank file + writes COMPLETE)"""
import argparse
import json
import os
import socket
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--finalize", action="store_true")
ap.add_argument("--min-free-gb", type=float, default=300.0)
a = ap.parse_args()
host = socket.gethostname().split(".")[0]
# aloe-v2pure-17352 copy (10-22): the 8,1,1 run records slinky-7 under slinky-10's LOGICAL slot (nodes16_s7.txt swap) in train_state.json
# hosts, so the physical host may read the ranks of its logical slot: CKX_HOST_AS=<logical host>. Unset = the original tool exactly.
host = os.environ.get("CKX_HOST_AS") or host


def log(*x):
    print(f"[{time.strftime('%H:%M:%S')} {host}]", *x, flush=True)


if a.finalize:
    meta = json.load(open(f"{a.out}/meta.json"))
    miss = [r for r in range(meta["world"]) if not os.path.isfile(f"{a.out}/rank{r}.pt")]
    assert not miss and os.path.isfile(f"{a.out}/rep.pt"), ("missing rank files", miss[:8])
    open(f"{a.out}/COMPLETE", "w").write(str(meta["step"]))
    log(f"COMPLETE {a.out}: {meta['world']} rank files + rep.pt, step {meta['step']}")
    raise SystemExit(0)

assert os.path.isfile(f"{a.src}/COMPLETE"), ("source save not COMPLETE on this node", a.src)
ts = json.load(open(f"{a.src}/train_state.json"))
hosts, step, world = ts["hosts"], int(ts["step"]), int(ts["world"])
mine = [r for r, h in enumerate(hosts) if h == host]
assert mine, (host, "holds no rank of this save")
os.makedirs(a.out, exist_ok=True)
import shutil
free = shutil.disk_usage(a.out).free / 1e9
# aloe-v2pure-17641 copy (10-22, coordinator): the gate = the real size (local-17352: 83 GB) + 3 GB margin above the floor (was 90 on rank 0's
# host only); every other host also checks its own ~5.4 GB, so no partial set is written once /data nears the floor
if 0 in mine:
    assert free - 86 >= a.min_free_gb, (f"/data free {free:.0f} GB: writing ~83 GB would go below the {a.min_free_gb} GB floor")
else:
    assert free - 6 >= a.min_free_gb, (f"/data free {free:.0f} GB: writing this host's ~5.4 GB would go below the {a.min_free_gb} GB floor")
t0 = time.time()
nb = 0
for r in mine:
    sd = torch.load(f"{a.src}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sd["model"].items()}
    nb += sum(t.numel() * 2 for t in out.values())
    torch.save(out, f"{a.out}/rank{r}.pt.tmp")
    os.replace(f"{a.out}/rank{r}.pt.tmp", f"{a.out}/rank{r}.pt")
    del sd, out
if 0 in mine:
    sr = torch.load(f"{a.src}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    out = {n: t.to(torch.bfloat16).contiguous().clone() for n, t in sr["model"].items()}
    torch.save(out, f"{a.out}/rep.pt")
    nb += sum(t.numel() * 2 for t in out.values())
    json.dump({"format": "ppm-local-bf16-shards-v1", "src": a.src, "step": step, "world": world, "hosts": hosts, "run_tag": ts["cfg"].get("run_tag"),
               "run_base": ts["cfg"].get("run_base"), "cfg": ts["cfg"], "save_time": ts.get("time"), "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
              open(f"{a.out}/meta.json", "w"), indent=1)
log(f"ranks {mine} -> {a.out} ({nb / 1e9:.2f} GB bf16, {time.time() - t0:.0f}s)")

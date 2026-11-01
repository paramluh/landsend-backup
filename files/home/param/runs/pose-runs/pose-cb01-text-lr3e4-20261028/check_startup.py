"""Read-only all-node startup health snapshot; stores evidence on the controller."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import datetime
import json
import math
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import json, socket, subprocess
root = Path("/home/param/pose-runs/pose-cb01-text-lr3e4-20261028")
job = (root / "cluster/latest").resolve()
def live(name):
    identity = json.loads((job / (name + ".json")).read_text())
    try:
        stat = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[19] == identity["start_ticks"] and stat[0] != "Z"
    except FileNotFoundError:
        return False
log = (job / "node.log").read_text(errors="replace")
errors = [line for line in log.splitlines() if any(key in line for key in ("Traceback (", "OutOfMemoryError", "ChildFailedError", "NCCL watchdog"))]
gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
rows = [[int(v.strip()) for v in row.split(",")] for row in gpu.splitlines()]
out = dict(host=socket.gethostname().split(".")[0], launch=job.name, supervisor_live=live("supervisor"),
           torchrun_live=live("torchrun"), exit_code=(job / "exit_code").read_text().strip() if (job / "exit_code").exists() else None,
           gpus=len(rows), gpu_utilization=[r[0] for r in rows], max_gpu_memory_gib=max(r[1] for r in rows)/1024,
           errors=errors, evaluations="[eval]" in log, checkpoint_dirs=[p.name for p in (root / "saves").glob("step-*")],
           export_dirs=[p.name for p in (root / "exports").glob("export-*")])
if (root / "runtime.json").exists():
    out["runtime"] = json.loads((root / "runtime.json").read_text())
    records = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    out["metrics"] = records[-1]
print(json.dumps(out))
'''

def check(host):
    command = "python3 -c " + shlex.quote(REMOTE)
    p = subprocess.run(["ssh", "-i", "/home/param/.ssh/id_ed25519_cluster", "-o", "BatchMode=yes",
                        "-o", "ConnectTimeout=10", host, command], capture_output=True, text=True, timeout=30, check=True)
    return json.loads(p.stdout)

minute = int(sys.argv[1])
with ThreadPoolExecutor(max_workers=4) as pool:
    nodes = list(pool.map(check, [f"landsend-{n}" for n in range(1, 5)]))
now = datetime.datetime.now(datetime.timezone.utc)
runtime = nodes[0]["runtime"]
age = (now - datetime.datetime.fromisoformat(runtime["updated_utc"])).total_seconds()
healthy = all(n["torchrun_live"] and n["supervisor_live"] and n["exit_code"] is None and n["gpus"] == 8
              and not n["errors"] and not n["evaluations"] and not n["checkpoint_dirs"] and not n["export_dirs"] for n in nodes)
healthy = healthy and runtime["allocator_ooms"] == 0 and age < 90
metrics = nodes[0]["metrics"]
healthy = healthy and "train/audio_loss" not in metrics and "train/pose_light_loss" not in metrics
healthy = healthy and math.isclose(metrics["train/loss"], metrics["train/text_loss"] + metrics["train/pose_cb0_loss"] + metrics["train/pose_cb1_loss"], rel_tol=1e-6)
healthy = healthy and all(math.isfinite(v) for v in nodes[0]["metrics"].values() if isinstance(v, (int, float)))
report = dict(minute=minute, checked_utc=now.isoformat(), healthy=healthy, runtime_age_seconds=age, nodes=nodes)
(ROOT / f"monitor-{minute:02d}.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(dict(minute=minute, healthy=healthy, workers_live=sum(n["torchrun_live"] for n in nodes),
                     **runtime, pose_cb1=nodes[0]["metrics"]["train/pose_cb1_loss"])))
if not healthy:
    print(json.dumps(report, indent=2))
    raise SystemExit(1)

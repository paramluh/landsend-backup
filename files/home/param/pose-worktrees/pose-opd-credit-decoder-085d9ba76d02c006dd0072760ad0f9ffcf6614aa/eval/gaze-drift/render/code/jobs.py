"""Append (add) or remove (rm) our GAZE-* entries in s20 control/vrefs_jobs.json atomically; never touches other entries.
   python3 jobs.py add KEY:IDN [...] | rm"""
import json
import os
import sys

JOBS = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control/vrefs_jobs.json"
VD = "/data/home/amartyavarma/ppm/gaze-drift/render/vdata"


def write(j):
    tmp = f"{JOBS}.gaze.tmp"
    json.dump(j, open(tmp, "w"), indent=1)
    os.replace(tmp, JOBS)


j = json.load(open(JOBS))
if sys.argv[1] == "add":
    have = {e["key"] for e in j}
    for s in sys.argv[2:]:
        key, idn = s.split(":")
        assert key.startswith("GAZE-") and os.path.exists(f"{VD}/{key}/pose_latents_electron_v3.safetensors"), key
        if key not in have:
            j.append({"key": key, "vdata": f"{VD}/{key}", "idn": idn, "prefill": 25})
    write(j)
elif sys.argv[1] == "rm":
    write([e for e in j if not e["key"].startswith("GAZE-")])
print(len(j), [e["key"] for e in json.load(open(JOBS)) if e["key"].startswith("GAZE-")])

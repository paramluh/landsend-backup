"""Measure render-agent videos not yet in eyes_series.json and append them."""
import glob, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, "/data/home/amartyavarma/ppm/gaze-drift/path")
import eyes as E
P = "/data/home/amartyavarma/ppm/gaze-drift/path"
S = json.load(open(f"{P}/eyes_series.json"))
jobs = []
for p in sorted(glob.glob(f"{E.RO}/GAZE-E*-*.mp4")):
    b = os.path.basename(p)[:-4]
    e, row = b.split("-", 2)[1], b.split("-", 2)[2]
    k = f"R{e}/{row}"
    if row in E.ROWS and k not in S:
        jobs.append((k, p, E.ROWS[row]))
RQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
for row, (rid, k) in json.load(open("/data/home/amartyavarma/ppm/gaze-drift/samp/gz_l500.json")).items():
    if row in E.ROWS and f"l500/{row}" not in S:
        jobs.append((f"l500/{row}", f"{RQ}/{rid}/s{k}/video.mp4", E.ROWS[row]))
with Pool(6) as pool:
    for r in pool.imap_unordered(E.measure, jobs):
        S[r["key"]] = r
json.dump(S, open(f"{P}/eyes_series.json.tmp", "w"))
os.replace(f"{P}/eyes_series.json.tmp", f"{P}/eyes_series.json")
print("added", [j[0] for j in jobs])

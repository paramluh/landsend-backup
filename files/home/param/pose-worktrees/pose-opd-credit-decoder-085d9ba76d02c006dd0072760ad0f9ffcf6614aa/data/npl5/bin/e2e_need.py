"""E2E helper: the parquet files one slot needs for the trainer micros the e2e dry run builds (every local rank at the resume step 17850, its
first npl5 row and the last step END-1), from the 4 plan dirs under <root>. -> <out>/n3/{LISTING.json, placement/<host>.tsv} (delivery-3 bucket)
and <out>/n5/{...} (delivery-5 bucket) for src_fetch.py; prints counts.
usage (qwen-s2 venv, python -I): e2e_need.py <root> <key> <end> <out>"""
import json, os, socket, sys
import numpy as np
root, key, end, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
SF, NPM, LW, RESUME = 16861, 8, 8, 17850
A = "/data/home/amartyavarma/ppm/code-811/fresh/append"; W = "/data/home/amartyavarma/ppm/pose5/npl5"
sys.path.insert(0, "/data/home/amartyavarma/ppm/code-811/fresh")
import fresh_pose as FP
pl = [FP.load_plan(f"{root}/{d}") for d in ("ppm-fresh-pose", "ppm-fresh-pose-npl2", "ppm-fresh-pose-npl3", "ppm-fresh-pose-npl5")]
need = {2: set(), 3: set()}
for l in range(LW):
    s = FP.append_stream([int(p["n_rows"]) for p in pl], l, LW, NPM, SF, 1, ())[:(end - SF) * NPM]
    w5 = np.flatnonzero((s >> FP.PLAN_SHIFT) == 3)
    for i in [(RESUME - SF) * NPM] + ([int(w5[0])] if len(w5) else []) + [len(s) - 1]:
        k, r = FP.split_row(int(s[i])); z = pl[k]["z"]
        assert k in need, ("micro row in plan", k + 1)
        need[k] |= {pl[k]["files"][int(j)] for j in np.unique(z["seg_file"][z["row_seg"][z["row_off"][r]:z["row_off"][r + 1]]])}
host = socket.gethostname().split(".")[0]
for k, sub, lst, L in ((2, "n3", f"{A}/npl3/listing.tsv", f"{A}/npl3/LISTING.json"), (3, "n5", f"{W}/listing.tsv", f"{W}/LISTING.json")):
    want = {f"jobs/{f}" for f in need[k]}
    os.makedirs(f"{out}/{sub}/placement", exist_ok=True)
    json.dump(json.load(open(L)), open(f"{out}/{sub}/LISTING.json", "w"))
    n = 0
    with open(f"{out}/{sub}/placement/{host}.tsv", "w") as o:
        for ln in open(lst):
            if ln.split("\t", 1)[0] in want:
                o.write("\t".join(ln.rstrip("\n").split("\t")[:3]) + "\n"); n += 1
    assert n == len(want), (sub, n, len(want))
    print(f"{key}: {sub} {n} files", flush=True)

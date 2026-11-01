"""merge the per-host podcast A-side FLAC manifests -> plan/podcast_flac_manifest.jsonl (+ coverage check vs the plan).
Required (fine-tune agent): for every (host, local rank) the FIRST NEED unseen codes + both eval codes. Adds 'side' (= chan_A_data_side) and
'frame0' (row frame 0 = sample 0 of the file; the file IS the row window) to lines written before those fields existed; re-derives the side
from data_ppm.side_of(row_key, 20261011) and asserts it.  usage: merge_manifest.py [NEED]"""
import glob
import hashlib
import json
import os
import sys

NEED = int(sys.argv[1]) if len(sys.argv) > 1 else 198
BASE = "/data/home/amartyavarma/ppm/si20k-vox-ppm/plan"
PLAN = f"{BASE}/PODCAST_PLAN_ppm.json"
OUT = f"{BASE}/podcast_flac_manifest.jsonl"


def side_of(key, seed=20261011):
    return int(int.from_bytes(hashlib.blake2b(f"ppm-side|{seed}|{key}".encode(), digest_size=8).digest(), "little") & 1)


P = json.load(open(PLAN))
lines, have = [], {}
for f in sorted(glob.glob(f"{BASE}/podflac/manifest/slinky-*.jsonl")):
    for l in open(f):
        m = json.loads(l)
        sd = "assistant" if side_of(m["row_key"]) == 1 else "user"
        assert m["chan_A_data_side"] == sd, (m["row_key"], m["chan_A_data_side"], sd)
        m["side"] = sd
        m["frame0"] = 0
        m["path"] = m["path"]
        have[(m["host"], m["local_rank"], m["code"])] = m
        lines.append(m)
miss, cover = [], {}
for host in P["prod_hosts"]:
    for j in range(8):
        rk = P["hosts"][host]["ranks"][str(j)]
        req = [int(c) for c in rk["unseen_codes"][:NEED]] + [int(c) for c in rk["eval_codes"]]
        mm = [c for c in req if (host, j, c) not in have]
        miss += [(host, j, c) for c in mm]
        n = 0
        for c in rk["unseen_codes"]:
            if (host, j, int(c)) not in have:
                break
            n += 1
        cover[f"{host}/{j}"] = n
order = {h: i for i, h in enumerate(P["prod_hosts"])}
lines.sort(key=lambda m: (order[m["host"]], m["local_rank"], m["kind"] != "train", m["stream_pos"]))
key_first = ["row_key", "host", "local_rank", "side", "path", "frame0"]
with open(OUT + ".tmp", "w") as f:
    for m in lines:
        f.write(json.dumps({**{k: m[k] for k in key_first}, **{k: v for k, v in m.items() if k not in key_first}}) + "\n")
os.replace(OUT + ".tmp", OUT)
summ = {"manifest": OUT, "lines": len(lines), "need_per_rank": NEED, "missing_required": len(miss), "missing_sample": miss[:10],
        "min_contiguous_prefix_per_rank": min(cover.values()), "max_contiguous_prefix_per_rank": max(cover.values()),
        "eval_rows": sum(1 for m in lines if m["kind"] == "eval"), "padded_rows": sum(1 for m in lines if m.get("padded_samples", 0)),
        "hosts": sorted({m["host"] for m in lines}), "bytes": sum(m["bytes"] for m in lines)}
json.dump({**summ, "prefix_per_rank": cover}, open(f"{BASE}/podflac/MANIFEST_SUMMARY.json", "w"), indent=1)
print(json.dumps(summ))

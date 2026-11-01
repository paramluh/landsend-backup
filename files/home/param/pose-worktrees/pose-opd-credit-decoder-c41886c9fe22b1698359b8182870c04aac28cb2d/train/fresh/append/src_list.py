"""APPEND ingest step 1 (fresh/append/ingest_more.sh): list the new pose data's source -> <W>/listing.tsv + <W>/LISTING.json.
usage (compute node; voicebank venv, python -I):  src_list.py <SRC> <W>
  SRC = hf://buckets/<owner>/<bucket>/<prefix>   an HF bucket path like new-pose-llm-data's (bucket listing API, like data/new-pose-llm-data-dl/
                                                 list.py; metadata only, no file is downloaded here)
      | /abs/dir                                  a local / Weka copy: the dir holding jobs/ (or holding exactly one <prefix>/jobs)
Kept: jobs/<job>/<release>/{part-*.parquet, release.json, complete.json} (all fresh_pose.py scan reads; inputs.json / runtime.json / _canary /
manifests are not needed). Excluded: every release (jobs/<job>/<release_sha256>, content-addressed) already placed for plan 1
(data/new-pose-llm-data-dl/PLACEMENT_files.tsv) or listed for an earlier appended tag (fresh/append/<tag>/listing.tsv): those segments are in
an earlier plan; a NEW release of an old job is kept (its segments are deduped by segment_id / clip / span in the plan step).
listing.tsv rows: rel_path (from the source root, jobs/...) TAB size TAB xet_hash ('-' for a local source) TAB job.
Test-only knobs (the smoke test on plan-1's Weka copy): LIST_LIMIT_JOBS=<n> (local source: the first n jobs), LIST_NO_EXCLUDE=1."""
import collections
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SRC, W = sys.argv[1], sys.argv[2]
A = "/data/home/amartyavarma/ppm/code-811/fresh/append"
PLAN1_FILES = "/data/home/amartyavarma/data/new-pose-llm-data-dl/PLACEMENT_files.tsv"
KEEP = lambda name: (name.startswith("part-") and name.endswith(".parquet")) or name in ("release.json", "complete.json")  # noqa: E731
os.makedirs(W, exist_ok=True)
t0 = time.time()


def rel_key(rel):
    p = rel.split("/")
    return "/".join(p[:3]) if len(p) >= 4 and p[0] == "jobs" else None


# ---- releases already used (plan 1 + earlier appended tags)
used = set()
for ln in open(PLAN1_FILES):
    path = ln.split("\t", 1)[0]
    i = path.find("/jobs/")
    if i >= 0:
        k = rel_key(path[i + 1:])
        if k:
            used.add(k)
n_plan1 = len(used)
for f in sorted(glob.glob(f"{A}/*/listing.tsv")):
    if os.path.dirname(os.path.abspath(f)) == os.path.abspath(W):
        continue
    for ln in open(f):
        k = rel_key(ln.split("\t", 1)[0])
        if k:
            used.add(k)
print(f"releases already used: plan 1 {n_plan1:,}, + earlier appended listings -> {len(used):,}", flush=True)

rows, st = [], collections.Counter()
if SRC.startswith("hf://buckets/"):
    from huggingface_hub import HfApi
    p = SRC[len("hf://buckets/"):].strip("/").split("/")
    bucket, prefix = "/".join(p[:2]), "/".join(p[2:])
    api = HfApi()
    for x in api.list_bucket_tree(bucket, prefix=prefix or None, recursive=True):
        if type(x).__name__ != "BucketFile":
            continue
        path = x.path
        rel = path[len(prefix) + 1:] if prefix and path.startswith(prefix + "/") else path
        st["files_listed"] += 1; st["bytes_listed"] += int(x.size)
        if st["files_listed"] % 200000 == 0:
            print(f"listed {st['files_listed']:,} {time.time() - t0:.0f}s", flush=True)
        q = rel.split("/")
        if len(q) != 4 or q[0] != "jobs" or not KEEP(q[3]):
            st["skip_not_needed"] += 1; continue
        rows.append((rel, int(x.size), str(x.xet_hash), q[1]))
    src = {"kind": "hf", "bucket": bucket, "prefix": prefix}
else:
    root = os.path.abspath(SRC)
    if not os.path.isdir(f"{root}/jobs"):
        cand = [d for d in glob.glob(f"{root}/*/jobs") if os.path.isdir(d)]
        assert len(cand) == 1, ("local source: no jobs/ under", root, cand)
        root = os.path.dirname(cand[0])
    jobs = sorted(os.listdir(f"{root}/jobs"))[:int(os.environ.get("LIST_LIMIT_JOBS", "0")) or None]

    def walk(job):
        out = []
        for rel_ in sorted(os.listdir(f"{root}/jobs/{job}")):
            d = f"{root}/jobs/{job}/{rel_}"
            if not os.path.isdir(d):
                continue
            for n in sorted(os.listdir(d)):
                if KEEP(n) and os.path.isfile(f"{d}/{n}"):
                    out.append((f"jobs/{job}/{rel_}/{n}", os.path.getsize(f"{d}/{n}"), "-", job))
        return out
    with ThreadPoolExecutor(32) as ex:
        for k, r in enumerate(ex.map(walk, jobs)):
            rows += r
            if k % 50000 == 0:
                print(f"walked {k:,}/{len(jobs):,} jobs {time.time() - t0:.0f}s", flush=True)
    st["files_listed"] = len(rows); st["bytes_listed"] = sum(r[1] for r in rows)
    src = {"kind": "local", "root": root}
keep, rels = [], collections.defaultdict(list)
for r in rows:
    rels[rel_key(r[0])].append(r)
for k, rs in rels.items():
    if k in used and os.environ.get("LIST_NO_EXCLUDE") != "1":
        st["skip_release_already_used"] += 1; st["skip_files_already_used"] += len(rs); continue
    names = {x[0].rsplit("/", 1)[1] for x in rs}
    if not any(n.endswith(".parquet") for n in names):
        st["skip_release_no_parquet"] += 1; continue
    if not names & {"release.json", "complete.json"}:
        st["skip_release_no_pin_file"] += 1; continue              # (scan would reject it: no pins)
    keep += rs
keep.sort()
with open(f"{W}/listing.tsv.tmp", "w") as f:
    for r in keep:
        f.write("\t".join(str(x) for x in r) + "\n")
os.replace(f"{W}/listing.tsv.tmp", f"{W}/listing.tsv")
info = {"src": src, "src_arg": SRC, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1), "stats": dict(st),
        "kept_files": len(keep), "kept_parquet": sum(1 for r in keep if r[0].endswith(".parquet")),
        "kept_bytes": sum(r[1] for r in keep), "kept_parquet_bytes": sum(r[1] for r in keep if r[0].endswith(".parquet")),
        "kept_releases": len({rel_key(r[0]) for r in keep}), "kept_jobs": len({r[3] for r in keep}), "releases_used_before": len(used)}
json.dump(info, open(f"{W}/LISTING.json", "w"), indent=1)
print(json.dumps(info), flush=True)
assert info["kept_parquet"] > 0, "nothing new to ingest (every release already used, or nothing listed)"

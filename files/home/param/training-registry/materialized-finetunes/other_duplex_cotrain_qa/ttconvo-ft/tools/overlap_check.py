"""tt-v3 vs PPM pretraining speech mix: episode / podcast overlap (s3-rows FULL rows manifest + the 4 extra slices = podcast_full_rows.parquet)."""
import json, sys, collections
import pyarrow.parquet as pq
T = pq.read_table("/data/home/amartyavarma/ppm/podcast_full_rows.parquet")
print("ppm cols", T.column_names, T.num_rows)
d = T.slice(0, 3).to_pylist(); print(d)
ep = T.column("ep").to_pylist() if "ep" in T.column_names else None
ppm_eps = set(ep)
def pod(e):
    p = e.split("/"); return p[-2]
def epi(e):
    return e.split("/")[-1]
ppm_epi = set(epi(e) for e in ppm_eps); ppm_pod = set(pod(e) for e in ppm_eps)
ppm_ds = collections.Counter(e.split("/")[0] for e in ppm_eps)
print("ppm episodes", len(ppm_eps), "podcasts", len(ppm_pod), "datasets", ppm_ds.most_common(10))
for name, path in [("tt-v3", "/data/home/amartyavarma/qwen-1ch-data/tt-v3/index.jsonl"), ("tt-s3(v1)", "/data/home/amartyavarma/qwen-1ch-data/tt-s3/index.jsonl")]:
    try:
        L = [json.loads(l) for l in open(path)]
    except FileNotFoundError:
        print(name, "no index", path); continue
    eps = [l["ep"] for l in L if l.get("usable", True)]
    ds = collections.Counter(e.split("/")[0] for e in eps)
    full = sum(1 for e in eps if e in ppm_eps); ide = sum(1 for e in eps if epi(e) in ppm_epi); pd = sum(1 for e in eps if pod(e) in ppm_pod)
    print(name, "episodes", len(eps), "datasets", ds.most_common(5), "| exact ep in PPM:", full, "| episode_id in PPM:", ide, "| podcast in PPM:", pd)

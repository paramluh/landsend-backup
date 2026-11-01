"""Repo-level check of a per-host full-state backup repo: private, anonymous refused, exactly <n> files under <K>/<host>/ for each host,
one _backup/MANIFEST.<K>.<host>.tsv per host, nothing else (bar .gitattributes).   hf_state_repo_check.py REPO K host1,host2,... N"""
import collections, sys
from huggingface_hub import HfApi
r, K, hosts, N = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
api = HfApi()
priv = api.repo_info(r).private
fs = {e.path: e.size for e in api.list_repo_tree(r, recursive=True) if hasattr(e, "blob_id")}
per = collections.Counter(p.split("/")[1] for p in fs if p.startswith(K + "/"))
mans = sorted(p for p in fs if p.startswith(f"_backup/MANIFEST.{K}."))
other = sorted(p for p in fs if not (p.startswith(K + "/") or p in mans or p == ".gitattributes"))
try:
    HfApi(token=False).repo_info(r); anon = "ANON CAN READ"
except Exception as e:
    anon = f"anonymous refused ({type(e).__name__})"
ok = priv and not other and set(per) == set(hosts) and all(per[h] == N for h in hosts) and len(mans) == len(hosts) and anon.startswith("anonymous refused")
print(f"{'PASS' if ok else 'FAIL'} {r}: private {priv}; {anon}; {len(fs)} files {sum(fs.values()) / 1e9:.1f} GB; per host {dict(sorted(per.items()))}; manifests {len(mans)}; other {other[:5]}")
sys.exit(0 if ok else 3)

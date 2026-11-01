"""Final repo-level check of the dcp-498 state backup repo: private, anonymous refused, exactly 20 files per host x 8 + 8 manifests + .gitattributes."""
import collections, sys
from huggingface_hub import HfApi
r = sys.argv[1]; hosts = sys.argv[2].split(",")
api = HfApi()
priv = api.repo_info(r).private
fs = {e.path: e.size for e in api.list_repo_tree(r, recursive=True) if hasattr(e, "blob_id")}
per = collections.Counter(p.split("/")[1] for p in fs if p.startswith("dcp-498/"))
other = sorted(p for p in fs if not (p.startswith("dcp-498/") or p.startswith("_backup/MANIFEST.dcp-498.") or p == ".gitattributes"))
try:
    HfApi(token=False).repo_info(r); anon = "ANON CAN READ"
except Exception as e:
    anon = f"anonymous refused ({type(e).__name__})"
ok = priv and not other and all(per.get(h) == 20 for h in hosts) and set(per) == set(hosts) and anon.startswith("anonymous refused")
print(f"{'PASS' if ok else 'FAIL'} {r}: private {priv}; {anon}; {len(fs)} files {sum(fs.values()) / 1e9:.1f} GB; per host {dict(per)}; other {other[:5]}")
sys.exit(0 if ok else 3)

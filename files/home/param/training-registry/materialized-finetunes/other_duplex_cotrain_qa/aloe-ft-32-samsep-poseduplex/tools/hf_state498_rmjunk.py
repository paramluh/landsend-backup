"""Remove the junk xet-cache/ folder that hf_state498_host.sbatch v1 uploaded (its HF_XET_CACHE sat inside the farm) from the PRIVATE state repo;
list what remains. Touches ONLY repo paths under xet-cache/ (never dcp-498/ or _backup/)."""
import collections, sys
from huggingface_hub import HfApi
r = sys.argv[1]
api = HfApi()
assert api.repo_info(r).private is True, "not private"
fs = [e.path for e in api.list_repo_tree(r, recursive=True) if hasattr(e, "blob_id")]
junk = [p for p in fs if p.startswith("xet-cache/")]
print(f"{r}: {len(fs)} files; junk under xet-cache/: {len(junk)}")
if junk:
    api.delete_folder("xet-cache", repo_id=r, commit_message="remove xet-cache/ uploaded by mistake (HF_XET_CACHE was inside the upload farm)")
fs = [e.path for e in api.list_repo_tree(r, recursive=True) if hasattr(e, "blob_id")]
c = collections.Counter(p.split("/")[0] + ("/" + p.split("/")[1] if p.startswith("dcp-498/") else "") for p in fs)
print("after:", len(fs), "files;", dict(sorted(c.items())))
assert not any(p.startswith("xet-cache/") for p in fs), "junk still there"

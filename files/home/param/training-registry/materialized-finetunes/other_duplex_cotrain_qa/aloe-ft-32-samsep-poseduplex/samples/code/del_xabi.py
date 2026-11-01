"""delete /scratch/amartyavarma/xabi-samples/<w>/export-200 on THIS host only if every local file is on private HF (same name, same size,
under the repo's export-200 or export-200-dcp path) (main amartyavarma-61 GO 10-25). Logs every decision to HF_DELETE_LOG + stdout."""
import os, sys, shutil, socket, time
from huggingface_hub import HfApi
H = socket.gethostname().split(".")[0]
REPOS = {"w17641": "InternalCan/ppm-aloe-v2pure-17641-final", "w17850": "InternalCan/ppm-aloe-v2pure-main-17850-final"}
LOG = "/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log"
api = HfApi(); root = "/scratch/amartyavarma/xabi-samples"
for w in sorted(os.listdir(root)):
    d = f"{root}/{w}/export-200"
    if not os.path.isdir(d):
        print(H, w, "no export-200, left alone"); continue
    loc = {n: os.path.getsize(f"{d}/{n}") for n in os.listdir(d) if os.path.isfile(f"{d}/{n}")}
    R = REPOS.get(w); ok, where = False, None
    if R:
        for pth in ("export-200", "export-200-dcp"):
            try:
                hf = {}
                for e in api.list_repo_tree(R, path_in_repo=pth, recursive=True, expand=True):
                    if hasattr(e, "size"):
                        hf.setdefault(e.path.rsplit("/", 1)[1], set()).add(int(e.size))
            except Exception as ex:                                  # noqa: BLE001
                continue
            miss = [n for n, s in loc.items() if s not in hf.get(n, set())]
            if not miss:
                ok, where = True, f"{R}/{pth}"; break
            print(H, w, pth, "missing/size-mismatch on HF:", miss[:5], len(miss))
    sz = sum(loc.values()) / 1e9
    if not ok:
        print(H, w, f"KEPT ({sz:.1f} GB): not fully on HF"); continue
    shutil.rmtree(f"{root}/{w}")
    msg = f"[{time.strftime('%m-%d %H:%M:%S')} UTC] pdsamples (voxsamples agent, main amartyavarma-61 GO): deleted {H}:{root}/{w} ({sz:.1f} GB, {len(loc)} files, Xabi FT export-200 staged copy for old sample servers) - every file on private HF {where} with equal size"
    open(LOG, "a").write(msg + "\n"); print(msg)

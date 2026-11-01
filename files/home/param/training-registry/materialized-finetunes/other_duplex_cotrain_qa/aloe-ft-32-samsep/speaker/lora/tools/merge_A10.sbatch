#!/bin/bash
#SBATCH -J aloespk-merge -p all -N1 -n1 -c 8 --mem=96G -t 02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/merge-%j.log
# main 86 / USER 10-24 ~09:40 PT: "host the Brianna version for me now, live": merge run A's export-10 delta (private HF, verified) into the
# vox17127 export-249 base (staged on slinky-18, sha-verified) -> FULL ppm-local-bf16-shards-v1 export at speaker/lora/merged-A10 (/data).
set -euo pipefail
L=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora; DL=/scratch/amartyavarma/aloespk-lora/dl-A10; BASE=/scratch/amartyavarma/aloespk-init-w8/export-249
mkdir -p $DL; cd /tmp
/data/home/amartyavarma/venvs/duplex/bin/python -I - <<'PY'
import os, hashlib, shutil
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
R = "InternalCan/ppm-aloespk-lora-fFP3835k-A"; T = "export-10"; DL = "/scratch/amartyavarma/aloespk-lora/dl-A10"
api = HfApi(); fs = [f for f in api.list_repo_files(R) if f.startswith(T + "/")]
info = {p.path: p for p in api.get_paths_info(R, fs)}
os.makedirs(f"{DL}/{T}", exist_ok=True)
def one(f):
    p = hf_hub_download(R, f, local_dir=f"{DL}/.dl")
    lfs = getattr(info[f], "lfs", None)
    if lfs is not None:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        assert h.hexdigest() == lfs.sha256, f
    else:
        assert os.path.getsize(p) == info[f].size, f
    shutil.move(p, f"{DL}/{T}/" + f.split("/", 1)[1]); return f
with ThreadPoolExecutor(8) as ex: got = list(ex.map(one, fs))
shutil.rmtree(f"{DL}/.dl", ignore_errors=True)
print(f"delta {T}: {len(got)} files from private HF {R}, sha256 == LFS")
PY
ls $DL/export-10
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I $L/tools/merge_lora_export.py $BASE $DL/export-10 $L/merged-A10
ls $L/merged-A10 | head; cat $L/merged-A10/COMPLETE

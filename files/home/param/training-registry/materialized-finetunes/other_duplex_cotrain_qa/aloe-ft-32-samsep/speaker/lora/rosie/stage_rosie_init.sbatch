#!/bin/bash
#SBATCH -J aloerosie-stage -p all -N1 -n1 -c 16 --mem=32G -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/rosie-stage-%j.log
# Rosie FT (user go 10-25 via main 7e): room + init on slinky-4. (1) delete my staged 16861 copy (HF-backed staging, main's standing go for
# caches/staging with HF records); (2) stage the FULL vox17127 export-249 (all 64 shards + rep/meta) from private HF, sha256 == LFS, and write the
# exact fp32 vgain theta (-5.621907711029053, from 29345's end save dcp-249; == logged s 0.0036177329326430177) into meta.json.
set -euo pipefail
source /data/home/amartyavarma/ppm/scratch_guard.sh
d=/scratch/amartyavarma/aloe-samsep-16861-init
if [ -f $d/COMPLETE ]; then sz=$(du -sh $d | cut -f1); guard_rm $d && echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] ALOEFT32 (Rosie FT room, standing go: staging with HF records): deleted slinky-4:$d ($sz, staged converted copy of PPM dcp-16861; source private HF InternalCan/ppm-v1-b/dcp-16861 + ledger base/ctl/ledger-16861-slinky-4.json)" | tee -a /data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log; fi
echo "free after delete: $(df -BG --output=avail /scratch | tail -1)"
cd /tmp
/data/home/amartyavarma/venvs/duplex/bin/python -I - <<'PY'
import os, json, hashlib, shutil, time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download
R, T = "InternalCan/ppm-aloe-samsep-vox17127-final", "export-249"; DEST = "/scratch/amartyavarma/aloerosie-init/export-249"
api = HfApi(); fs = [f for f in api.list_repo_files(R) if f.startswith(T + "/") and f.count("/") == 1]
need = [f for f in fs if f.split("/")[1].startswith("rank") or f.split("/")[1] in ("rep.pt", "meta.json", "COMPLETE")]
info = {p.path: p for p in api.get_paths_info(R, need)}
part, dl = DEST + ".part", DEST + ".dl"; os.makedirs(part, exist_ok=True); os.makedirs(dl, exist_ok=True); t0 = time.time()
def one(f):
    p = hf_hub_download(R, f, local_dir=dl); lfs = getattr(info[f], "lfs", None)
    if lfs is not None:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        assert h.hexdigest() == lfs.sha256, f
    else:
        assert os.path.getsize(p) == info[f].size, f
    shutil.move(p, part + "/" + f.split("/")[1]); return f
with ThreadPoolExecutor(16) as ex: got = list(ex.map(one, need))
nr = sum(1 for g in got if g.split("/")[1].startswith("rank"))
m = json.load(open(part + "/meta.json")); assert m["format"] == "ppm-local-bf16-shards-v1" and int(m["world"]) == 64 == nr, (m.get("world"), nr)
m["vgain_theta_fp32"] = -5.621907711029053; m["vox_a_scale"] = 0.0036177329326430177
m["theta_source"] = "29345 end save dcp-249 (fp32), == logged s at step 249 (speaker prep 10-24)"
json.dump(m, open(part + "/meta.json", "w"), indent=1)
shutil.rmtree(dl, ignore_errors=True); os.rename(part, DEST)
print(f"staged {R}/{T} -> {DEST}: {len(got)} files ({nr} rank shards), sha256 == LFS, {time.time() - t0:.0f}s; theta written")
PY
echo "free after stage: $(df -BG --output=avail /scratch | tail -1)"

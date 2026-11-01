#!/bin/bash
#SBATCH -J hf-pdx498 -p all -w slinky-18 -x slinky-9 -N1 -n1 -c 8 --mem=64G --gres=none -t 08:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498/hf-export498-%j.out
# Pose-duplex FT (job 30074) consolidated bf16 export export-poseduplex-v5-498 (/data, 88.5 GB) -> NEW PRIVATE HF model repo (HF ONLY, never GCS).
# = release_pipeline.sbatch's upload step: hfb_up.py (symlink farm on node-local /scratch, upload_large_folder, retries, verify = private +
# every file present with its size + HF LFS sha256 == upload-time sha256 + 3 files streamed back), then hfb_sha_all.py (fresh sha256 of EVERY
# file vs HF), then an anonymous-access check (must be refused). Source read-only, never deleted. CPU only, nice 10, 4 workers.
set -u
SRC=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/export-poseduplex-v5-498
REPO=${HF498_REPO:-InternalCan/ppm-aloe-poseduplex-v5-498}; DEST=export-poseduplex-v5-498; TAG=pdx-v5-498-export
PY=/data/home/amartyavarma/venvs/duplex/bin/python; HFB=/data/home/amartyavarma/setup-logs/hfb
S=/scratch/amartyavarma/hfb/pdx-v5-498-export; F=$S/farm; O=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
step() { echo "== [$(pt)] $*"; }
case "$(hostname -s)" in slinky-9|slinky-10) step "ABORT: never on $(hostname -s)"; exit 1;; esac
[ -f $SRC/COMPLETE ] && [ "$(cat $SRC/COMPLETE)" = 498 ] || { step "ABORT: $SRC/COMPLETE != 498"; exit 2; }
FREE=$(df -B1 --output=avail /scratch | tail -1)
[ "$FREE" -ge 40000000000 ] || { step "ABORT: /scratch free $FREE < 40 GB (30 GB floor + upload-cache margin; farm = symlinks)"; exit 5; }
mkdir -p $S $O
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$S/xet-cache HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HFB_OWNER=$SLURM_JOB_ID
step "job $SLURM_JOB_ID on $(hostname -s): $SRC ($(ls -A $SRC | wc -l) files, $(du -sb $SRC | cut -f1) B) -> private $REPO/$DEST/; /scratch free $((FREE / 1000000000)) GB"
# the repo must be NEW (absent) or one this script already started (only files under $DEST/ + _backup/ + .gitattributes)
nice -n 10 $PY -I - "$REPO" "$DEST" <<'P' || { step "ABORT: $REPO exists with unrelated content"; exit 2; }
import sys
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
api = HfApi(); r, d = sys.argv[1], sys.argv[2]
print("hf user", api.whoami()["name"])
try:
    i = api.repo_info(r)
except RepositoryNotFoundError:
    print("repo absent: will be created private"); sys.exit(0)
fs = [e.path for e in api.list_repo_tree(r, recursive=True) if hasattr(e, "blob_id")]
other = [p for p in fs if not (p.startswith(d + "/") or p.startswith("_backup/") or p == ".gitattributes")]
print("repo exists, private", i.private, "files", len(fs), "unrelated", other[:5]); sys.exit(1 if other or not i.private else 0)
P
echo "{\"dirs\": [[\"$SRC\", \"$DEST\"]], \"exclude\": [\"HF_BACKUP.json\"]}" > $S/spec.json
nice -n 10 $PY $HFB/hfb_up.py --repo $REPO --type model --farm $F --spec $S/spec.json --manifest-name MANIFEST.$TAG.tsv --tag $TAG \
  --require $SRC/COMPLETE --workers 4 --max-fails 6 --verify-out $O; rc=$?
step "hfb_up rc $rc (0 = upload + VERIFY PASS)"
[ $rc = 0 ] || exit 6
nice -n 10 $PY $HFB/hfb_sha_all.py --repo $REPO --type model --farm $F --tag $TAG --threads 4 --verify-out $O; rc=$?
step "hfb_sha_all rc $rc (0 = fresh sha256 of every file == HF)"
[ $rc = 0 ] || exit 7
nice -n 10 $PY -I - "$REPO" <<'P'; rc=$?
import sys
from huggingface_hub import HfApi
r = sys.argv[1]
assert HfApi().repo_info(r).private is True, "not private"
try:
    HfApi(token=False).repo_info(r); print("ANON CAN READ", r); sys.exit(8)
except Exception as e:
    print("private True; anonymous access refused:", type(e).__name__)
P
step "privacy check rc $rc"
[ $rc = 0 ] || exit 8
step "DONE: $REPO/$DEST/ uploaded + verified (hfb_up VERIFY PASS + sha256all PASS + private, anonymous refused); source kept"
exit 0

#!/bin/bash
#SBATCH -J vhf-pdx498 -p all -w slinky-18 -x slinky-9 -N1 -n1 -c 8 --mem=24G --gres=none -t 02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498/verify-hf-export498-%j.out
# READ-ONLY re-verify of ppmguard job 30215's upload: /data export-poseduplex-v5-498 vs PRIVATE InternalCan/ppm-aloe-poseduplex-v5-final/export-498-dcp/.
# hfb_sha_all.py over a fresh symlink farm (export-498-dcp/<file> -> the /data file; HF_BACKUP.json excluded as the watcher does): fresh sha256 of
# EVERY file vs HF LFS sha256 (small files streamed back), count + size + private; plus: no extra / missing files under export-498-dcp/ on HF,
# anonymous access refused. Uploads nothing.
SRC=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/export-poseduplex-v5-498
REPO=InternalCan/ppm-aloe-poseduplex-v5-final; DEST=export-498-dcp; TAG=export-498-dcp-recheck
PY=/data/home/amartyavarma/venvs/duplex/bin/python; HFB=/data/home/amartyavarma/setup-logs/hfb
O=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498; F=/scratch/amartyavarma/hfb/pdx-v5-498-recheck-farm
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
step() { echo "== [$(pt)] $*"; }
case "$(hostname -s)" in slinky-9|slinky-10) step "ABORT: never on $(hostname -s)"; exit 1;; esac
source /data/home/amartyavarma/ppm/scratch_guard.sh
[ -d $F ] && guard_rm $F
mkdir -p $F/$DEST
for f in $(ls -A $SRC); do [ "$f" = HF_BACKUP.json ] && continue; ln -s $SRC/$f $F/$DEST/$f; done
step "farm $F: $(ls -A $F/$DEST | wc -l) files -> $REPO/$DEST/"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
nice -n 10 $PY $HFB/hfb_sha_all.py --repo $REPO --type model --farm $F --tag $TAG --threads 4 --verify-out $O; rc=$?
step "hfb_sha_all rc $rc"
nice -n 10 $PY -I - "$REPO" "$DEST" "$F/$DEST" <<'P'; rc2=$?
import os, sys
from huggingface_hub import HfApi
r, d, farm = sys.argv[1:4]
api = HfApi()
priv = api.repo_info(r).private
hf = {e.path[len(d) + 1:]: e.size for e in api.list_repo_tree(r, path_in_repo=d, recursive=True) if hasattr(e, "blob_id")}
loc = {f: os.path.getsize(os.path.join(farm, f)) for f in os.listdir(farm)}
extra, miss = sorted(set(hf) - set(loc)), sorted(set(loc) - set(hf))
try:
    HfApi(token=False).repo_info(r); anon = "ANON CAN READ"
except Exception as e:
    anon = f"anonymous refused ({type(e).__name__})"
print(f"HF {d}/: {len(hf)} files {sum(hf.values())} B; local {len(loc)} files {sum(loc.values())} B; missing {miss}; extra {extra}; private {priv}; {anon}")
sys.exit(0 if priv and not extra and not miss and anon.startswith("anonymous refused") else 3)
P
step "listing / privacy check rc $rc2"
guard_rm $F
[ $rc = 0 ] && [ $rc2 = 0 ] && { step "PASS: $REPO/$DEST/ == /data export (every file fresh sha256), private"; exit 0; }
step "FAIL"; exit 3

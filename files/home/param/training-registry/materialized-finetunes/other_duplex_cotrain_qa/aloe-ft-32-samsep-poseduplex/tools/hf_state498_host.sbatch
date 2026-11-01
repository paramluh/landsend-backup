#!/bin/bash
#SBATCH -p all -N1 -n1 -c 8 --mem=48G --gres=none -t 08:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498/state-%x-%j.out
# Pose-duplex FT (job 30074) node-local FULL STATE dcp-498 of ONE host -> PRIVATE HF model repo, path dcp-498/<host>/ (HF ONLY, never GCS).
# = ppmvox/companions/hf_export_host.sbatch (hfb_up.py symlink farm, upload_large_folder, retries, verify = private + every file + size +
# HF LFS sha256 vs upload-time sha256; then hfb_sha_all.py fresh sha256 of EVERY file vs HF) + a cross-check of every HF sha256 against the
# task-1 verify json (the sha256 recorded when each rank file's digest was re-verified). Source read-only, NEVER deleted. CPU only, nice 10,
# 4 upload workers (hosts may get GPU jobs).   sbatch -w <host> -J st498-<host> hf_state498_host.sbatch
H=$(hostname -s); K=dcp-498
SRC=/scratch/amartyavarma/aloe-ft-32-samsep/saves-poseduplex/ppm-aloe-poseduplex-v5-j30074/$K
R=${STATE498_REPO:-InternalCan/ppm-aloe-poseduplex-v5-498-state}; D=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python
O=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/backup-498; F=/scratch/amartyavarma/hfb/pdx-v5-state-$K-$H; SP=$O/spec-state-$K-$H.json
TAG=$K-$H; VJ1=$O/verify-dcp498-$H.json
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
step() { echo "== [$(pt)] $H $*"; }
T0=$(date +%s)
case "$H" in slinky-9|slinky-10) step "ABORT: never on $H"; exit 1;; esac
[ -f $SRC/COMPLETE ] && [ "$(cat $SRC/COMPLETE)" = 498 ] || { step "ABORT: $SRC/COMPLETE missing or != 498"; exit 2; }
[ -f $VJ1 ] && grep -q '"PASS": true' $VJ1 || { step "ABORT: task-1 verify json $VJ1 missing or not PASS"; exit 2; }
FREE=$(df -B1 --output=avail /scratch | tail -1)
[ "$FREE" -ge 32000000000 ] || { step "ABORT: /scratch free $FREE < 32 GB (30 GB floor + upload-cache margin; the farm is symlinks)"; exit 5; }
XC=/scratch/amartyavarma/hfb/xet-cache-pdx498-$H   # 10-25 fix: the xet cache must live OUTSIDE the farm (inside it, it was uploaded as xet-cache/)
source /data/home/amartyavarma/ppm/scratch_guard.sh
[ -e $F/xet-cache ] && { guard_rm $F/xet-cache || { step "ABORT: cannot clear the old in-farm xet cache"; exit 5; }; step "removed the old in-farm xet cache $F/xet-cache"; }
mkdir -p $F $O $XC
echo "{\"dirs\": [[\"$SRC\", \"$K/$H\"]]}" > $SP
step "start job $SLURM_JOB_ID: $SRC ($(ls -A $SRC | wc -l) files, $(du -sb $SRC | cut -f1) B) -> private $R/$K/$H/; /scratch free $((FREE / 1000000000)) GB"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$XC HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HFB_OWNER=$SLURM_JOB_ID OMP_NUM_THREADS=2
UP="$PY $D/hfb_up.py --repo $R --type model --farm $F --spec $SP --manifest-name MANIFEST.$K.$H.tsv --tag $TAG --require $SRC/COMPLETE --workers 4 --verify-out $O"
bo=60; att=0; vfail=0; hrc=0
while true; do
  att=$((att+1))
  if ! nice -n 10 $UP --verify-only --no-verify; then step "attempt $att: repo create / farm failed; retry ${bo}s"; [ $att -ge 8 ] && { hrc=2; break; }; sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  step "attempt $att: uploading"
  nice -n 10 $PY $D/hfb_up.py --child --repo $R --type model --farm $F --spec $SP --workers 4; rc=$?
  if [ $rc -ne 0 ]; then step "attempt $att: upload rc $rc; retry ${bo}s"; [ $att -ge 8 ] && { hrc=1; break; }; sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  step "attempt $att: upload done after $(( $(date +%s) - T0 )) s; verifying"
  nice -n 10 $UP --verify-only; vrc=$?
  if [ $vrc -eq 0 ]; then
    nice -n 10 $PY $D/hfb_sha_all.py --repo $R --type model --farm $F --tag $TAG --threads 2 --verify-out $O; vrc=$?
    [ $vrc -eq 0 ] && break
  fi
  vfail=$((vfail+1)); step "VERIFY FAIL rc $vrc ($vfail)"; [ $vfail -ge 4 ] && { hrc=3; break; }; sleep $bo
done
if [ $hrc -eq 0 ]; then   # every HF sha256 == the sha256 recorded by the task-1 digest verify (same bytes that passed the digest check)
  nice -n 10 $PY -I - "$O/${R//\//__}__$TAG-sha256all__$H.json" "$VJ1" "$K/$H" <<'P'; hrc=$?
import json, sys
s, v, pre = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
want = {r["file"]: r["sha256"] for r in v["files"]}
want.update(v["small_files_sha256"])
pf = s["per_file"]
bad = [f for f, h in want.items() if pf.get(f"{pre}/{f}", {}).get("sha256_hf") != h]
print(f"cross-check vs task-1: {len(want) - len(bad)}/{len(want)} files' HF sha256 == the digest-verified sha256; private {s['private']}; bad {bad}")
sys.exit(0 if not bad and s["PASS"] and s["private"] else 3)
P
fi
[ $hrc -eq 0 ] && guard_rm "$F" "$XC"   # own symlink farm + upload / xet caches only (never the save)
step "$K/$H HF backup rc $hrc (0 = hfb_up VERIFY PASS + sha256all PASS + task-1 cross-check, private $R) after $(( $(date +%s) - T0 )) s"
exit $hrc

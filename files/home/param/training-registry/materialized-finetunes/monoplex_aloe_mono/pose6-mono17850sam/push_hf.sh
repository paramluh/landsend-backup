#!/bin/bash
#SBATCH -J pose6-m17850-hf -p all -N1 -n1 -c 4 --mem=16G --gres=none -t 1-00:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/pose6/mono17850sam/logs/push_hf-%j.log
# pose6 mono17850: push the FINAL consolidated export to the PRIVATE HF repo InternalCan/ppm-pose6-mono17850sam-final, path export-357-dcp/, and verify.
# = the 27757 chain's method (aloe-v2pure-main-17850/hf_final_export.sbatch -> setup-logs/hfb/hfb_up.py; repo naming ppm-<run>-final; its hf_one
# retry / back-off and vj_ok checks copied verbatim): upload_large_folder from a symlink farm (the export is only READ; nothing written into it),
# verify = repo private + every local file on HF with the same size + HF LFS sha256 == upload-time local sha256 of EVERY file + 3 files streamed
# back and sha256'd. NEVER GCS. The token is read by huggingface_hub from its default path; never printed. Deletes nothing.
# Submit (small CPU allocation, after 31101 resumed, on any node with 4 free CPUs, never slinky-9):
#   sbatch -w <node> /data/home/amartyavarma/ppm/pose6/mono17850sam/push_hf.sh [<export path>]     (default mono17850/export-357)
# Farm + upload cache + xet cache under /data/home/amartyavarma/ppm/pose6/mono17850sam/hf (small; no node /scratch writes). Result line ->
# logs/BACKUP_LOG.txt; on PASS logs/HF_BACKUP-export-357.json (repo, path, verify json). rc 0 only on VERIFY PASS.
set -uo pipefail
[ -n "${SLURM_JOB_ID:-}" ] || { echo "ABORT: submit with sbatch (never on the login node)"; exit 1; }
M=/data/home/amartyavarma/ppm/pose6/mono17850sam
X=$(readlink -f "${1:-$M/export-357}")
R=InternalCan/ppm-pose6-mono17850sam-final; TAG=export-357-dcp; WK=4; BO=60
UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py; PY=/data/home/amartyavarma/venvs/duplex/bin/python
LD=$M/logs; FB=$M/hf; VO=$LD/hf_verify; SPD=$LD/hf_spec; BLOG=$LD/BACKUP_LOG.txt; H=$(hostname -s); L=$LD/push_hf-$SLURM_JOB_ID-$H.detail.log
mkdir -p $VO $SPD $FB/xet-cache
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$FB/xet-cache HFB_OWNER=$SLURM_JOB_ID
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
blog() { echo "[$(pt)] $*" | tee -a $L >> $BLOG; }
[ -f "$X/COMPLETE" ] && [ -f "$X/.metadata" ] || { blog "ABORT: $X is not a COMPLETE consolidated export; job $SLURM_JOB_ID"; exit 1; }
case "$X" in /data/*) ;; *) blog "ABORT: $X not on /data; job $SLURM_JOB_ID"; exit 1;; esac

vj_ok() {   # = 27757's: verify json fresh (>= $3) and clean, >= $2 files sha-checked
  [ -f "$1" ] && [ "$(stat -c %Y "$1")" -ge "$3" ] || return 1
  $PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); n=int(sys.argv[2])
ok = j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch") \
     and int(j.get("sha_checked_vs_upload_cache", 0)) >= n
sys.exit(0 if ok else 1)' "$1" "$2"
}
hf_one() {  # = 27757's hf_one: <src dir> <tag> <manifest> <farm> <require> <exclude json>; rc 0 VERIFY PASS, 1 upload, 2 repo / farm, 3 verify
  local W=$1 TAG=$2 MAN=$3 F=$4 REQ=$5 EXC=$6 S UP bo=$BO att=0 vfail=0 hrc=0 rc vrc
  S=$SPD/spec-$TAG-$H.json
  mkdir -p "$F"
  printf '{"dirs": [["%s", "%s"]], "exclude": %s}\n' "$W" "$TAG" "$EXC" > $S
  echo "[$(pt)] $H start $W ($(ls -A $W | wc -l) files, $(du -sbL $W | cut -f1) B) -> private $R/$TAG/ (farm $F, manifest $MAN)" >> $L
  UP="$PY $UPY --repo $R --type model --farm $F --spec $S --manifest-name $MAN --tag $TAG --require $REQ --workers $WK --verify-out $VO"
  while true; do
    att=$((att+1))
    if ! nice -n 10 $UP --verify-only --no-verify >> $L 2>&1; then
      echo "[$(pt)] $H $TAG attempt $att: repo create / farm failed; retry ${bo}s" >> $L; [ $att -ge 8 ] && { hrc=2; break; }
      sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
    nice -n 10 $PY $UPY --child --repo $R --type model --farm $F --spec $S --workers $WK >> $L 2>&1; rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(pt)] $H $TAG attempt $att: upload rc $rc; retry ${bo}s" >> $L; [ $att -ge 8 ] && { hrc=1; break; }
      sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
    nice -n 10 $UP --verify-only >> $L 2>&1; vrc=$?
    [ $vrc -eq 0 ] && break
    vfail=$((vfail+1)); echo "[$(pt)] $H $TAG VERIFY FAIL rc $vrc ($vfail)" >> $L; [ $vfail -ge 4 ] && { hrc=3; break; }; sleep $bo
  done
  return $hrc
}

echo "== push_hf job $SLURM_JOB_ID on $H $(pt): $X -> private $R/$TAG/ (detail $L)"
npt=$(ls $X/*.distcp 2>/dev/null | wc -l); nf=$(ls -A $X | wc -l); sz=$(du -sh $X | cut -f1); t0=$(date +%s)
hf_one "$X" "$TAG" "MANIFEST.$TAG.tsv" "$FB/farm-$TAG" "$X/COMPLETE" '[]'; r=$?
VJ=$VO/${R//\//__}__${TAG}__${H}.json
if [ $r = 0 ] && vj_ok $VJ $npt $t0; then
  printf '{"repo": "%s", "path": "%s/", "private": true, "src": "%s", "files": %s, "verify": "%s", "time": "%s", "job": "%s"}\n' \
    "$R" "$TAG" "$X" "$nf" "$VJ" "$(pt)" "$SLURM_JOB_ID" > $LD/HF_BACKUP-export-357.json
else [ $r = 0 ] && r=3; fi
blog "$TAG HF upload rc $r (0 = VERIFY PASS: private, sizes, LFS sha256 of every file, 3 streamed back) $X $nf files $sz -> private $R/$TAG/ job $SLURM_JOB_ID"
echo "== exit $r $(pt)"; exit $r

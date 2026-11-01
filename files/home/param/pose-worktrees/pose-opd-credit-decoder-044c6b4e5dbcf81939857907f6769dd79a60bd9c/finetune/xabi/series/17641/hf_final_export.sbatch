#!/bin/bash
#SBATCH -J aloev2p17641-hffinal -p all --ntasks-per-node=1 -c 4 --mem=40G --gres=none -t 1-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/hf_final-%j.log
# aloe-v2pure-17641 FINAL export backup (10-22) = aloe-v2pure-13533/hf_final_export.sbatch (job 26237) with ONLY the run name / dirs / repo changed:
#   the node-local export-200 shards on this FT's hosts (disjoint files per host) -> private HF InternalCan/ppm-aloe-v2pure-17641-final, path
#   export-200/. Chained afterok on the training job, one task per training host: sbatch --dependency=afterok:<train> -N<n> -w <hosts> <this>.
#   Never GCS. Nothing deleted (INTER empty; the final export stays node-local).
# (template header follows) aloe-v2vox-asr-w15973 (10-20): = aloe-v2vox-raw-w14503/hf_local_exports.sbatch (job 25965) with ONLY the run name / dirs / repo
#   (private InternalCan/ppm-aloe-v2vox-asr-w15973) changed; 8 hosts; the consolidated /data export-253 exists only if /data had room above the floor.
# aloe-v2vox-raw-w14503 (10-20): HF-ONLY backup of the run's exports (NEVER GCS) = aloe-v2vox-raw/gcs_local_exports.sbatch (job 23082) with gcs_up.py
# replaced by hfb_up.py (aloe-nextvox-6242/tools/hf_up.sbatch's recipe: symlink farm on node scratch, upload_large_folder, retry / back-off, verify).
# One srun task per training node, chained after the training job:
#   sbatch --dependency=afterok:<train job> -N3 -w slinky-7,slinky-15,slinky-18 hf_local_exports.sbatch
#  * every node uploads each node-local export dir /scratch/amartyavarma/aloe-v2vox-raw-w14503/exports/export-N (its 8 rank<R>.pt; + rep.pt /
#    meta.json / COMPLETE on rank 0's node) to the PRIVATE HF model repo InternalCan/ppm-aloe-v2vox-raw-w14503, path export-N/. The 3 nodes' files
#    are disjoint; per host: farm /scratch/amartyavarma/hfb/aloe-v2vox-raw-w14503-export-N-<host>, manifest _backup/MANIFEST.export-N.<host>.tsv,
#    spec logs/hf_spec/spec-export-N-<host>.json, verify json logs/hf_verify/InternalCan__ppm-aloe-v2vox-raw-w14503__export-N__<host>.json.
#  * verify = hfb_up.py's: repo private, every local file on HF with the same size, HF LFS sha256 == upload-time local sha256 of every LFS file,
#    3 files streamed back + sha256'd vs the local file (the only download).
#  * ONLY for the 5 INTERMEDIATE exports (42 84 126 169 211), the node-local copy is deleted (scratch_guard guard_rm) when ALL hold: hfb_up rc 0
#    (VERIFY PASS), a verify json written by this attempt with PASS / private / n_missing 0 / no size or sha mismatch / every *.pt sha-checked,
#    8 rank*.pt, no *.tmp, and on rank 0's node rep.pt + meta.json + COMPLETE. export-253 (and anything else) stays node-local.
#  * node 0 (first host sorted -V = training rank 0's node, as in the trainer) also uploads the consolidated bf16 DCP export
#    /data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503/export-253 as path export-253-dcp/ and writes HF_BACKUP.json into it on PASS.
#  * one line per result appended to logs/BACKUP_LOG.txt; per-node detail in logs/hf_local-<job>-<host>.log. Sources are only read, except the
#    guarded deletions and HF_BACKUP.json. Exit rc 0 only if every upload passed and every expected export was found.
# Dry test (uploads nothing, nothing real touched): HFLX_DRY=1 HFLX_ROOT=/scratch/amartyavarma/hflx-dry-<x> (fake exports/, weka/export-253, logs/,
#   hfb/ under it) HFLX_UP=<hfb_up.py stub> [HFLX_HOSTS=a,b,c] [HFLX_GUARD_LIST=<real list + extra lines>]; harness tests/hf_local_dry.sbatch.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2pure-17641
if [ "${HFLX_DRY:-0}" = 1 ]; then
  : "${HFLX_ROOT:?}" "${HFLX_UP:?}"
  case "$HFLX_ROOT" in /scratch/amartyavarma/hflx-dry-*) ;; *) echo "ABORT: dry root $HFLX_ROOT not /scratch/amartyavarma/hflx-dry-*"; exit 1;; esac
  EXR=$HFLX_ROOT/exports; WX=$HFLX_ROOT/weka/export-253; LD=$HFLX_ROOT/logs; FB=$HFLX_ROOT/hfb; UPY=$HFLX_UP; BO=1
  R=InternalCan/DRYRUN-no-upload; HOSTS=${HFLX_HOSTS:-}
  [ -n "${HFLX_GUARD_LIST:-}" ] && export SCRATCH_GUARD_LIST=$HFLX_GUARD_LIST
else
  EXR=/scratch/amartyavarma/aloe-v2pure-17641/exports; WX=$Q/export-200-absent; LD=$Q/logs; FB=/scratch/amartyavarma/hfb
  UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py; BO=60; R=InternalCan/ppm-aloe-v2pure-17641-final; HOSTS=
  unset SCRATCH_GUARD_LIST                                   # the guard's own default list (PROTECTED_SCRATCH.txt), never an inherited one
fi
[ -n "$HOSTS" ] || HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
DRY=${HFLX_DRY:-0}; BLOG=$LD/BACKUP_LOG.txt; VO=$LD/hf_verify; SPD=$LD/hf_spec; PY=/data/home/amartyavarma/venvs/duplex/bin/python
INTER=""; STEPS="200"; NRANK=8; WK=8; NAME=aloe-v2pure-17641
mkdir -p $VO $SPD
export Q EXR WX LD FB UPY BO R HOSTS DRY BLOG VO SPD PY INTER STEPS NRANK WK NAME

pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
blog() { echo "[$(pt)] $*" | tee -a $L >> $BLOG; }          # one result line -> BACKUP_LOG.txt (+ the node log)

# verify json fresh (written at/after $3) and clean: PASS, private, nothing missing, no size / sha mismatch, >= $2 files sha-checked (every *.pt / *.distcp)
vj_ok() {
  [ -f "$1" ] && [ "$(stat -c %Y "$1")" -ge "$3" ] || return 1
  $PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); n=int(sys.argv[2])
ok = j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch") \
     and int(j.get("sha_checked_vs_upload_cache", 0)) >= n
sys.exit(0 if ok else 1)' "$1" "$2"
}

# hf_one <src dir> <repo path / tag> <manifest name> <farm> <require path> <exclude json list>: upload + verify with retry / back-off
# (= hf_up.sbatch's loop). rc 0 = VERIFY PASS, 1 upload failed, 2 repo create / farm failed, 3 verify failed.
hf_one() {
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

node_main() {
  set -uo pipefail
  H=$(hostname -s); L=$LD/hf_local-$SLURM_JOB_ID-$H.log
  local I ER d T r rc=0 nrk npt ntmp t0 VJ why sz s f nf
  I=$(echo "$HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
  [ -n "$I" ] || { echo "ABORT: $H not in $HOSTS"; exit 1; }
  source /data/home/amartyavarma/ppm/scratch_guard.sh
  export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$FB/xet-cache HFB_OWNER=$SLURM_JOB_ID
  mkdir -p $FB/xet-cache
  echo "[$(pt)] $H node $I of $HOSTS (dry $DRY): $EXR -> private $R; guard list ${SCRATCH_GUARD_LIST}" >> $L
  if [ -L "$EXR" ] || [ ! -d "$EXR" ]; then blog "$H ABORT: $EXR missing or a symlink (expected a real node-local dir); job $SLURM_JOB_ID"; exit 1; fi
  ER=$(readlink -f "$EXR")
  case "$ER" in /scratch/*) ;; *) blog "$H ABORT: $EXR resolves to $ER (not node-local /scratch); job $SLURM_JOB_ID"; exit 1;; esac
  for s in $STEPS; do [ -d $ER/export-$s ] || { blog "$H export-$s MISSING on node-local scratch ($ER); job $SLURM_JOB_ID"; rc=1; }; done
  for d in $ER/export-200; do                              # FINAL export only (intermediates untouched)
    T=$(basename $d)
    if [ -L "$d" ] || [ ! -d "$d" ]; then blog "$H $T is not a real dir: skipped (rc 1); job $SLURM_JOB_ID"; rc=1; continue; fi
    nrk=$(ls $d/rank*.pt 2>/dev/null | wc -l); npt=$(ls $d/*.pt 2>/dev/null | wc -l); ntmp=$(find $d -name '*.tmp' | wc -l)
    nf=$(ls -A $d | wc -l); sz=$(du -sh $d | cut -f1)
    t0=$(date +%s)
    hf_one "$d" "$T" "MANIFEST.$T.$H.tsv" "$FB/$NAME-$T-$H" "$d" '[]'; r=$?
    VJ=$VO/${R//\//__}__${T}__${H}.json
    blog "$T HF upload rc $r (0 = VERIFY PASS) $H node-local $nf files $sz -> private $R/$T/ job $SLURM_JOB_ID"
    [ $r = 0 ] || rc=1
    case " $INTER " in
      *" $T "*)
        why=""
        [ $r = 0 ] || why="upload / verify rc $r"
        [ $r = 0 ] && { vj_ok $VJ $npt $t0 || why="verify json $VJ not fresh / clean"; }
        [ $nrk = $NRANK ] || why="$why; $nrk rank*.pt != $NRANK"
        [ $ntmp = 0 ] || why="$why; $ntmp *.tmp"
        if [ $I = 1 ]; then for f in rep.pt meta.json COMPLETE; do [ -f $d/$f ] || why="$why; no $f (rank 0's node)"; done; fi
        if [ -n "$why" ]; then blog "$H $T node-local copy KEPT: ${why#; }"
        elif guard_rm "$d" 2>> $L; then blog "$H $T node-local copy DELETED ($sz, $d) after HF VERIFY PASS"
        else blog "$H $T node-local copy KEPT: scratch_guard refused $d"; rc=1; fi;;
      *) blog "$H $T node-local copy kept (final / not intermediate)";;
    esac
  done
  if [ $I = 1 ] && false; then                             # no consolidated /data export (refused)
    if [ -f $WX/COMPLETE ]; then
      npt=$(ls $WX/*.distcp 2>/dev/null | wc -l); nf=$(ls -A $WX | wc -l); sz=$(du -sh $WX | cut -f1)
      t0=$(date +%s)
      hf_one "$WX" export-253-dcp MANIFEST.export-253-dcp.tsv "$FB/$NAME-export-253-dcp-$H" "$WX/COMPLETE" '["HF_BACKUP.json"]'; r=$?
      VJ=$VO/${R//\//__}__export-253-dcp__${H}.json
      if [ $r = 0 ] && vj_ok $VJ $npt $t0; then
        printf '{"repo": "%s", "path": "export-253-dcp/", "private": true, "verify": "%s", "time": "%s", "job": "%s"}\n' "$R" "$VJ" "$(pt)" "$SLURM_JOB_ID" > $WX/HF_BACKUP.json
      else rc=1; [ $r = 0 ] && r=3; fi
      blog "export-253-dcp HF upload rc $r (0 = VERIFY PASS) $H $WX $nf files $sz -> private $R/export-253-dcp/ job $SLURM_JOB_ID$( [ $r = 0 ] && echo '; HF_BACKUP.json written')"
    else blog "export-253-dcp NOT uploaded: $WX/COMPLETE missing (final /data export refused by the 300 GB floor? node-local export-253 is the final copy); job $SLURM_JOB_ID"; rc=1; fi
  fi
  echo "[$(pt)] $H done rc $rc; $(df -BG /scratch | tail -1)" >> $L
  exit $rc
}
export -f pt blog vj_ok hf_one node_main

echo "== hf_local_exports job $SLURM_JOB_ID $(TZ=America/Los_Angeles date '+%F %T PT') hosts $HOSTS dry $DRY repo $R"
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c ${SLURM_CPUS_PER_TASK:-4} --kill-on-bad-exit=0 bash -c node_main
rc=$?
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; tail -n 40 $BLOG 2>/dev/null | grep -F "job $SLURM_JOB_ID"
exit $rc

#!/bin/bash
#SBATCH -J ttconvo-hfb-final -p all --ntasks-per-node=1 -c 4 --mem=40G --gres=none -t 0-06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/ttconvo-ft/logs/hf_final-%j.log
# ttconvo-ft (10-21): HF-ONLY backup (NEVER GCS) of the run's node-local weight exports = aloe-v2vox-raw-w14503/hf_local_exports.sbatch
# (hfb_up.py recipe: symlink farm on node scratch, upload_large_folder, retry / back-off, verify) with these changes:
#  * runs DURING training, started by hf_backup_watch.py (job ttconvo-hfb-watch, slinky-15) as soon as an export is COMPLETE on slinky-0:
#      TTC_LAUNCH=overlap TTC_JOB=<train job> TTC_EXPORTS=<N> hf_local_exports.sh  ->  one CPU-only overlap step inside the training job:
#      srun --jobid=<train job> --overlap -N16 --ntasks-per-node=1 -c 4 --mem=40G --gres=none --cpu-bind=none nice -n 19 ionice -c3 ... (node_main)
#    gentle like the PPM uploads beside training (setup-logs/hfb/hfb_train_node.sh): 2 upload workers, HF_XET_FIXED_UPLOAD_CONCURRENCY 4,
#    adaptive concurrency off, MAX_CONCURRENT_FILE_INGESTION 2, nice 19 / ionice idle, and the PPM step-time throttle (hfb_throttle.py's
#    rules, run inside the watcher): hfb_up.py --control hf_backup/control/<host>.workers (0 = paused) instead of a bare --child.
#  * fallback (training job gone, export still not on HF, the 16 nodes free): the watcher submits THIS FILE as a normal CPU-only job
#      sbatch -p all --gres=none -N16 -w <16 hosts> --ntasks-per-node=1 -c 4 --mem=40G  (TTC_LAUNCH=alloc): exact Aloe recipe (8 workers,
#    no throttle); the batch shell (on slinky-0 = rank 0's node) first keeps only the requested exports that are COMPLETE there.
#  * every host uploads its node-local /scratch/amartyavarma/ttconvo-ft/exports/export-N (its 8 rank<R>.pt, R = 8 x host index + 0..7; + rep.pt /
#    meta.json / COMPLETE on slinky-0) to the PRIVATE HF model repo InternalCan/ppm-ttconvo-ft, path export-N/ (the 16 hosts' files are
#    disjoint); farm /scratch/amartyavarma/hfb/ttconvo-ft-export-N-<host>, manifest _backup/MANIFEST.export-N.<host>.tsv, spec / verify json
#    under hf_backup/spec, hf_backup/verify; per-host log logs/hf_up-export-N-<host>.log.
#  * verify = hfb_up.py's: repo private, every local file on HF with the same size, HF LFS sha256 == upload-time local sha256 of every LFS file,
#    3 files streamed back + sha256'd vs the local file (the only download). PASS + a fresh clean verify json -> hf_backup/state/export-N/<host>.done
#  * deletion rule (unchanged): ONLY for an INTERMEDIATE export (in plan/EXPORT_STEPS.txt and not its last entry) the node-local copy is
#    deleted (scratch_guard guard_rm) when ALL hold: hfb_up rc 0 (VERIFY PASS), verify json of this attempt with PASS / private / n_missing 0 /
#    no size or sha mismatch / every *.pt sha-checked, this host's 8 rank files, no *.tmp, and on slinky-0 rep.pt + meta.json + COMPLETE.
#    The FINAL export (last of EXPORT_STEPS; or every export if that file is unreadable) stays node-local.
#  * one line per result appended to logs/BACKUP_LOG.txt; sources are only read, except the guarded deletions. A host whose export is
#    already verified (.done) is skipped, so a relaunch only redoes what is missing. End marker hf_backup/state/export-N/<host>.end
#    (= "<launch id> <rc>") tells the watcher's throttle this host no longer needs upload workers.
# Dry test (uploads nothing, never touches the training job or real exports): HFLX_DRY=1 HFLX_ROOT=/scratch/amartyavarma/hflx-dry-ttconvo*
#   (fake exports/ + farms hfb/ on each host) HFLX_SHARED=/data/home/amartyavarma/ppm/ttconvo-ft/tests/dry-* (logs/, hf_backup/, plan/)
#   HFLX_UP=<hfb_up.py stub> HFLX_HOSTS=a,b HFLX_JOBMAP=a:<job on a>,b:<job on b> [HFLX_GUARD_LIST=<real list + extra lines>],
#   TTC_LAUNCH=dry (per-host overlap steps -c1 --mem=0 in those jobs, throttled) | dry-alloc (same, fixed workers, node-0 COMPLETE filter).
#   Harness: tests/hf_dry.sh.
set -uo pipefail
B=/data/home/amartyavarma/ppm/ttconvo-ft
MODE=${TTC_LAUNCH:-alloc}
EXPORTS=${TTC_EXPORTS:?TTC_EXPORTS (export steps, comma-separated) not set}
LID=${TTC_LAUNCH_ID:-$MODE-${SLURM_JOB_ID:-nojob}-$(date +%s)}
if [ "${HFLX_DRY:-0}" = 1 ]; then
  : "${HFLX_ROOT:?}" "${HFLX_SHARED:?}" "${HFLX_UP:?}" "${HFLX_HOSTS:?}"
  case "$HFLX_ROOT" in /scratch/amartyavarma/hflx-dry-ttconvo*) ;; *) echo "ABORT: dry root $HFLX_ROOT not /scratch/amartyavarma/hflx-dry-ttconvo*"; exit 1;; esac
  case "$HFLX_SHARED" in $B/tests/dry-*) ;; *) echo "ABORT: dry shared dir $HFLX_SHARED not $B/tests/dry-*"; exit 1;; esac
  case "$MODE" in dry|dry-alloc) ;; *) echo "ABORT: dry run with TTC_LAUNCH=$MODE (only dry / dry-alloc)"; exit 1;; esac
  EXR=$HFLX_ROOT/exports; LD=$HFLX_SHARED/logs; HB=$HFLX_SHARED/hf_backup; PLAN=$HFLX_SHARED/plan/EXPORT_STEPS.txt
  FB=$HFLX_ROOT/hfb; UPY=$HFLX_UP; BO=1; R=InternalCan/DRYRUN-no-upload; HOSTS=$HFLX_HOSTS
  [ -n "${HFLX_GUARD_LIST:-}" ] && export SCRATCH_GUARD_LIST=$HFLX_GUARD_LIST
else
  case "$MODE" in overlap|alloc) ;; *) echo "ABORT: TTC_LAUNCH=$MODE (overlap | alloc)"; exit 1;; esac
  EXR=/scratch/amartyavarma/ttconvo-ft/exports; LD=$B/logs; HB=$B/hf_backup; PLAN=$B/plan/EXPORT_STEPS.txt
  FB=/scratch/amartyavarma/hfb; UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py; BO=60; R=InternalCan/ppm-ttconvo-ft
  HOSTS=slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19
  unset SCRATCH_GUARD_LIST                                   # the guard's own default list (PROTECTED_SCRATCH.txt), never an inherited one
fi
case "$MODE" in overlap|dry) THR=1; WK=2;; *) THR=0; WK=8;; esac   # throttled beside training (PPM: --workers 2) | Aloe recipe (8 workers)
NH=$(echo $HOSTS | tr , '\n' | wc -l)
BLOG=$LD/BACKUP_LOG.txt; VO=$HB/verify; SPD=$HB/spec; PY=/data/home/amartyavarma/venvs/duplex/bin/python
NRANK=8; NAME=ttconvo-ft
mkdir -p $LD $VO $SPD $HB/control $HB/state
export B MODE EXPORTS LID EXR LD HB PLAN FB UPY BO R HOSTS NH THR WK BLOG VO SPD PY NRANK NAME
export HFLX_DRY=${HFLX_DRY:-0}

pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
blog() { echo "[$(pt)] $*" | tee -a $L >> $BLOG; }          # one result line -> BACKUP_LOG.txt (+ the host log)

# verify json fresh (written at/after $3) and clean: PASS, private, nothing missing, no size / sha mismatch, >= $2 files sha-checked (every *.pt)
vj_ok() {
  [ -f "$1" ] && [ "$(stat -c %Y "$1")" -ge "$3" ] || return 1
  $PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); n=int(sys.argv[2])
ok = j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch") \
     and int(j.get("sha_checked_vs_upload_cache", 0)) >= n
sys.exit(0 if ok else 1)' "$1" "$2"
}

# final export = last entry of plan/EXPORT_STEPS.txt; unreadable plan -> every export counts as final (kept node-local)
is_intermediate() {
  local s; s=$(tr -s ', \n' '\n\n\n' < "$PLAN" 2>/dev/null | grep -E '^[0-9]+$') || return 1
  [ -n "$s" ] || return 1
  echo "$s" | grep -qx "$1" && [ "$(echo "$s" | tail -1)" != "$1" ]
}

# hf_one <src dir> <repo path / tag> <manifest name> <farm> <require path> <exclude json list>: upload + verify with retry / back-off
# (= the Aloe tool's loop). rc 0 = VERIFY PASS, 1 upload failed, 2 repo create / farm failed, 3 verify failed, 4 source vanished, 5 owner job gone.
hf_one() {
  local W=$1 TAG=$2 MAN=$3 F=$4 REQ=$5 EXC=$6 S UP CH bo=$BO att=0 vfail=0 hrc=0 rc vrc
  S=$SPD/spec-$TAG-$H.json
  mkdir -p "$F"
  printf '{"dirs": [["%s", "%s"]], "exclude": %s}\n' "$W" "$TAG" "$EXC" > $S
  echo "[$(pt)] $H start $W ($(ls -A $W | wc -l) files, $(du -sbL $W | cut -f1) B) -> private $R/$TAG/ (farm $F, manifest $MAN, mode $MODE, workers $WK$( [ $THR = 1 ] && echo ", throttle $CTL"))" >> $L
  UP="$PY $UPY --repo $R --type model --farm $F --spec $S --manifest-name $MAN --tag $TAG --require $REQ --workers $WK --verify-out $VO"
  if [ $THR = 1 ]; then CH="$UP --no-verify --control $CTL --max-fails 3"          # upload loop; waits while the control file says 0
  else CH="$PY $UPY --child --repo $R --type model --farm $F --spec $S --workers $WK"; fi
  while true; do
    att=$((att+1))
    if ! nice -n 10 $UP --verify-only --no-verify >> $L 2>&1; then
      echo "[$(pt)] $H $TAG attempt $att: repo create / farm failed; retry ${bo}s" >> $L; [ $att -ge 8 ] && { hrc=2; break; }
      sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
    nice -n 10 $CH >> $L 2>&1; rc=$?
    if [ $rc = 4 ] || [ $rc = 5 ]; then echo "[$(pt)] $H $TAG attempt $att: upload rc $rc ($( [ $rc = 4 ] && echo 'source vanished' || echo 'owner job gone')): stop" >> $L; hrc=$rc; break; fi
    if [ $rc -ne 0 ]; then
      echo "[$(pt)] $H $TAG attempt $att: upload rc $rc; retry ${bo}s" >> $L; [ $att -ge 8 ] && { hrc=1; break; }
      sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
    nice -n 10 $UP --verify-only >> $L 2>&1; vrc=$?
    [ $vrc -eq 0 ] && break
    vfail=$((vfail+1)); echo "[$(pt)] $H $TAG VERIFY FAIL rc $vrc ($vfail)" >> $L; [ $vfail -ge 4 ] && { hrc=3; break; }; sleep $bo
  done
  return $hrc
}

one_export() {   # one_export <N> -> 0 verified (and handled), 1 not
  local N=$1 T d ER r ok rc=0 npt nrk ntmp nf sz nb t0 VJ why f k miss DN EN LOC
  T=export-$N; L=$LD/hf_up-$T-$H.log; DN=$HB/state/$T/$H.done; EN=$HB/state/$T/$H.end; LOC=$HB/state/$T/$H.local
  mkdir -p $HB/state/$T
  echo "[$(pt)] $H $T launch $LID (job ${SLURM_JOB_ID:-?} step ${SLURM_STEP_ID:-?}, mode $MODE, dry $HFLX_DRY)" >> $L
  if [ -f $DN ]; then echo "[$(pt)] $H $T already verified ($DN): skipped" >> $L; echo "$LID 0 skip" > $EN; return 0; fi
  if [ -L "$EXR" ] || [ ! -d "$EXR" ]; then blog "$H ABORT: $EXR missing or a symlink (expected a real node-local dir); launch $LID"; echo "$LID 1 noroot" > $EN; return 1; fi
  ER=$(readlink -f "$EXR")
  case "$ER" in /scratch/*) ;; *) blog "$H ABORT: $EXR resolves to $ER (not node-local /scratch); launch $LID"; echo "$LID 1 noroot" > $EN; return 1;; esac
  d=$ER/$T
  if [ -L "$d" ] || [ ! -d "$d" ]; then blog "$T NOT uploaded: $H has no node-local $d (missing / not a real dir); launch $LID"; echo "$LID 1 missing" > $EN; return 1; fi
  # this host's files must be complete BEFORE anything is uploaded: rank 8(I-1) .. 8(I-1)+7, nothing else, no *.tmp; slinky-0 (I = 1) also
  # rep.pt + meta.json (format ppm-local-bf16-shards-v1, step N, world 8 x hosts) + COMPLETE
  miss=""
  for k in $(seq $((NRANK*(I-1))) $((NRANK*I-1))); do [ -f $d/rank$k.pt ] || miss="$miss rank$k.pt"; done
  nrk=$(ls $d/rank*.pt 2>/dev/null | wc -l); ntmp=$(find $d -name '*.tmp' | wc -l)
  if [ $I = 1 ]; then
    for f in rep.pt meta.json COMPLETE; do [ -f $d/$f ] || miss="$miss $f"; done
    [ -f $d/meta.json ] && ! $PY -I -c 'import json,sys; m=json.load(open(sys.argv[1]))
sys.exit(0 if m.get("format") == "ppm-local-bf16-shards-v1" and int(m.get("step", -1)) == int(sys.argv[2]) and int(m.get("world", -1)) == int(sys.argv[3]) else 1)' \
      $d/meta.json $N $((NRANK*NH)) && miss="$miss meta.json(format/step/world)"
  fi
  if [ -n "$miss" ] || [ $nrk != $NRANK ] || [ $ntmp != 0 ]; then
    blog "$T NOT uploaded: $H node-local copy incomplete (missing:${miss:- none}; $nrk rank*.pt, want $NRANK; $ntmp *.tmp); launch $LID"
    echo "$LID 1 incomplete" > $EN; return 1; fi
  npt=$(ls $d/*.pt 2>/dev/null | wc -l); nf=$(find $d -type f | wc -l); sz=$(du -sh $d | cut -f1)
  nb=$(find $d -type f -printf '%s\n' | awk '{s += $1} END {printf "%.0f", s}')        # exact file bytes (= the HF listing's); %.0f, NOT %d: mawk clamps %d at 2^31-1 (10-21: every .done said 2147483647 -> false CHECK FAILED)
  t0=$(date +%s)
  hf_one "$d" "$T" "MANIFEST.$T.$H.tsv" "$FB/$NAME-$T-$H" "$d" '[]'; r=$?
  VJ=$VO/${R//\//__}__${T}__${H}.json
  ok=0; [ $r = 0 ] && vj_ok $VJ $npt $t0 && ok=1
  [ $r = 0 ] && [ $ok = 0 ] && r=3
  blog "$T HF upload rc $r (0 = VERIFY PASS) $H node-local $nf files $sz -> private $R/$T/ launch $LID"
  if [ $ok = 1 ]; then
    $PY -I -c 'import json,sys; v=json.load(open(sys.argv[1])); a=sys.argv
json.dump({"host": a[3], "export": a[4], "repo": v["repo"], "path": a[4] + "/", "private": v["private"], "files": int(a[5]), "bytes": int(a[6]),
           "verify_json": a[1], "verify_time": v.get("time"), "sha_checked": v.get("sha_checked_vs_upload_cache"),
           "sample_download_sha256": v.get("sample_download_sha256"), "launch": a[7], "job": a[8], "time": a[9]}, open(a[2], "w"), indent=1)' \
      $VJ $DN.tmp $H $T $nf $nb $LID ${SLURM_JOB_ID:-?} "$(pt)" && mv $DN.tmp $DN || { ok=0; rc=1; blog "$H $T verified but the done marker could not be written"; }
  else rc=1; fi
  if is_intermediate $N; then
    why=""
    [ $ok = 1 ] || why="upload / verify rc $r"
    [ $nrk = $NRANK ] && [ $(ls $d/rank*.pt 2>/dev/null | wc -l) = $NRANK ] || why="$why; rank*.pt != $NRANK"
    [ $(find $d -name '*.tmp' | wc -l) = 0 ] || why="$why; *.tmp present"
    if [ $I = 1 ]; then for f in rep.pt meta.json COMPLETE; do [ -f $d/$f ] || why="$why; no $f (rank 0's node)"; done; fi
    if [ -n "$why" ]; then blog "$H $T node-local copy KEPT: ${why#; }"; echo "kept: ${why#; }" > $LOC
    elif guard_rm "$d" 2>> $L; then blog "$H $T node-local copy DELETED ($sz, $d) after HF VERIFY PASS"; echo "deleted $sz $(pt)" > $LOC
    else blog "$H $T node-local copy KEPT: scratch_guard refused $d"; echo "kept: scratch_guard refused" > $LOC; rc=1; fi
  else
    blog "$H $T node-local copy kept (final export)"; echo "kept: final export" > $LOC
  fi
  echo "$LID $r" > $EN
  return $rc
}

node_main() {
  set -uo pipefail
  H=$(hostname -s); L=$LD/hf_up-node-$H.log
  local N rc=0
  I=$(echo "$HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
  [ -n "$I" ] || { echo "ABORT: $H not in $HOSTS"; exit 1; }
  source /data/home/amartyavarma/ppm/scratch_guard.sh
  export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$FB/xet-cache HFB_OWNER=${SLURM_JOB_ID:-}
  if [ $THR = 1 ]; then     # beside training: = setup-logs/hfb/hfb_train_node.sh's gentle xet settings
    export HF_XET_FIXED_UPLOAD_CONCURRENCY=4 HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false HF_XET_MAX_CONCURRENT_FILE_INGESTION=2
    CTL=$HB/control/$H.workers; [ -f $CTL ] || echo 0 > $CTL
  else export HF_XET_HIGH_PERFORMANCE=1; CTL=; fi
  mkdir -p $FB/xet-cache
  for N in $(echo $EXPORTS | tr , ' '); do one_export $N || rc=1; done
  echo "[$(pt)] $H launch $LID exports $EXPORTS done rc $rc; $(df -BG /scratch | tail -1)" >> $LD/hf_up-node-$H.log
  exit $rc
}
export -f pt blog vj_ok is_intermediate hf_one one_export node_main

unset_slurm() { local v; for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*'); do unset $v; done; }
NICE="nice -n 19 ionice -c3"
echo "== hf_local_exports $(TZ=America/Los_Angeles date '+%F %T PT') mode $MODE exports $EXPORTS launch $LID hosts $HOSTS repo $R dry $HFLX_DRY"
case "$MODE" in
overlap)
  : "${TTC_JOB:?TTC_JOB (training job id) not set}"
  unset_slurm
  srun --jobid=$TTC_JOB --overlap -N$NH -n$NH -w $HOSTS --ntasks-per-node=1 -c 4 --mem=40G --gres=none --cpu-bind=none -J ttconvo-hfb-up \
       --kill-on-bad-exit=0 $NICE bash -c node_main
  rc=$? ;;
alloc)
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "ABORT: alloc mode must run as the sbatch job"; exit 1; }
  AH=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
  [ "$AH" = "$HOSTS" ] || { echo "ABORT: allocation $AH != $HOSTS"; exit 1; }
  H0=${HOSTS%%,*}; KEEP=""
  for N in $(echo $EXPORTS | tr , ' '); do      # COMPLETE is on rank 0's node only: check it there (a step of this job); an export whose
    # rank-0 part is already verified (its intermediate copy may be deleted there) counts as complete
    if [ -f $HB/state/export-$N/$H0.done ] || srun -N1 -n1 -w $H0 -c1 --mem=1G --cpu-bind=none --overlap $NICE test -f $EXR/export-$N/COMPLETE; then KEEP="$KEEP,$N"
    else echo "export-$N: no COMPLETE on $H0: not uploaded"
      echo "[$(pt)] export-$N NOT uploaded by fallback job $SLURM_JOB_ID: no $EXR/export-$N/COMPLETE on $H0" >> $BLOG; fi; done
  EXPORTS=${KEEP#,}; export EXPORTS
  [ -n "$EXPORTS" ] || { echo "nothing COMPLETE to upload"; exit 1; }
  srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 4 --mem=40G --cpu-bind=none --kill-on-bad-exit=0 $NICE bash -c node_main
  rc=$? ;;
dry|dry-alloc)
  : "${HFLX_JOBMAP:?}"
  unset_slurm
  jid() { echo "$HFLX_JOBMAP" | tr , '\n' | awk -F: -v h=$1 '$1==h {print $2}'; }
  if [ "$MODE" = dry-alloc ]; then
    H0=${HOSTS%%,*}; KEEP=""
    for N in $(echo $EXPORTS | tr , ' '); do
      if [ -f $HB/state/export-$N/$H0.done ] || srun --jobid=$(jid $H0) --overlap -w $H0 -N1 -n1 -c1 --mem=0 --gres=none --cpu-bind=none -J ttconvo-hfb-dry $NICE test -f $EXR/export-$N/COMPLETE; then KEEP="$KEEP,$N"
      else echo "[$(pt)] export-$N NOT uploaded by fallback (dry): no COMPLETE on $H0" >> $BLOG; fi; done
    EXPORTS=${KEEP#,}; export EXPORTS
    [ -n "$EXPORTS" ] || { echo "nothing COMPLETE to upload"; exit 1; }
  fi
  pids=""
  for h in $(echo $HOSTS | tr , ' '); do
    srun --jobid=$(jid $h) --overlap -w $h -N1 -n1 -c1 --mem=0 --gres=none --cpu-bind=none -J ttconvo-hfb-dry $NICE bash -c node_main & pids="$pids $!"
  done
  rc=0; for p in $pids; do wait $p || rc=1; done ;;
esac
echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; tail -n 80 $BLOG 2>/dev/null | grep -F "launch $LID"
exit $rc

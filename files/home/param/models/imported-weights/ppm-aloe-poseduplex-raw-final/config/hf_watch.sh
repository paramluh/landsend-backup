#!/bin/bash
# aloe-ft-32-samsep HF watcher: ONE task per training host (run.sbatch starts it as a nice'd --overlap --cpu-bind=none step; can also be run
# again by hand as an overlap step of the job, or in a CPU-only job on the host, to finish uploads). PRIVATE HF ONLY, NEVER GCS.
# = aloe-zoom-cotrain/hf_local_exports.sbatch's hf_one / vj_ok / guarded deletion (hfb_up.py: symlink farm, upload_large_folder, retry /
#   back-off, verify = repo private + every file present with its size + LFS sha256 == upload-time sha256 + 3 files streamed back), as a LOOP:
#   every 60 s, each node-local export dir with NODE_COMPLETE (the trainer writes it once every rank's file is on disk) not yet uploaded from
#   this host -> HF: intermediates (export-<N>, N in QQA_EXPORT_STEPS) -> $HF_REPO_INTER/export-<N>/, the final export-<QQA_STEPS> and any
#   export-stop<N> -> $HF_REPO_FINAL/<name>/. Hosts upload disjoint files (each its own rank<R>.pt; rep.pt / meta.json / COMPLETE from rank 0's host).
#   HFLX_DELETE_INTER=1 (main approved 10-24): an INTERMEDIATE node-local copy is deleted (scratch_guard guard_rm) only when ALL hold: hfb_up rc 0
#   (VERIFY PASS), a fresh clean verify json (PASS / private / n_missing 0 / no size or sha mismatch / every *.pt sha-checked), the rank files
#   NODE_COMPLETE lists are all there, no *.tmp, and on rank 0's host rep.pt + meta.json + COMPLETE. Each deletion -> $HF_DELETE_LOG + BACKUP_LOG.
#   The final / stop exports are never deleted. After the trainer exits (control/TRAIN_EXITED-<job>) and nothing is pending, rank 0's host also
#   uploads the consolidated /data export ($QQA_EXPORT, if COMPLETE) as $HF_REPO_FINAL/export-<last>-dcp/, then the watcher exits.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/poseduplex-raw   # [raw] poseduplex-raw copy; was /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex # (10-25): logs / done markers under poseduplex/logs; else verbatim
set -a; source ${QQA_ENV:-$Q/run_poseduplex.env}; set +a
H=$(hostname -s); J=${SLURM_JOB_ID:-manual}; TJ=${HFW_TRAIN_JOB:-$J}
EXR=$ALOE_LOCAL_EXPORT_ROOT; WX=$QQA_EXPORT; LD=$Q/logs; FB=/scratch/amartyavarma/hfb; UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py
BO=60; WK=8; NAME=${HFW_NAME:-aloe-samsep-16861}; LAST=$QQA_STEPS; BLOG=$LD/BACKUP_LOG.txt; VO=$LD/hf_verify; SPD=$LD/hf_spec
PY=/data/home/amartyavarma/venvs/duplex/bin/python; L=$LD/hf_watch-$J-$H.log
DONE=$LD/hf_done/j$TJ
INTER=" $(echo $QQA_EXPORT_STEPS | tr , ' ' | sed 's/[0-9]\+/export-&/g') "
R0H=$(echo $PPM_HOSTS | cut -d, -f1)                         # training rank 0's host (rep.pt / meta.json / COMPLETE live there)
mkdir -p $VO $SPD $DONE $FB/xet-cache
unset SCRATCH_GUARD_LIST
source /data/home/amartyavarma/ppm/scratch_guard.sh
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$FB/xet-cache HFB_OWNER=$J

pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
blog() { echo "[$(pt)] $*" | tee -a $L >> $BLOG; }

vj_ok() {
  [ -f "$1" ] && [ "$(stat -c %Y "$1")" -ge "$3" ] || return 1
  $PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); n=int(sys.argv[2])
ok = j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch") \
     and int(j.get("sha_checked_vs_upload_cache", 0)) >= n
sys.exit(0 if ok else 1)' "$1" "$2"
}

# hf_one <src dir> <repo> <repo path / tag> <manifest name> <farm> <require path> <exclude json list> (= the reference's, repo as an argument)
hf_one() {
  local W=$1 R=$2 TAG=$3 MAN=$4 F=$5 REQ=$6 EXC=$7 S UP bo=$BO att=0 vfail=0 hrc=0 rc vrc
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

one_export() {                                                  # $1 = export dir
  local d=$1 T R r t0 VJ why nrk npt ntmp nf sz want f
  T=$(basename $d)
  case "$INTER" in *" $T "*) R=$HF_REPO_INTER;; *) R=$HF_REPO_FINAL;; esac
  nf=$(ls -A $d | wc -l); sz=$(du -sh $d | cut -f1); npt=$(ls $d/*.pt 2>/dev/null | wc -l); t0=$(date +%s)
  hf_one "$d" "$R" "$T" "MANIFEST.$T.$H.tsv" "$FB/$NAME-$T-$H" "$d/NODE_COMPLETE" '["NODE_COMPLETE"]'; r=$?   # 10-24 w64: per-host NODE_COMPLETE collided on HF (verify sample mismatch): not uploaded
  VJ=$VO/${R//\//__}__${T}__${H}.json
  blog "$T HF upload rc $r (0 = VERIFY PASS) $H node-local $nf files $sz -> private $R/$T/ job $J"
  [ $r = 0 ] && touch $DONE/$T.$H
  case "$INTER" in
    *" $T "*)
      [ "${HFLX_DELETE_INTER:-0}" = 1 ] || { blog "$H $T node-local copy kept (deletion off: HFLX_DELETE_INTER=0)"; return $r; }
      why=""
      [ $r = 0 ] || why="upload / verify rc $r"
      [ $r = 0 ] && { vj_ok $VJ $npt $t0 || why="verify json $VJ not fresh / clean"; }
      want=$($PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); print(" ".join(f"rank{r}.pt" for r in j["ranks"]))' $d/NODE_COMPLETE)
      for f in $want; do [ -f $d/$f ] || why="$why; no $f"; done
      nrk=$(ls $d/rank*.pt 2>/dev/null | wc -l); [ $nrk = $(echo $want | wc -w) ] || why="$why; $nrk rank*.pt != $(echo $want | wc -w)"
      ntmp=$(find $d -name '*.tmp' | wc -l); [ $ntmp = 0 ] || why="$why; $ntmp *.tmp"
      if [ $H = $R0H ]; then for f in rep.pt meta.json COMPLETE; do [ -f $d/$f ] || why="$why; no $f (rank 0's host)"; done; fi
      if [ -n "$why" ]; then blog "$H $T node-local copy KEPT: ${why#; }"
      elif guard_rm "$d" 2>> $L; then
        blog "$H $T node-local copy DELETED ($sz, $d) after HF VERIFY PASS ($R/$T/)"
        echo "[$(pt)] aloe-ft-32-samsep job $J: deleted $H:$d ($sz, node-local INTERMEDIATE export, ppm-local-bf16-shards-v1) after HF VERIFY PASS -> private $R/$T/ (verify $VJ)" >> $HF_DELETE_LOG
      else blog "$H $T node-local copy KEPT: scratch_guard refused $d"; fi;;
    *) blog "$H $T node-local copy kept (final / stop export)";;
  esac
  return $r
}

echo "[$(pt)] $H watcher job $J (train job $TJ): $EXR -> private $HF_REPO_INTER (intermediates$INTER) / $HF_REPO_FINAL (final export-$LAST, export-stop*); delete intermediates after VERIFY: ${HFLX_DELETE_INTER:-0}" >> $L
case "$(readlink -f $EXR)" in /scratch/*) ;; *) blog "$H ABORT: $EXR not node-local /scratch; job $J"; exit 1;; esac
rc=0
while true; do
  pend=0
  for d in $(ls -d $EXR/export-* 2>/dev/null | sort -V); do
    T=$(basename $d)
    [ -f $DONE/$T.$H ] && continue
    [ -L "$d" ] && continue
    [ -f $d/NODE_COMPLETE ] || { pend=$((pend + 1)); continue; }
    one_export $d || rc=1
  done
  if [ -f $QQA_CONTROL_DIR/TRAIN_EXITED-$TJ ] && [ $pend = 0 ]; then break; fi
  if [ -f $QQA_CONTROL_DIR/TRAIN_EXITED-$TJ ]; then                       # trainer gone: a dir still without NODE_COMPLETE never completes
    for d in $(ls -d $EXR/export-* 2>/dev/null); do [ -f $d/NODE_COMPLETE ] || [ -f $DONE/$(basename $d).$H ] || blog "$H $(basename $d) INCOMPLETE (no NODE_COMPLETE; trainer exited): not uploaded; job $J"; done
    break
  fi
  sleep 60
done
if [ $H = $R0H ]; then
  if [ -f $WX/COMPLETE ] && [ ! -f $DONE/export-$LAST-dcp ]; then
    npt=$(ls $WX/*.distcp 2>/dev/null | wc -l); t0=$(date +%s)
    hf_one "$WX" "$HF_REPO_FINAL" export-$LAST-dcp MANIFEST.export-$LAST-dcp.tsv "$FB/$NAME-export-$LAST-dcp-$H" "$WX/COMPLETE" '["HF_BACKUP.json"]'; r=$?
    VJ=$VO/${HF_REPO_FINAL//\//__}__export-$LAST-dcp__${H}.json
    if [ $r = 0 ] && vj_ok $VJ $npt $t0; then
      printf '{"repo": "%s", "path": "%s", "private": true, "verify": "%s", "time": "%s", "job": "%s"}\n' "$HF_REPO_FINAL" "export-$LAST-dcp/" "$VJ" "$(pt)" "$J" > $WX/HF_BACKUP.json
      touch $DONE/export-$LAST-dcp
    else rc=1; [ $r = 0 ] && r=3; fi
    blog "export-$LAST-dcp HF upload rc $r (0 = VERIFY PASS) $H $WX -> private $HF_REPO_FINAL/export-$LAST-dcp/ job $J"
  elif [ ! -f $WX/COMPLETE ]; then blog "export-$LAST-dcp not uploaded: $WX/COMPLETE missing (/data floor refusal or run stopped early); node-local export + HF = final copy; job $J"; fi
fi
echo "[$(pt)] $H watcher done rc $rc; $(df -BG /scratch | tail -1)" >> $L
exit $rc

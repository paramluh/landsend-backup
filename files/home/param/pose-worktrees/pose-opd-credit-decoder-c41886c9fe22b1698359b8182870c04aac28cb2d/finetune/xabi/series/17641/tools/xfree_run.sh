#!/bin/bash
# xfree_run.sh RUN REPO STEPS FINAL HOSTS NTASKS [TRAINLOG] [TRAINJOB]   (10-22, aloe-v2pure-17352 / -17641 scaling FTs)
# = setup-logs/hfb/scratchfree/x1022/xfree1_host.sh's rule (upload + VERIFY PASS -> delete) for one host of an aloe-v2pure FT run, adapted to
# (a) uneven per-host rank counts (HOSTS / NTASKS aligned lists; rank 0's host = the first of HOSTS: + rep.pt / meta.json / COMPLETE) and
# (b) a RUNNING trainer: with TRAINLOG, each intermediate export-<s> is taken only after the trainer logged "[export-local] .../export-<s> on every
#     node" (all ranks written); if TRAINJOB ends first, the exports present are still processed. No step-time throttle controller (FT runs):
#     2 upload workers, nice 19 / ionice idle. The FINAL export is never touched.
# Per INTERMEDIATE export (s in STEPS, s != FINAL): upload this host's shard files to the PRIVATE HF model repo REPO path export-<s>/ (hfb_up.py:
# symlink farm, upload_large_folder, verify = private, every file with its size, LFS sha256 == upload-time local sha256, 3 files streamed back +
# re-hashed); delete the node-local dir ONLY when rc 0 + fresh clean verify json + rank*.pt count == this host's ranks + no *.tmp + rank 0's
# extras + no hard links + not in use; via scratch_guard guard_rm. One line per result -> RUN's logs/BACKUP_LOG.txt; deletions also ->
# setup-logs/hfb/DELETED-1021-hfb-hold.log. Runs as one srun task per host (xfree_run.sbatch).
set -uo pipefail
RUN=$1; R=$2; STEPS=$3; FINAL=$4; HOSTS=$5; NT=$6; TLOG=${7:-}; TJOB=${8:-}
HD=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python
H=$(hostname -s); Q=/data/home/amartyavarma/ppm/$RUN; BLOG=$Q/logs/BACKUP_LOG.txt; DLOG=$HD/DELETED-1021-hfb-hold.log
L=$Q/logs/xfree-$H.log; VO=$Q/logs/hf_verify; SPD=$Q/logs/hf_spec; EXR=/scratch/amartyavarma/$RUN/exports
mkdir -p $VO $SPD
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
blog() { echo "[$(pt)] $*" | tee -a $L >> $BLOG; }
free_g() { df -BG --output=avail /scratch | tail -1 | tr -dc 0-9; }
I=$(echo "$HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
[ -n "$I" ] || { echo "[$(pt)] ABORT: $H not in $HOSTS" >> $L; exit 1; }
NR=$(echo "$NT" | tr , '\n' | sed -n ${I}p)
unset SCRATCH_GUARD_LIST; source /data/home/amartyavarma/ppm/scratch_guard.sh
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=/scratch/amartyavarma/hfb/xet-cache HFB_OWNER=${SLURM_JOB_ID:-}
export HF_XET_FIXED_UPLOAD_CONCURRENCY=4 HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false HF_XET_MAX_CONCURRENT_FILE_INGESTION=2 CUDA_VISIBLE_DEVICES=
echo "[$(pt)] $H host $I of $HOSTS ($NR ranks): $RUN intermediate exports [$STEPS] -> private $R; log ${TLOG:-none} job ${TJOB:-none}; /scratch free $(free_g)G" >> $L
alive() { [ -n "$TJOB" ] && squeue -h -j "$TJOB" 2>/dev/null | grep -q .; }
rc=0
for s in $STEPS; do
  T=export-$s; d=$EXR/$T
  [ "$s" = "$FINAL" ] && { blog "$H $T is the FINAL export: never deleted here"; continue; }
  if [ -n "$TLOG" ]; then   # wait for the trainer's all-ranks-done line
    while ! grep -qF "[export-local] $d on every node" "$TLOG" 2>/dev/null; do alive || break; sleep 20; done
    grep -qF "[export-local] $d on every node" "$TLOG" 2>/dev/null || { blog "$H $T: trainer job $TJOB ended before logging it: skipped"; rc=1; continue; }
  fi
  if [ -L "$d" ] || [ ! -d "$d" ]; then blog "$H $T not a real dir (already removed?): skipped"; continue; fi
  nrk=$(ls $d/rank*.pt 2>/dev/null | wc -l); npt=$(ls $d/*.pt 2>/dev/null | wc -l); ntmp=$(find $d -name '*.tmp' | wc -l)
  nf=$(ls -A $d | wc -l); sz=$(du -sh $d | cut -f1); t0=$(date +%s)
  S=$SPD/spec-$RUN-$T-$H.json; printf '{"dirs": [["%s", "%s"]]}\n' "$d" "$T" > $S
  F=/scratch/amartyavarma/hfb/xfree-$RUN-$T-$H
  echo "[$(pt)] $H $T start: $nf files $sz -> $R/$T/" >> $L
  nice -n 19 ionice -c3 $PY $HD/hfb_up.py --repo $R --type model --farm $F --spec $S --manifest-name MANIFEST.$T.$H.tsv --tag $T \
      --require $d --workers 2 --verify-out $VO --max-fails 5 >> $L 2>&1; r=$?
  VJ=$VO/${R//\//__}__${T}__${H}.json
  blog "$T HF upload rc $r (0 = VERIFY PASS) $H node-local $nf files $sz -> private $R/$T/ (xfree_run, job ${SLURM_JOB_ID:-})"
  why=""
  [ $r = 0 ] || why="upload / verify rc $r"
  if [ $r = 0 ]; then
    $PY -I -c 'import json,os,sys; p,n,t0=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]); j=json.load(open(p))
ok = os.path.getmtime(p) >= t0 and j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") \
     and not j.get("sha_mismatch") and int(j.get("sha_checked_vs_upload_cache", 0)) >= n and not j.get("sample_deferred")
sys.exit(0 if ok else 1)' "$VJ" "$npt" "$t0" || why="verify json $VJ not fresh / clean"
  fi
  [ "$nrk" = "$NR" ] || why="$why; $nrk rank*.pt != $NR"
  [ $ntmp = 0 ] || why="$why; $ntmp *.tmp"
  if [ $I = 1 ]; then for f in rep.pt meta.json COMPLETE; do [ -f $d/$f ] || why="$why; no $f (rank 0's host)"; done; fi
  [ -z "$(find $d -type f -links +1 | head -1)" ] || why="$why; hard-linked files"
  rd=$(readlink -f $d); inuse=$(for p in /proc/[0-9]*; do for x in $p/cwd $p/fd/*; do t=$(readlink "$x" 2>/dev/null) && case "$t" in "$rd"|"$rd"/*) echo "${p#/proc/}";; esac; done; done 2>/dev/null | head -2)
  [ -z "$inuse" ] || why="$why; in use by pid $inuse"
  if [ -n "$why" ]; then blog "$H $T node-local copy KEPT: ${why#; }"; rc=1; continue; fi
  b=$(du -sb $d | cut -f1); f0=$(free_g)
  if guard_rm "$d" 2>> $L; then
    guard_rm "$F" 2>> $L
    blog "$H $T node-local copy DELETED ($sz, $d) after HF VERIFY PASS; /scratch free $(free_g)G"
    line="$(TZ=America/Los_Angeles date '+%F %T PT') DEL $H $d ($b B, $nf files) rc 0 free ${f0}G -> $(free_g)G :: $RUN (10-22 v2pure scaling FT) intermediate node-local export shard step $s of this host ($NR ranks): uploaded to private HF $R/$T/ and VERIFY PASS (private, every file with its size, LFS sha256 == upload-time sha256 of all $npt *.pt, 3 files streamed back + re-hashed; $VJ); no open fds, no hard links; final export-$FINAL kept node-local"
    ( flock -w 60 9; echo "$line" >> $DLOG ) 9>>$DLOG.lock
  else blog "$H $T node-local copy KEPT: scratch_guard refused $d"; rc=1; fi
done
echo "[$(pt)] $H done rc $rc; /scratch free $(free_g)G" >> $L
exit $rc

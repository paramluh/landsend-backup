#!/bin/bash
#SBATCH -p all -N1 -n1 -c 2 --mem=8G --gres=none -t 12:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025/%x-%j.out
# ONE host's full-state save (safety_ppm.save_local layout) -> PRIVATE HF model repo, folder dcp-<STEP>/<LABEL>/ (HF ONLY, never GCS).
#   1. digest check: verify_dcp_host.py (COMPLETE == STEP, exact file set for this host's ranks, train_state step + hosts, every rank<R>.pt and
#      rep.pt re-digested with the saving run's own safety_ppm.digest_state == digest-<R>.txt / digest-rep.txt; records each file's sha256)
#   2. hfb_up.py (symlink farm, upload_large_folder; verify = private + every file + size + HF LFS sha256 vs upload-time sha256)
#   3. hfb_sha_all.py (fresh sha256 of EVERY file vs HF)   4. every HF sha256 == the sha256 recorded in step 1 (the digest-verified bytes)
# Light for runs inside MoT: nice 19 + ionice idle, 2 tokio / rayon threads, 1 upload worker, xet fixed upload concurrency 2 (anon RSS ~0.45 GB),
# xet cache OUTSIDE the farm. Source read-only, NEVER deleted.
# v4 (10-25 06:20 PT) = v3 with a per-host pause file PAUSE-hfstate-<host> (up to 3 hosts at once). v3 (10-25 06:00 PT) = v2 + COOPERATIVE PAUSE: the driver touches $O/PAUSE-hfstate when MoT slows > 3 %; this script waits (idle) at every pause
# point: before / between the digest check's files, before EACH file's upload (files go up one at a time, ~2-3 min each), before the final
# verifies. Nothing is ever killed or signalled. Early self-exit: STOP file $O/STOP-hfstate -> rc 9 at a pause point; OOM (rc 137) twice -> rc 10;
# a file failing 3 times -> rc 1. Success marker $O/ok-<repo>-dcp-<step>-<host> (srun's rc is != 0 after any contained OOM event).
#   hf_state_host_v3.sbatch <SRC dir> <STEP> <host index> <REPO> <LABEL host> [allowed extra files, comma list]
SRC=$1; STEP=$2; IDX=$3; R=$4; LABEL=$5; EXTRA=${6:-}
H=$(hostname -s); K=dcp-$STEP; RS=${R#*/}
D=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python; PYT=/data/home/amartyavarma/venvs/qwen-s2/bin/python
T=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools; O=/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025
F=/scratch/amartyavarma/hfb/st-$RS-$K-$LABEL; XC=/scratch/amartyavarma/hfb/xetc-$RS-$K-$LABEL; SP=$O/spec-$RS-$K-$LABEL.json; SPF=$O/spec1-$RS-$K-$LABEL.json
TAG=$K-$LABEL; VJ1=$O/verify-$RS-$K-$LABEL.json; PZ=$O/PAUSE-hfstate-$LABEL; PZG=$O/PAUSE-hfstate; STOP=$O/STOP-hfstate   # v4: per-host pause file (+ the global one)
LOW="nice -n 19 ionice -c3"
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
step() { echo "== [$(pt)] $LABEL (on $H) $*"; }
pause_point() {                                                     # $1 = where; returns 1 on STOP
  local w=0
  while [ -f $PZ ] || [ -f $PZG ]; do [ $w = 0 ] && step "PAUSED at $1 (driver: MoT slower than 1.03 x idle baseline)"; w=1; sleep 20; [ -f $STOP ] && break; done
  [ $w = 1 ] && step "resumed at $1"
  [ -f $STOP ] && { step "STOP file at $1: exiting (rc 9)"; return 1; }
  return 0
}
T0=$(date +%s)
case "$H" in slinky-9|slinky-10) step "ABORT: never on $H"; exit 1;; esac
[ -n "$LABEL" ] && [ -n "$R" ] && [ -f $SRC/COMPLETE ] && [ "$(cat $SRC/COMPLETE)" = "$STEP" ] || { step "ABORT: $SRC/COMPLETE missing or != $STEP"; exit 2; }
FREE=$(df -B1 --output=avail /scratch | tail -1)
[ "$FREE" -ge 2000000000 ] || { step "ABORT: /scratch free $FREE < 2 GB (farm = symlinks + upload metadata, xet cache without chunk cache: ~100 MB)"; exit 5; }
step "job ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}: $SRC ($(ls -A $SRC | wc -l) files, $(du -sb $SRC | cut -f1) B) -> private $R/$K/$LABEL/; /scratch free $((FREE / 1000000000)) GB"
# ---- 1. digest check (verify_dcp_host.py waits at VERIFY_PAUSE_FILE before each file)
pause_point "digest check" || exit 9
export VERIFY_SAFETY_PY=/data/home/amartyavarma/ppm/ppmvox/code/safety_ppm.py VERIFY_HOST_LABEL=$LABEL VERIFY_ALLOW_EXTRA=$EXTRA VERIFY_PAUSE_FILE=$PZ,$PZG \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [ -f $VJ1 ] && grep -q '"PASS": true' $VJ1; then step "digest check already PASS ($VJ1), not repeated"
else $LOW $PYT -I $T/verify_dcp_host.py $SRC $STEP $IDX 8 $VJ1 1 || { step "ABORT: digest / layout check FAILED ($VJ1); nothing uploaded"; exit 3; }; fi
# ---- 2. upload, one file at a time (pause point before each)
source /data/home/amartyavarma/ppm/scratch_guard.sh
mkdir -p $F $XC
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$XC HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HFB_OWNER=${SLURM_JOB_ID:-}
export HF_XET_FIXED_UPLOAD_CONCURRENCY=${HFS_XET_UPC:-2} HF_XET_MAX_CONCURRENT_FILE_INGESTION=1 HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false
export TOKIO_WORKER_THREADS=2 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1   # --cpu-bind=none => 64 CPUs in the affinity => 64 xet workers = OOM at 2G
NWK=${HFS_WORKERS:-1}; noom=0; hrc=0
COMMON="--repo $R --type model --farm $F --manifest-name MANIFEST.$K.$LABEL.tsv --tag $TAG --require $SRC/COMPLETE --workers $NWK --verify-out $O --defer-sample"
for f in $(ls -A $SRC | sort); do
  n=0
  while true; do
    pause_point "upload of $f" || { hrc=9; break 2; }
    n=$((n+1))
    echo "{\"files\": [[\"$SRC/$f\", \"$K/$LABEL/$f\"]]}" > $SPF
    $LOW $PY $D/hfb_up.py $COMMON --spec $SPF --verify-only --no-verify > /dev/null 2>&1 \
      && { $LOW $PY $D/hfb_up.py --child --repo $R --type model --farm $F --spec $SPF --workers $NWK > $F.child.log 2>&1; rc=$?; } || rc=99
    [ $rc = 0 ] && { step "uploaded $f ($(stat -c %s $SRC/$f) B) after $(( $(date +%s) - T0 )) s"; break; }
    [ $rc = 137 ] && noom=$((noom+1))
    step "upload of $f attempt $n rc $rc (99 = repo / farm step failed); $(tail -2 $F.child.log 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
    [ $noom -ge 2 ] && { step "upload killed by OOM (rc 137) $noom times: giving up (rc 10)"; hrc=10; break 2; }
    [ $n -ge 3 ] && { step "$f failed 3 times: giving up (rc 1)"; hrc=1; break 2; }
    sleep $((60 * n))
  done
done
# ---- full manifest + 3-4. verifies
if [ $hrc -eq 0 ]; then
  pause_point "final verify" || hrc=9
fi
if [ $hrc -eq 0 ]; then
  echo "{\"dirs\": [[\"$SRC\", \"$K/$LABEL\"]]}" > $SP
  rm -f $F/_backup/MANIFEST.$K.$LABEL.tsv.new
  $LOW $PY $D/hfb_up.py $COMMON --spec $SP --verify-only --no-verify > /dev/null 2>&1 \
    && $LOW $PY $D/hfb_up.py --child --repo $R --type model --farm $F --spec $SP --workers $NWK > $F.child.log 2>&1 || { step "full-manifest upload FAILED"; hrc=1; }
fi
if [ $hrc -eq 0 ]; then
  $LOW $PY $D/hfb_up.py $COMMON --spec $SP --verify-only || hrc=3
  [ $hrc -eq 0 ] && { $LOW $PY $D/hfb_sha_all.py --repo $R --type model --farm $F --tag $TAG --threads 2 --verify-out $O || hrc=3; }
fi
if [ $hrc -eq 0 ]; then
  $LOW $PY -I - "$O/${R//\//__}__$TAG-sha256all__$H.json" "$VJ1" "$K/$LABEL" <<'P'; hrc=$?
import json, sys
s, v, pre = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
want = {r["file"]: r["sha256"] for r in v["files"]}
want.update(v["small_files_sha256"])
pf = s["per_file"]
bad = [f for f, h in want.items() if pf.get(f"{pre}/{f}", {}).get("sha256_hf") != h]
extra = sorted(p for p in pf if not p.startswith("_backup/") and p[len(pre) + 1:] not in want)
print(f"cross-check vs the digest check: {len(want) - len(bad)}/{len(want)} files' HF sha256 == the digest-verified sha256; private {s['private']}; bad {bad}; unexpected {extra}")
sys.exit(0 if not bad and not extra and s["PASS"] and s["private"] else 3)
P
fi
[ $hrc -eq 0 ] && guard_rm "$F" "$XC" "$F.child.log"   # own symlink farm + caches + child log only (never the save)
[ $hrc -eq 0 ] && echo "ok $(pt) job ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} on $H (v3)" > $O/ok-$RS-$K-$LABEL
step "$K/$LABEL HF backup rc $hrc (0 = digest check + hfb_up VERIFY PASS + sha256all PASS + cross-check, private $R) after $(( $(date +%s) - T0 )) s"
exit $hrc

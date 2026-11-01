#!/bin/bash
#SBATCH -p all -N1 -n1 -c 2 --mem=8G --gres=none -t 12:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025/%x-%j.out
# ONE host's full-state save (safety_ppm.save_local layout) -> PRIVATE HF model repo, folder dcp-<STEP>/<LABEL>/ (HF ONLY, never GCS).
# = hf_state498_host.sbatch (verified 10-25 on the pose-duplex dcp-498), generalised + LIGHT for runs beside MoT (-c 2, 8G, nice 19, ionice idle,
# 2 upload workers):
#   1. digest check: verify_dcp_host.py (COMPLETE == STEP, exact file set for this host's ranks, train_state step + hosts, every rank<R>.pt and
#      rep.pt re-digested with the saving run's own safety_ppm.digest_state == digest-<R>.txt / digest-rep.txt; records each file's sha256)
#   2. hfb_up.py (symlink farm, upload_large_folder, retries; verify = private + every file + size + HF LFS sha256 vs upload-time sha256)
#   3. hfb_sha_all.py (fresh sha256 of EVERY file vs HF)   4. every HF sha256 == the sha256 recorded in step 1 (the digest-verified bytes)
# The xet cache lives OUTSIDE the farm. Source read-only, NEVER deleted.
# v2 (10-25 05:40 PT, after v1 OOMed at 2G inside MoT): ONE upload worker, xet fixed upload concurrency 2 + 1 file ingested at a time (small RSS);
# self-exits early: an upload killed by OOM (rc 137) twice -> rc 10; STOP file $O/STOP-hfstate -> rc 9 before any attempt; at most 4 attempts.
#   sbatch -w <run host> -J <name> hf_state_host.sbatch <SRC dir> <STEP> <host index> <REPO> <LABEL host> [allowed extra files, comma list]
SRC=$1; STEP=$2; IDX=$3; R=$4; LABEL=$5; EXTRA=${6:-}
H=$(hostname -s); K=dcp-$STEP; RS=${R#*/}
D=/data/home/amartyavarma/setup-logs/hfb; PY=/data/home/amartyavarma/venvs/duplex/bin/python; PYT=/data/home/amartyavarma/venvs/qwen-s2/bin/python
T=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/tools; O=/data/home/amartyavarma/ppm/ppmvox/companions/logs/hfstate-1025
F=/scratch/amartyavarma/hfb/st-$RS-$K-$LABEL; XC=/scratch/amartyavarma/hfb/xetc-$RS-$K-$LABEL; SP=$O/spec-$RS-$K-$LABEL.json
TAG=$K-$LABEL; VJ1=$O/verify-$RS-$K-$LABEL.json
LOW="nice -n 19 ionice -c3"
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
step() { echo "== [$(pt)] $LABEL (on $H) $*"; }
T0=$(date +%s)
case "$H" in slinky-9|slinky-10) step "ABORT: never on $H"; exit 1;; esac
[ -n "$LABEL" ] && [ -n "$R" ] && [ -f $SRC/COMPLETE ] && [ "$(cat $SRC/COMPLETE)" = "$STEP" ] || { step "ABORT: $SRC/COMPLETE missing or != $STEP"; exit 2; }
FREE=$(df -B1 --output=avail /scratch | tail -1)
[ "$FREE" -ge 2000000000 ] || { step "ABORT: /scratch free $FREE < 2 GB (farm = symlinks + upload metadata, xet cache without chunk cache: ~100 MB)"; exit 5; }
step "job $SLURM_JOB_ID: $SRC ($(ls -A $SRC | wc -l) files, $(du -sb $SRC | cut -f1) B) -> private $R/$K/$LABEL/; /scratch free $((FREE / 1000000000)) GB"
# ---- 1. digest check
export VERIFY_SAFETY_PY=/data/home/amartyavarma/ppm/ppmvox/code/safety_ppm.py VERIFY_HOST_LABEL=$LABEL VERIFY_ALLOW_EXTRA=$EXTRA OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
$LOW $PYT -I $T/verify_dcp_host.py $SRC $STEP $IDX 8 $VJ1 1 || { step "ABORT: digest / layout check FAILED ($VJ1); nothing uploaded"; exit 3; }
# ---- 2-4. upload + verifies
source /data/home/amartyavarma/ppm/scratch_guard.sh
mkdir -p $F $XC
echo "{\"dirs\": [[\"$SRC\", \"$K/$LABEL\"]]}" > $SP
export HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$XC HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HFB_OWNER=$SLURM_JOB_ID
export HF_XET_FIXED_UPLOAD_CONCURRENCY=${HFS_XET_UPC:-2} HF_XET_MAX_CONCURRENT_FILE_INGESTION=1 HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY=false
# --cpu-bind=none puts all 64 CPUs in the affinity -> hf_xet starts 64 tokio workers (anon RSS 1.75 GB in the slinky-4 test = the 10-25 OOM at 2G);
# 2 workers: 0.44 GB (memtest 30319). Same for rayon / OpenMP.
export TOKIO_WORKER_THREADS=2 RAYON_NUM_THREADS=2 OMP_NUM_THREADS=1
NWK=${HFS_WORKERS:-1}; noom=0
UP="$PY $D/hfb_up.py --repo $R --type model --farm $F --spec $SP --manifest-name MANIFEST.$K.$LABEL.tsv --tag $TAG --require $SRC/COMPLETE --workers $NWK --verify-out $O --defer-sample"   # no 3-file download on MoT hosts (sha256all checks every file)
bo=30; att=0; vfail=0; hrc=0
while true; do
  att=$((att+1))
  [ -f $O/STOP-hfstate ] && { step "STOP file $O/STOP-hfstate: exiting before attempt $att"; hrc=9; break; }
  if ! $LOW $UP --verify-only --no-verify; then step "attempt $att: repo create / farm failed; retry ${bo}s"; [ $att -ge 4 ] && { hrc=2; break; }; sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  step "attempt $att: uploading"
  $LOW $PY $D/hfb_up.py --child --repo $R --type model --farm $F --spec $SP --workers $NWK; rc=$?
  [ $rc = 137 ] && noom=$((noom+1))
  [ $noom -ge 2 ] && { step "upload killed by OOM (rc 137) $noom times: giving up (rc 10)"; hrc=10; break; }
  if [ $rc -ne 0 ]; then step "attempt $att: upload rc $rc; retry ${bo}s"; [ $att -ge 4 ] && { hrc=1; break; }; sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  step "attempt $att: upload done after $(( $(date +%s) - T0 )) s; verifying"
  $LOW $UP --verify-only; vrc=$?
  if [ $vrc -eq 0 ]; then
    $LOW $PY $D/hfb_sha_all.py --repo $R --type model --farm $F --tag $TAG --threads 2 --verify-out $O; vrc=$?
    [ $vrc -eq 0 ] && break
  fi
  vfail=$((vfail+1)); step "VERIFY FAIL rc $vrc ($vfail)"; [ $vfail -ge 4 ] && { hrc=3; break; }; sleep $bo
done
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
[ $hrc -eq 0 ] && guard_rm "$F" "$XC"   # own symlink farm + caches only (never the save)
[ $hrc -eq 0 ] && echo "ok $(pt) job ${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} on $H" > $O/ok-$RS-$K-$LABEL   # success marker (srun rc can be != 0 after a contained OOM event)
step "$K/$LABEL HF backup rc $hrc (0 = digest check + hfb_up VERIFY PASS + sha256all PASS + cross-check, private $R) after $(( $(date +%s) - T0 )) s"
exit $hrc

#!/bin/bash
# LEG-1 HOOK (dcp-17850 -> 18022 on plans [plan 1, npl2, npl3], no npl5). Callable from C's 16-host staging job, one task per host; idempotent.
#   on slinky-15 (plays slot slinky-9): STAGE slot 9's node-local fresh-pose state, then CHECK it:
#     plan dirs /scratch/amartyavarma/ppm-fresh-pose{,-npl2,-npl3}/{PLAN.json,PLAN.npz} <- G/plans (off-host byte-identical rebuilds of slinky-9's
#       plans, ppmvox v5slot7 rebuild_plan.py from the Weka scans; sha256 == code-811/fresh/append/npl3/dryrun/slinky-9.json; existing files are
#       compared, never overwritten)
#     npl3 parquet /scratch/amartyavarma/npl-append/npl3/jobs/... <- G/npl3 (the 5,434 files / 10.24 GB slot 9 reads from 17850 to the end of its
#       plan 1-3 rows; fetched from the HF delivery-3 bucket, xet-verified); files already present at the right size are kept
#   on every other 811 host: CHECK only (its own plans + npl3 window); FETCH_MISSING=1 fetches missing npl3 files from the HF delivery-3 bucket.
#   CHECK = bin/leg1_check.py: digests == dcp-17850 pose_digests[key] (+ node-local dcp-17850 train_state if present), every needed parquet
#   present at its size, the trainer's micro of all 8 ranks at 17850 and 18021 (MICROS=0 skips); then xet verify of the slot-9 window (slinky-15).
# NEVER touches slinky-9 (refuses to run there). Output: G/results/<host>.json, G/results/<host>.{OK,FAIL}, log G/logs/hook-<host>.log.
# usage: bash /data/home/amartyavarma/ppm/pose5/npl5/bin/leg1_slot9_hook.sh     (exit 0 = ready)
set -uo pipefail
G=/data/home/amartyavarma/ppm/pose5/npl5/leg1/slot9; B=/data/home/amartyavarma/ppm/pose5/npl5/bin; S=/scratch/amartyavarma
A=/data/home/amartyavarma/ppm/code-811/fresh/append
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
P=$(hostname -s); [ "$P" = slinky-9 ] && { echo "REFUSED: never on slinky-9"; exit 1; }
if [ -n "${HOOK_AS:-}" ]; then   # rehearsal on a SPARE only: act as HOOK_AS (writes the canonical dirs on the spare; clean them up after)
  case "$P" in slinky-4|slinky-10|slinky-18) P=$HOOK_AS;; *) echo "HOOK_AS is for a spare rehearsal only"; exit 1;; esac
fi
K=$([ "$P" = slinky-15 ] && echo slinky-9 || echo $P)
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $G/results $G/logs; rm -f "${G:?}/results/${P:?}.OK" "${G:?}/results/${P:?}.FAIL"
LOG=$G/logs/hook-$P.log
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [$P=$K] $*" | tee -a $LOG; }
die() { say "FAIL: $*"; echo "$(pt) $*" > $G/results/$P.FAIL; exit 1; }
T0=$(date +%s); say "start: /scratch free $(df -BG --output=avail $S | tail -1 | tr -d ' ')"
if [ "$P" = slinky-15 ]; then
  [ -f $G/npl3.STAGED ] || die "staged npl3 window missing on /data ($G/npl3.STAGED)"
  for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3; do
    if [ -e $S/$d ]; then
      for f in PLAN.json PLAN.npz; do cmp -s $G/plans/$d/$f $S/$d/$f || die "$S/$d/$f exists and differs from the slinky-9 plan"; done
    else
      rm -rf "${S:?}/${d:?}.tmp-hook"; mkdir -p $S/$d.tmp-hook && cp $G/plans/$d/PLAN.json $G/plans/$d/PLAN.npz $S/$d.tmp-hook/ && mv $S/$d.tmp-hook $S/$d || die "copy $d"
    fi
    for f in PLAN.json PLAN.npz; do
      w=$(python3 -c "import json;print(json.load(open('$G/PLAN_SHA.json'))['plan_sha256']['$d/$f'])")
      [ "$(sha256sum < $S/$d/$f | cut -c1-64)" = "$w" ] || die "$S/$d/$f sha256 != recorded"
    done
  done
  say "plan dirs in place + sha256 ok"
  # npl3 window: copy what is missing from the /data staging (parallel, tmp + rename), keep what is present at the right size
  $PYV -I - <<EOF >> $LOG 2>&1 || die "npl3 copy"
import os, shutil
from concurrent.futures import ThreadPoolExecutor
G, D = "$G", "$S/npl-append/npl3"
rows = [l.rstrip("\n").split("\t") for l in open(f"{G}/NEED_npl3.tsv")]
def one(r):
    rel, sz = r[0], int(r[1]); dst = f"{D}/{rel}"
    if os.path.isfile(dst) and os.path.getsize(dst) == sz: return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copyfile(f"{G}/npl3/{rel}", dst + ".part"); assert os.path.getsize(dst + ".part") == sz; os.replace(dst + ".part", dst); return 1
with ThreadPoolExecutor(16) as ex: n = sum(ex.map(one, rows))
print(f"npl3 window: {len(rows)} files, copied {n}, present {len(rows) - n}", flush=True)
EOF
  say "npl3 window in place ($(tail -n 1 $LOG))"
  V=$G/verify-run-$P; rm -rf "${G:?}/verify-run-${P:?}"; mkdir -p $V/placement; cp $G/fetchw/LISTING.json $V/; cp $G/NEED_npl3.tsv $V/placement/$(hostname -s).tsv
  VERIFY_THREADS=16 $PYV -I $A/src_verify.py $V $S/npl-append/npl3 >> $LOG 2>&1 || die "xet verify of the npl3 window"
  say "npl3 window xet-verified"
fi
NT=$G/results/missing-$P.tsv
$PYQ -I $B/leg1_check.py $S $K $G/results/$P.json --need-tsv $NT $([ "${MICROS:-1}" = 1 ] && echo --micros) >> $LOG 2>&1; rc=$?
if [ $rc = 2 ] && [ "${FETCH_MISSING:-0}" = 1 ]; then
  say "$(wc -l < $NT) npl3 files missing -> fetching from the HF delivery-3 bucket"
  F=$G/fetch-run-$P; rm -rf "${G:?}/fetch-run-${P:?}"; mkdir -p $F/placement; cp $G/fetchw/LISTING.json $F/; cp $NT $F/placement/$(hostname -s).tsv
  HF_RATE=${HF_RATE:-30} THREADS_HF=32 $PYV -I $A/src_fetch.py $F $S/npl-append/npl3 >> $LOG 2>&1 && VERIFY_THREADS=16 $PYV -I $A/src_verify.py $F $S/npl-append/npl3 >> $LOG 2>&1 \
    || die "fetch / verify of missing npl3 files"
  $PYQ -I $B/leg1_check.py $S $K $G/results/$P.json $([ "${MICROS:-1}" = 1 ] && echo --micros) >> $LOG 2>&1; rc=$?
fi
[ $rc = 0 ] || die "leg1_check rc $rc: $(python3 -c "import json;d=json.load(open('$G/results/$P.json'));print('missing',d.get('missing'),d.get('missing_bytes'),'B; e.g.',d.get('missing_examples'))" 2>/dev/null) (FETCH_MISSING=1 fetches them)"
echo "$(pt) $(( $(date +%s) - T0 )) s" > $G/results/$P.OK
say "OK in $(( $(date +%s) - T0 )) s: digests == dcp-17850, npl3 window complete$([ "${MICROS:-1}" = 1 ] && echo ', 16 micros built'); /scratch free $(df -BG --output=avail $S | tail -1 | tr -d ' ')"

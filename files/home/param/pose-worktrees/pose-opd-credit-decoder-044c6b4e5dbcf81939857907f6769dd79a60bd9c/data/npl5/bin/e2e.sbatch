#!/bin/bash
#SBATCH -J npl5-e2e -p all -w slinky-4 -x slinky-9 -N1 -n1 -c2 --mem=16G --gres=none -t 0:45:00 --time-min=0:20:00 --deadline=2026-10-27T17:50:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/e2e-%j.out
# npl5 END-TO-END DRY RUN on a spare (slinky-4), every slot in turn, as its host would at launch: the slot's REAL 4 plans at the canonical dirs
# (plans 1-3 = bin/rebuild13.py off-host rebuilds, digests == dcp-17850; npl5 = W/plans) -> the parquet its micros need (HF buckets) -> bin/
# dryrun_offhost.py host: plans load + digests == RESULT (dcp-17850 prefix + npl5), npl5 sha256 == PLANS.json, the trainer's micros of all 8
# ranks at 17850 / the first npl5 row / the last step. Slot 9 is checked under its plan key slinky-9 (played by slinky-15). Canonical dirs on
# this spare are created per slot and removed after. -> W/e2e/<slot>.json, W/e2e/E2E.json
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; B=$W/bin; S=/scratch/amartyavarma; A=/data/home/amartyavarma/ppm/code-811/fresh/append
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
END=$(python3 -c "import json;print(json.load(open('$W/dryrun/RESULT.json'))['max_steps'])")
O=$W/e2e; mkdir -p $O
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
DIRS="ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3 ppm-fresh-pose-npl5"
clean() { for d in $DIRS; do rm -rf "${S:?}/${d:?}"; done; rm -rf "${S:?}/npl-append/npl3" "${S:?}/npl-append/npl5"; rmdir "${S:?}/npl-append" 2>/dev/null; }   # (exact test paths; this job aborts if they pre-exist)
for d in $DIRS npl-append; do [ -e $S/$d ] && { echo "ABORT: $S/$d exists on $(hostname -s)"; exit 1; }; done
ok=0
for K in ${SLOTS:-slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-9 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19}; do
  case "$(hostname -s)" in slinky-10|slinky-18)   # (main: slinky-10/18 go to motasr2 at the ~11:30 PT switch: stop by 11:20 PT there)
    [ "$(TZ=America/Los_Angeles date +%H%M)" -ge "${E2E_DEADLINE:-1045}" ] && { echo "== $(pt) deadline on $(hostname -s): stopping before $K"; break; };; esac
  t0=$(date +%s); clean
  for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3; do mkdir -p $S/$d; cp $W/rebuild13/$K/$d/PLAN.json $W/rebuild13/$K/$d/PLAN.npz $S/$d/; done
  mkdir -p $S/ppm-fresh-pose-npl5; cp $W/plans/$K/ppm-fresh-pose-npl5/PLAN.json $W/plans/$K/ppm-fresh-pose-npl5/PLAN.npz $S/ppm-fresh-pose-npl5/
  N=$O/need-$K; rm -rf "${O:?}/need-${K:?}"
  $PYQ -I $B/e2e_need.py $S $K $END $N > $O/$K.need.log 2>&1 || { echo "== $(pt) $K need FAILED"; continue; }
  unset HF_HUB_OFFLINE
  HF_RATE=30 THREADS_HF=16 $PYV -I $A/src_fetch.py $N/n3 $S/npl-append/npl3 > $O/$K.fetch3.log 2>&1 && \
  HF_RATE=30 THREADS_HF=16 $PYV -I $A/src_fetch.py $N/n5 $S/npl-append/npl5 > $O/$K.fetch5.log 2>&1 && \
  $PYV -I $A/src_verify.py $N/n3 $S/npl-append/npl3 > $O/$K.verify3.log 2>&1 && $PYV -I $A/src_verify.py $N/n5 $S/npl-append/npl5 > $O/$K.verify5.log 2>&1 \
    || { echo "== $(pt) $K fetch/verify FAILED"; export HF_HUB_OFFLINE=1; continue; }
  export HF_HUB_OFFLINE=1
  if NPL5_W=$W $PYQ -I $B/dryrun_offhost.py host $O $K > $O/$K.host.log 2>&1; then
    ok=$((ok + 1)); echo "== $(pt) $K OK ($(( $(date +%s) - t0 )) s): $(tail -n 1 $O/$K.host.log | cut -c1-200)"
  else
    echo "== $(pt) $K HOST CHECK FAILED: $(tail -n 3 $O/$K.host.log | cut -c1-300)"
  fi
done
clean
echo "{\"slots_ok\": $ok, \"end\": $END, \"time\": \"$(pt)\"}" > $O/E2E.json
echo "== $(pt) e2e: $ok slots OK; canonical dirs removed from $(hostname -s)"

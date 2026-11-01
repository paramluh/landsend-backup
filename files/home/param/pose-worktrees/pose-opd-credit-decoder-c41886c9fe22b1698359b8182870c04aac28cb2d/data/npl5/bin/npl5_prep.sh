#!/bin/bash
# npl5 PREP on one host (= C's B_NPL5_PREP, called by kit/stage_host.sh step 6 AFTER B's slot-9 / npl3 hook, with NPL5_AHEAD exported).
# Idempotent; CPU-light; ~1 min. Never on slinky-9.
#  1 npl5 plan dir: W/plans/<key>/ppm-fresh-pose-npl5 -> /scratch/amartyavarma/ppm-fresh-pose-npl5 (existing files compared, never overwritten;
#    sha256 == W/plans/PLANS.json); key = slinky-9 on slinky-15, else the host (slinky-7 = slot slinky-10)
#  2 a SMALL first window (npl5 files read before this host's first npl5 step + PREP_STEPS, default 10 steps ~0.6 GB) via npl5_roll.py window,
#    from the HF bucket, xet-checked: enough for the checks below; the in-job roll daemon (npl5_roll_super.sbatch) fetches the rest ahead of
#    the trainer (its first npl5 step is ~184 steps = ~3.5 h after 17850, so the daemon has ~1.6 h before it even needs to start)
#  3 fresh_pose.py check (rows 0,1,2: raw parquet == built rows / micros)   4 dryrun_offhost.py host: the 4 plans load, digests == the dry run's
#    (dcp-17850 prefix + npl5), the trainer's micros of all 8 ranks at 17850 (npl3) and at the first npl5 row
#  5 echo "$NPL5_AHEAD" is NOT written here (C writes kit/state/AHEAD-<host>; the daemon re-reads it every pass)
# -> W/launch/prep/<host>.{OK,FAIL}, logs W/launch/prep/<host>/
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; B=$W/bin; S=/scratch/amartyavarma; C=/data/home/amartyavarma/ppm/code-811
PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
P=$(hostname -s); [ "$P" = slinky-9 ] && { echo "REFUSED on slinky-9"; exit 1; }
if [ -n "${PREP_AS:-}" ]; then   # rehearsal on a SPARE only (writes the canonical dirs there; clean up after)
  case "$P" in slinky-4|slinky-10|slinky-18) P=$PREP_AS;; *) echo "PREP_AS is for a spare rehearsal only"; exit 1;; esac
fi
K=$([ "$P" = slinky-15 ] && echo slinky-9 || echo $P)
O=$W/launch/prep; L=$O/$P; mkdir -p $L; rm -f "${O:?}/${P:?}.OK" "${O:?}/${P:?}.FAIL"
F=$C/fresh; PRIOR=$F/scan,$F/append/npl2/scan,$F/append/npl3/scan
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NPL5_W=$W
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [npl5-prep $P=$K] $*" | tee -a $L/prep.log; }
die() { say "FAIL: $*"; echo "$(pt) $*" > $O/$P.FAIL; exit 1; }
T0=$(date +%s)
[ -f $W/state/dryrun.ok ] || die "the off-host plans + dry run are not done ($W/state/dryrun.ok missing)"
END=$(python3 -c "import json;print(json.load(open('$W/dryrun/RESULT.json'))['max_steps'])")
say "start: key $K, END $END, NPL5_AHEAD ${NPL5_AHEAD:-unset}, /scratch free $(df -BG --output=avail $S | tail -1 | tr -d ' ')"
D=$S/ppm-fresh-pose-npl5
read J5 Z5 < <(python3 -c "import json;s=json.load(open('$W/plans/PLANS.json'))['slots']['$K']['sha256'];print(s['PLAN.json'],s['PLAN.npz'])")
if [ -e $D ]; then for f in PLAN.json PLAN.npz; do cmp -s $W/plans/$K/ppm-fresh-pose-npl5/$f $D/$f || die "$D/$f exists and differs"; done
else rm -rf "${D:?}.tmp-prep"; mkdir -p $D.tmp-prep && cp $W/plans/$K/ppm-fresh-pose-npl5/PLAN.json $W/plans/$K/ppm-fresh-pose-npl5/PLAN.npz $D.tmp-prep/ && mv $D.tmp-prep $D || die "copy plan"; fi
[ "$(sha256sum < $D/PLAN.json | cut -c1-64)" = "$J5" ] && [ "$(sha256sum < $D/PLAN.npz | cut -c1-64)" = "$Z5" ] || die "npl5 plan sha256 != PLANS.json"
for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3; do [ -f $S/$d/PLAN.npz ] || die "$S/$d missing (B's slot-9/npl3 hook first)"; done
say "npl5 plan in place, sha256 ok"
unset HF_HUB_OFFLINE
NPL5_ROLL_DIR=$O/roll nice -n19 $PYQ -I $B/npl5_roll.py window --end $END --upto-rel ${PREP_STEPS:-10} --phys $P --key $K --hf-rate ${HF_RATE:-60} --threads 32 > $L/window.log 2>&1 \
  || die "first window fetch ($L/window.log)"
export HF_HUB_OFFLINE=1
say "first window: $(grep -h 'ok:\|window:' $L/window.log | tail -n 2 | cut -c40-220 | tr '\n' ' ')"
env FRESH_SRC=$S/npl-append/npl5/jobs FRESH_OUT=$D FRESH_SCAN=$L FRESH_TAG=npl5 FRESH_PRIOR_SCANS=$PRIOR FRESH_PRIOR_VIDEO=overlap FRESH_HOST=$K \
  $PYQ -I $F/fresh_pose.py check --rows 0,1,2 > $L/check.log 2>&1 || die "fresh_pose check ($L/check.log)"
SKIP_LAST_MICRO=1 $PYQ -I $B/dryrun_offhost.py host $L $K > $L/dryhost.log 2>&1 || die "host dry run ($L/dryhost.log)"
echo "$(pt) $(( $(date +%s) - T0 )) s" > $O/$P.OK
say "OK in $(( $(date +%s) - T0 )) s: plan sha ok, first window in place, check rows 0-2 ok, $(python3 -c "import json;d=json.load(open('$L/host-$K.json'));print(len(d['micros']),'micros (17850 + first npl5), digests ok')"); /scratch free $(df -BG --output=avail $S | tail -1 | tr -d ' ')"

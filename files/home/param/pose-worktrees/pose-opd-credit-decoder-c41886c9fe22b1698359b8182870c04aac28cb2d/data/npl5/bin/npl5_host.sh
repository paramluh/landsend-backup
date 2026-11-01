#!/bin/bash
# npl5 LAUNCH-DAY per-host step (one task per physical host, from npl5_launchday.sbatch). Puts this host's npl5 share and plan at the canonical
# paths, plus (slinky-15 = slot slinky-9) slinky-9's plan-1/npl2/npl3 plan dirs, and any npl3 parquet this slot still reads after 17850.
#  1 plan dirs: W/plans/<key>/ppm-fresh-pose-npl5 -> /scratch/amartyavarma/ppm-fresh-pose-npl5 (sha256 == PLANS.json); slinky-15 also gets
#    ppmvox/launch/v5slot7/plans/slinky-9/{ppm-fresh-pose,-npl2,-npl3} (byte-identical rebuilds, sha256 == npl3/dryrun/slinky-9.json)
#  2 npl5 parquet from the HF bucket (src_fetch.py, unchanged; placement = W/launch/npl5/placement/<this host>.tsv) -> /scratch/amartyavarma/
#    npl-append/npl5, in parallel with 3: the npl3 parquet still needed after 17850 (dryrun_offhost.py need3 on the node-local plans) that
#    is missing here -> fetched from the delivery-3 bucket into /scratch/amartyavarma/npl-append/npl3
#  4 xet verify of both (src_verify.py)   5 fresh_pose.py check (npl5, 3 rows)   6 dryrun_offhost.py host (4 plans load + digests ==
#    RESULT / dcp-17850, micros at 17850 / first npl5 row / last step)   7 /scratch free vs the gate.
set -uo pipefail
echo "SUPERSEDED (10-27 audit #15): the launch-day path is C's kit/stage_host.sh -> B's bin/leg1_slot9_hook.sh + bin/npl5_prep.sh, and the roll supervisor bin/npl5_roll_super.sbatch. This full-share copy no longer fits the hosts." >&2; exit 1
B=/data/home/amartyavarma/ppm/pose5/npl5/bin; W=${NPL5_W:-/data/home/amartyavarma/ppm/pose5/npl5}; L=${L:?run dir}; C=/data/home/amartyavarma/ppm/code-811; A=$C/fresh/append
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
P=${PHYS_AS:-$(hostname -s)}; K=$([ "$P" = slinky-15 ] && echo slinky-9 || echo $P)
# REHEARSAL on a spare only (never on a production host): PHYS_AS=<host it pretends to be>, R13=<dir with that key's plan 1-3 dirs> (copied like
# slinky-15's), NO_NPL3_FETCH=1 (report the npl3 remainder, do not fetch delivery 3), SKIP_RESUME_MICRO=1 (no npl3 parquet here).
[ -n "${PHYS_AS:-}" ] && case "$(hostname -s)" in slinky-4|slinky-10|slinky-18) ;; *) echo "PHYS_AS is for a spare rehearsal only"; exit 1;; esac
S=/scratch/amartyavarma; V=$([ "${WITH_JSON:-0}" = 1 ] && echo npl5json || echo npl5)
F=$C/fresh; PRIOR=$F/scan,$F/append/npl2/scan,$F/append/npl3/scan
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NPL5_W=$W
mkdir -p $L/logs $L/dry
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [$P=$K] $*" | tee -a $L/logs/host-$P.log; }
die() { say "FAIL: $*"; echo "$(pt) $*" > $L/FAIL-$P; exit 1; }
tm() { printf "launchday_%s\t%s\t%ss\t%s\t%s\n" "$1" "$K" "$2" "$P" "$(pt)" >> $W/TIMES.tsv; }
free_gb() { df -BG --output=avail /scratch | tail -1 | tr -dc 0-9; }
T0=$(date +%s); F0=$(free_gb); NEED=$(python3 -c "import json;print(round(json.load(open('$W/launch/PLACEMENT_PHYS.json'))['shares']['$V']['$P']['GB']))")
say "start: /scratch free ${F0}G, npl5 share ${NEED} GB ($V), gate ${GATE_GB:-122}G after"
[ "$F0" -ge $(( NEED * 93 / 100 + ${GATE_GB:-122} )) ] || [ "${FORCE_DISK:-0}" = 1 ] || die "/scratch free ${F0}G < share ${NEED} GB (~$(( NEED * 93 / 100 )) GiB) + gate ${GATE_GB:-122}G (FORCE_DISK=1 to override)"
# ---- 1 plan dirs
put_plan() {   # <src dir> <dst dir> <sha256 PLAN.json> <sha256 PLAN.npz>
  local src=$1 dst=$2
  if [ -e $dst ]; then for f in PLAN.json PLAN.npz; do cmp -s $src/$f $dst/$f || die "$dst/$f exists and differs from $src/$f"; done
  else rm -rf "${dst:?}.tmp"; mkdir -p $dst.tmp; cp $src/PLAN.json $src/PLAN.npz $dst.tmp/ && mv $dst.tmp $dst || die "copy $src"; fi
  [ "$(sha256sum < $dst/PLAN.json | cut -c1-64)" = "$3" ] && [ "$(sha256sum < $dst/PLAN.npz | cut -c1-64)" = "$4" ] || die "$dst sha256 != recorded"
}
read J5 Z5 < <(python3 -c "import json;s=json.load(open('$W/plans/PLANS.json'))['slots']['$K']['sha256'];print(s['PLAN.json'],s['PLAN.npz'])")
put_plan $W/plans/$K/ppm-fresh-pose-npl5 $S/ppm-fresh-pose-npl5 $J5 $Z5
if [ "$P" = slinky-15 ] || [ -n "${R13:-}" ]; then
  R9=${R13:-/data/home/amartyavarma/ppm/ppmvox/launch/v5slot7/plans/slinky-9}
  for t in npl npl2 npl3; do
    d=ppm-fresh-pose$([ $t = npl ] || echo -$t)
    read Jt Zt < <(python3 -c "import json;s=json.load(open('$A/npl3/dryrun/$K.json'))['plan_sha256'];print(s['$t/PLAN.json'],s['$t/PLAN.npz'])")
    put_plan $R9/$d $S/$d $Jt $Zt
  done
fi
for d in ppm-fresh-pose ppm-fresh-pose-npl2 ppm-fresh-pose-npl3; do [ -f $S/$d/PLAN.npz ] || die "$S/$d missing on $P (slot $K)"; done
say "plan dirs in place ($(( $(date +%s) - T0 )) s)"
# ---- 2 + 3 fetch npl5 share || npl3 remainder
t0=$(date +%s)
mkdir -p $L/$V/fetch; [ -e $L/$V/LISTING.json ] || cp $W/launch/$V/LISTING.json $L/$V/ 2>/dev/null; mkdir -p $L/$V/placement; cp $W/launch/$V/placement/$P.tsv $L/$V/placement/$(hostname -s).tsv   # (src_fetch / src_verify key on the real hostname; = $P except in a rehearsal)
# ROLL=1 (default, 10-27 main GO: no room for full shares): only the INITIAL WINDOW (npl5 files read before step UPTO, default 18160 = the
# first ~126 npl5 steps; ~6-7 GB) via npl5_roll.py window; the rolling daemon inside the trainer job (npl5_roll_super.sbatch) keeps it ahead.
# ROLL=0: the full share (src_fetch.py over the whole placement, as in the original plan).
END5=$(python3 -c "import json;print(json.load(open('$W/dryrun/RESULT.json'))['max_steps'])")
if [ "${ROLL:-1}" = 1 ]; then
  ( NPL5_ROLL_DIR=$L/roll nice -n19 $PYQ -I $B/npl5_roll.py window --end $END5 --upto ${UPTO:-18160} --phys $P --key $K --hf-rate ${HF_RATE:-90} --threads ${THREADS_HF:-64} \
      > $L/logs/fetch5-$P.log 2>&1 ) & F5=$!
else
  ( HF_RATE=${HF_RATE:-90} THREADS_HF=${THREADS_HF:-96} $PYV -I $A/src_fetch.py $L/$V $S/npl-append/npl5 > $L/logs/fetch5-$P.log 2>&1 ) & F5=$!
fi
$PYQ -I $B/dryrun_offhost.py need3 $L/dry $S $K $(hostname -s) > $L/logs/need3-$P.log 2>&1 || die "need3 ($L/logs/need3-$P.log)"
N3=$L/dry/need3-$K-ppm-fresh-pose-npl3
if [ -d $N3 ] && [ "${NO_NPL3_FETCH:-0}" = 1 ]; then
  say "npl3 remainder missing here (rehearsal, not fetched): $(python3 -c "import json;d=json.load(open('$L/dry/need3-$K.json'));print(d['missing'])")"
elif [ -d $N3 ]; then
  say "npl3: $(python3 -c "import json;d=json.load(open('$L/dry/need3-$K.json'));print(d['missing'])") -> fetching from the delivery-3 bucket"
  HF_RATE=${HF_RATE3:-30} THREADS_HF=32 $PYV -I $A/src_fetch.py $N3 $S/npl-append/npl3 > $L/logs/fetch3-$P.log 2>&1 || die "npl3 fetch ($L/logs/fetch3-$P.log)"
  $PYV -I $A/src_verify.py $N3 $S/npl-append/npl3 > $L/logs/verify3-$P.log 2>&1 || die "npl3 verify"
fi
wait $F5 || die "npl5 fetch ($L/logs/fetch5-$P.log)"
tm fetch $(( $(date +%s) - t0 )); say "fetched ($(( $(date +%s) - t0 )) s): $(tail -n 1 $L/logs/fetch5-$P.log | cut -c1-200)"
if [ "${ROLL:-1}" != 1 ]; then   # (ROLL=1: every file was xet-checked by npl5_roll.py before its rename)
  t0=$(date +%s); VERIFY_THREADS=32 $PYV -I $A/src_verify.py $L/$V $S/npl-append/npl5 > $L/logs/verify5-$P.log 2>&1 || die "npl5 verify ($L/logs/verify5-$P.log)"
  tm verify $(( $(date +%s) - t0 ))
fi
# ---- 5 check, 6 host dry run
t0=$(date +%s)
env FRESH_SRC=$S/npl-append/npl5/jobs FRESH_OUT=$S/ppm-fresh-pose-npl5 FRESH_SCAN=$L FRESH_TAG=npl5 FRESH_PRIOR_SCANS=$PRIOR FRESH_PRIOR_VIDEO=overlap FRESH_HOST=$K \
  $PYQ -I $C/fresh/fresh_pose.py check $([ "${ROLL:-1}" = 1 ] && echo "--rows 0,1,2") > $L/logs/check-$P.log 2>&1 || die "fresh_pose check ($L/logs/check-$P.log)"
SKIP_LAST_MICRO=${ROLL:-1} $PYQ -I $B/dryrun_offhost.py host $L/dry $K > $L/logs/dryhost-$P.log 2>&1 || die "host dry run ($L/logs/dryhost-$P.log)"
tm checks $(( $(date +%s) - t0 ))
F1=$(free_gb); echo "$(pt) free ${F0}G -> ${F1}G in $(( $(date +%s) - T0 )) s" > $L/OK-$P
say "OK in $(( $(date +%s) - T0 )) s; /scratch free ${F0}G -> ${F1}G $([ $F1 -ge ${GATE_GB:-122} ] && echo '>= gate' || echo "BELOW the ${GATE_GB:-122}G gate")"

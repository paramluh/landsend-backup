#!/bin/bash
# npl5 off-host dry run (inside the plan job on a spare; after plan_offhost.py): ends/max_steps -> fetch the few parquet files the checks
# need (HF, delivery 5 only) to the CANONICAL /scratch/amartyavarma/npl-append/npl5 on THIS spare -> fresh_pose.py check per slot (code-811,
# unchanged) -> trainer micros per rank -> trainer guard on dcp-17850's real train_state (code-811 trainer; and the s2p copy under the launch
# env if it imports) -> code-811 test_append_cpu.py map3 regression -> remove the canonical dir again. Logs W/logs/dry-*.log, W/dryrun/*.
set -uo pipefail
W=${NPL5_W:-/data/home/amartyavarma/ppm/pose5/npl5}; B=/data/home/amartyavarma/ppm/pose5/npl5/bin; D=$W/dryrun
C=/data/home/amartyavarma/ppm/code-811; A=$C/fresh/append; S2P=/data/home/amartyavarma/ppm/pose5/code-811-s2p
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
ENV5=${ENV5:-/data/home/amartyavarma/ppm/pose5/prod_811_fresh_append5_s2p.env}
CAN=/scratch/amartyavarma/npl-append/npl5; ME=$(hostname -s); N="nice -n19 ionice -c3"
F=$C/fresh; PRIOR=$F/scan,$F/append/npl2/scan,$F/append/npl3/scan
export NPL5_W=$W OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p $D $W/logs $W/state
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) [dry@$ME] $*" | tee -a $W/logs/dryrun.log; }
die() { say "ABORT: $*"; exit 1; }
step() { local n=$1 t0=$(date +%s); shift; "$@" > $W/logs/dry-$n.log 2>&1 || die "$n failed: $W/logs/dry-$n.log"; printf "dry_%s\t-\t%ss\t%s\t%s\n" $n "$(( $(date +%s) - t0 ))" "$ME" "$(pt)" >> $W/TIMES.tsv; say "$n ok ($(( $(date +%s) - t0 )) s)"; }
[ -e $CAN ] && die "$CAN already exists on $ME (not ours): refusing to write there"
step ends $N $PYQ -I $B/dryrun_offhost.py ends $D
MAXS=$(python3 -c "import json;print(json.load(open('$D/RESULT.json'))['max_steps'])"); say "max_steps $MAXS"
step need $N $PYQ -I $B/dryrun_offhost.py need $D $ME
step fetch env HF_RATE=60 THREADS_HF=32 $N $PYV -I $A/src_fetch.py $D/need $CAN
step verify env VERIFY_THREADS=4 $N $PYV -I $A/src_verify.py $D/need $CAN
chk() { for h in slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-9 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19; do
  env FRESH_SRC=$CAN/jobs FRESH_OUT=$W/plans/$h/ppm-fresh-pose-npl5 FRESH_SCAN=$W/scan FRESH_TAG=npl5 FRESH_PRIOR_SCANS=$PRIOR FRESH_PRIOR_VIDEO=overlap \
    FRESH_HOST=$h $N $PYQ -I $C/fresh/fresh_pose.py check || return 1; done; }
step check chk
step micros $N $PYQ -I $B/dryrun_offhost.py micros $D
guard() {   # <train_py> <suffix> <env file> [drop PPM_S2P]
  ( for v in $(env | grep -oE '^PPM_[A-Za-z0-9_]*'); do unset $v; done
    set -a; source $3; set +a; export PPM_MAX_STEPS=$MAXS; [ "${4:-}" = nos2p ] && unset PPM_S2P PPM_G0COND
    export RANK=0 WORLD_SIZE=128 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 PPM_RUN=ppm-v1-jDRY PPM_RUN_BASE=ppm-v1
    TRAIN_PY=$1 GUARD_SUFFIX=$2 CODE811=$(dirname $1) $N $PYQ $B/dryrun_offhost.py guard $D ); }
step guard_811 guard $C/train_ppm.py "" $ENV5 nos2p
step guard_s2p guard $S2P/train_ppm.py -s2p $ENV5      # (audit S12: FATAL - the launch trainer + the final env must pass, else no dryrun.ok)
step unittest_map3 env OMP_NUM_THREADS=1 $N $PYQ $A/test_append_cpu.py map3
rm -rf "${CAN:?}"; rmdir /scratch/amartyavarma/npl-append 2>/dev/null; say "removed $CAN from $ME"
echo "$(pt) max_steps $MAXS" > $W/state/dryrun.ok
say "DRY RUN OK: max_steps $MAXS (RESULT $D/RESULT.json, GUARD $D/GUARD.json, checks $W/scan/checks)"

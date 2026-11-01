#!/bin/bash
# rolling-fetcher SIM v2 on a SPARE (audit S12): the FINAL npl5_roll.py daemon beside sim_trainer.py v2 (real read sequence of one slot; keep=1 +
# chained hold + pin, a relaunch from the held save with no step line during init, a LOWMEM window). Test dirs only; the daemon writes the
# canonical /scratch/amartyavarma/npl-append/npl5 on this spare (aborts if it exists; removed after).
# usage: sim_run.sh <root with ppm-fresh-pose-npl5> <key> <end> <stop> [extra daemon args]   (AUDIT13=1, RELAUNCH_AT, LOWMEM=A:B, SIM_RATE)
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; B=$W/bin; R=$1; K=$2; END=$3; STOP=$4; shift 4
T=$W/sim/run-${SLURM_JOB_ID:-x}-$K; mkdir -p $T/kit; S=/scratch/amartyavarma
[ -e $S/npl-append/npl5 ] && { echo "ABORT: $S/npl-append/npl5 exists on $(hostname -s)"; exit 1; }
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; AU=$([ "${AUDIT13:-0}" = 1 ] && echo --audit13)
export OMP_NUM_THREADS=1 NPL5_ROLL_DIR=$T/roll NPL5_KIT_STATE=$T/kit NPL5_FAKE_CGROUP=$T/cgroup
mkdir -p $T/cgroup $T/ckpt $T/hold $T/pin
$PY -I $B/sim_trainer.py --root $R --key $K --end $END --stop $STOP $AU --dir $T --rate ${SIM_RATE:-0.5} --fast-rate ${SIM_FAST:-1.0} \
  --relaunch-at ${RELAUNCH_AT:-0} --lowmem ${LOWMEM:-} > $T/sim.out 2>&1 &
SIMP=$!
sleep 5
$PY -I $B/npl5_roll.py daemon --end $END --phys $K --key $K --root $R $AU --ckpt-dir $T/ckpt --hold-dir $T/hold --pin-dir $T/pin --log $T/trainer.log \
  --poll 5 "$@" > $T/daemon.out 2>&1 &
DP=$!
sleep 3
$PY -I $B/npl5_roll.py daemon --end $END --phys $K --key $K --root $R $AU --ckpt-dir $T/ckpt --hold-dir $T/hold --pin-dir $T/pin --log $T/trainer.log \
  --poll 5 "$@" > $T/daemon2.out 2>&1; echo "second daemon rc $? (3 = refused by the host lock)" >> $T/RESULT.txt
wait $SIMP; rc=$?
kill $DP 2>/dev/null; wait $DP 2>/dev/null
{ echo "== sim rc $rc"; tail -n 1 $T/sim.out; echo "daemon passes $(grep -c 'ok:\|ROLL ' $T/daemon.out); alerts: $(grep -o 'ROLL [A-Z+]*' $T/daemon.out | sort | uniq -c | tr '\n' ' ')";
  grep -m3 "LOWMEM\|resumed at step\|floor lowered" $T/daemon.out | cut -c1-220; ls $T/roll; } >> $T/RESULT.txt
cat $T/RESULT.txt
du -sh $S/npl-append/npl5 2>/dev/null
rm -rf "${S:?}/npl-append/npl5"; rmdir "${S:?}/npl-append" 2>/dev/null; echo "cleaned $S/npl-append/npl5 on $(hostname -s)"
exit $rc

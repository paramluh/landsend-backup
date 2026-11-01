#!/bin/bash
# full-size save/resume gate inside the real run: wait for the first hourly save of job $1 to land, then STOP_AFTER_SAVE, wait for the clean
# exit, resubmit the same env (PPM_RESUME_FROM=auto) on the same hosts, wait for the resume line (per-rank digests asserted) + 2 steps.
J=$1; L=/data/home/amartyavarma/ppm/logs; C=/data/home/amartyavarma/ppm/code; CTL=/data/home/amartyavarma/ppm/control/ppm-v1
pt() { TZ=America/Los_Angeles date '+%T PT'; }
until grep -qE "\[save\] /scratch|Traceback|== exit" $L/ppm-$J.log; do sleep 30; done
grep -E "\[save\]" $L/ppm-$J.log | head -3; echo "[$(pt)] first save landed -> STOP_AFTER_SAVE"
touch $CTL/STOP_AFTER_SAVE
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 15; done
grep -E "\[save\]|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | tail -6
cd /data/home/amartyavarma/ppm
J2=$(PPM_ENV=$C/prod_ppm.env sbatch --parsable -N 16 -w $(cat $C/nodes16.txt) -t 5-12:00:00 $C/ppm.sbatch); echo "[$(pt)] resubmitted as $J2"
until grep -qE "step [0-9]+/.*|Traceback" $L/ppm-$J2.log 2>/dev/null && [ $(grep -cE "^\[.*step [0-9]+/" $L/ppm-$J2.log) -ge 2 ] || grep -q Traceback $L/ppm-$J2.log 2>/dev/null; do sleep 20; done
grep -E "\[resume\]|resumed|step [0-9]+/|Traceback" $L/ppm-$J2.log | head -6 | cut -c1-200
echo "[$(pt)] RESUME GATE DONE job $J2"

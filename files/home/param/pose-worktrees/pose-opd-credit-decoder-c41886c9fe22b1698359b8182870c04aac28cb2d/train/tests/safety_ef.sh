#!/bin/bash
# T16 part 2 (fresh run base, current code): E wall-clock cadence (save every 1 min, eval every 0.5 min) then STOP; F resume + /scratch-full
# skip path (pretend 100 TB needed): STOP file -> save SKIPPED, run continues; then scancel. Cleanup after.
set -u
C=/data/home/amartyavarma/ppm/code; L=/data/home/amartyavarma/ppm/logs; NODES=${NODES:-slinky-11,slinky-13}
E=$C/tests/safety2_ppm.env; BASE=ppm-safety-ef; CK=/scratch/amartyavarma/ppm/ckpt-test/$BASE
pt() { TZ=America/Los_Angeles date '+%T PT'; }
sub() { sbatch --parsable -N 2 -w $NODES -t 40 --export=ALL "$@" $C/ppm.sbatch; }
lsck() { for n in ${NODES//,/ }; do srun -w $n -c 1 --mem=1G -t 2 bash -c "echo \$(hostname -s): \$(ls $CK 2>/dev/null | tr '\n' ' ')"; done; }
rm -rf /data/home/amartyavarma/ppm/control/$BASE
echo "[$(pt)] E"; J=$(PPM_ENV=$E PPM_RUN=$BASE PPM_MAX_STEPS=400 PPM_SAVE_EVERY=0 PPM_FINAL_SAVE=0 PPM_SAVE_EVERY_MIN=1 PPM_EVAL_EVERY_MIN=0.5 sub); echo "E job $J"
until grep -qE "step 1/" $L/ppm-$J.log 2>/dev/null; do sleep 5; done; sleep 200; touch /data/home/amartyavarma/ppm/control/$BASE/STOP_AFTER_SAVE
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 5; done
grep -E "step 1/|\[save\] /|EVAL step|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | cut -c1-110; lsck
echo "[$(pt)] F"; J=$(PPM_ENV=$E PPM_RUN=$BASE PPM_RESUME_FROM=auto PPM_MAX_STEPS=400 PPM_SAVE_EVERY=0 PPM_FINAL_SAVE=0 PPM_TEST_SCRATCH_NEED_GB=100000 sub); echo "F job $J"
until grep -qE "resumed|Traceback" $L/ppm-$J.log 2>/dev/null; do sleep 5; done; sleep 30; touch /data/home/amartyavarma/ppm/control/$BASE/STOP_AFTER_SAVE; sleep 45; scancel $J
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 5; done
grep -E "resumed|SKIPPED|short|need|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | sort -u | cut -c1-200 | head -8; lsck
echo "[$(pt)] cleanup"; for n in ${NODES//,/ }; do srun -w $n -c 1 --mem=1G -t 2 rm -rf $CK; done; rm -rf /data/home/amartyavarma/ppm/control/$BASE; lsck
echo "[$(pt)] SAFETY EF DONE"

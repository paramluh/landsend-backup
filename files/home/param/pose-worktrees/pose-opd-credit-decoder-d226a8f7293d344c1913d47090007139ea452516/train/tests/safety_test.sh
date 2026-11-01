#!/bin/bash
# T16 end-to-end test driver (login node: submits / waits / inspects only). Phases on 2 nodes (reduced depth, one run base):
#  A fresh, 8-step horizon, save every 2, STOP_AFTER_SAVE control file at step 5 (test knob), export at 4 -> dcp-4, dcp-5 kept, exit 0
#  B resume auto (5), SAVE_NOW at 7 (test knob), final save at 8, export at 8 with an impossible floor -> REFUSED; keep dcp-7, dcp-8
#  C resume auto (8), long horizon; `scancel -b -s USR1` (= Slurm's --signal=B:USR1) once step >= 10 -> trap -> STOP_AFTER_SAVE -> save, exit 0
#  D resume auto; `scancel -s USR1` to the TASKS (python SIGUSR1 handler) once step >= +2 -> save, exit 0
# then lists the node-local dirs and deletes every test checkpoint / export / control file.
set -u
C=/data/home/amartyavarma/ppm/code; L=/data/home/amartyavarma/ppm/logs; NODES=${NODES:-slinky-11,slinky-13}
E=$C/tests/safety2_ppm.env; BASE=ppm-safety2; CK=/scratch/amartyavarma/ppm/ckpt-test/$BASE; EXR=/data/home/amartyavarma/ppm/ckpt-weka-test
pt() { TZ=America/Los_Angeles date '+%T PT'; }
sub() { sbatch --parsable -N 2 -w $NODES -t 40 --export=ALL "$@" $C/ppm.sbatch; }
lsck() { for n in ${NODES//,/ }; do srun -w $n -c 1 --mem=1G -t 2 bash -c "echo \$(hostname -s): \$(ls $CK 2>/dev/null | tr '\n' ' ')"; done; }
waitstep() { until grep -qE "step ($2)/|== exit|Traceback" $L/ppm-$1.log 2>/dev/null; do sleep 5; done; }
rm -rf /data/home/amartyavarma/ppm/control/$BASE
echo "[$(pt)] A"; J=$(PPM_ENV=$E PPM_MAX_STEPS=8 PPM_HORIZON=8 PPM_SAVE_EVERY=2 PPM_TEST_STOP_AT=5 PPM_EXPORT_EVERY=4 PPM_EXPORT_ROOT=$EXR sub --wait); echo "A job $J rc $?"
grep -E "\[save\]|\[export\]|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | cut -c1-250; lsck
echo "[$(pt)] B"; J=$(PPM_ENV=$E PPM_RESUME_FROM=auto PPM_MAX_STEPS=8 PPM_HORIZON=8 PPM_SAVE_EVERY=2 PPM_TEST_SAVENOW_AT=7 PPM_EXPORT_EVERY=8 PPM_EXPORT_ROOT=$EXR PPM_EXPORT_MIN_FREE_GB=100000 sub --wait); echo "B job $J rc $?"
grep -E "\[resume\]|resumed|\[save\]|\[export\]|\[control\]|== exit|Traceback" $L/ppm-$J.log | cut -c1-250; lsck
echo "[$(pt)] C"; J=$(PPM_ENV=$E PPM_RESUME_FROM=auto PPM_MAX_STEPS=60 PPM_HORIZON=8 PPM_SAVE_EVERY=0 PPM_FINAL_SAVE=0 sub); echo "C job $J"
waitstep $J "1[0-9]"; echo "[$(pt)] C: batch-shell USR1"; scancel -b -s USR1 $J
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 5; done
grep -E "resumed|USR1|\[save\]|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | cut -c1-250; lsck
S=$(ls -d $CK 2>/dev/null >/dev/null; grep -oE "STOPPED at step [0-9]+" $L/ppm-$J.log | grep -oE "[0-9]+$")
echo "[$(pt)] D"; J=$(PPM_ENV=$E PPM_RESUME_FROM=auto PPM_MAX_STEPS=60 PPM_HORIZON=8 PPM_SAVE_EVERY=0 PPM_FINAL_SAVE=0 sub); echo "D job $J"
waitstep $J "$((S + 2))"; echo "[$(pt)] D: task-level USR1"; scancel -s USR1 $J
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 5; done
grep -E "resumed|SIGUSR1|\[save\]|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | cut -c1-250; lsck
echo "[$(pt)] E: wall-clock cadence (save every 1 min, eval every 0.5 min), ~4 min"
J=$(PPM_ENV=$E PPM_RESUME_FROM=auto PPM_MAX_STEPS=200 PPM_HORIZON=8 PPM_SAVE_EVERY=0 PPM_FINAL_SAVE=0 PPM_SAVE_EVERY_MIN=1 PPM_EVAL_EVERY_MIN=0.5 sub); echo "E job $J"
sleep 240; touch /data/home/amartyavarma/ppm/control/$BASE/STOP_AFTER_SAVE
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 5; done
grep -E "resumed|\[save\] /|EVAL step|\[control\]|STOPPED|== exit|Traceback" $L/ppm-$J.log | cut -c1-120; lsck
echo "[$(pt)] F: /scratch-full path (pretend 100 TB needed): STOP file at step +2 -> save SKIPPED, no stop; then real stop by USR1? no: scancel after 3 more steps"
J=$(PPM_ENV=$E PPM_RESUME_FROM=auto PPM_MAX_STEPS=200 PPM_HORIZON=8 PPM_SAVE_EVERY=0 PPM_FINAL_SAVE=0 PPM_TEST_SCRATCH_NEED_GB=100000 sub); echo "F job $J"
waitstep $J "[0-9]+"; touch /data/home/amartyavarma/ppm/control/$BASE/STOP_AFTER_SAVE; sleep 60; scancel $J
while squeue -j $J -h 2>/dev/null | grep -q .; do sleep 5; done
grep -E "resumed|SKIPPED|short|\[control\]|STOPPED|== exit" $L/ppm-$J.log | sort | uniq | cut -c1-200 | head; lsck
ls -la /data/home/amartyavarma/ppm/control/$BASE; du -sh $EXR/$BASE/* 2>/dev/null
echo "[$(pt)] cleanup"; for n in ${NODES//,/ }; do srun -w $n -c 1 --mem=1G -t 2 rm -rf $CK; done; rm -rf $EXR /data/home/amartyavarma/ppm/control/$BASE; lsck; df -h /data | tail -1
echo "[$(pt)] SAFETY TEST DRIVER DONE"

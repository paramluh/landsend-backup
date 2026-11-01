#!/bin/bash
# Overlap-step SAFETY TEST on a spare (slinky-4), run from a CPU job there (it submits a dummy "trainer" job and pokes it):
#  D = dummy trainer job: ONE homogeneous component like ppm811.sbatch, main step `srun --mem=<all> --kill-on-bad-exit=1` writing trainer-format
#      'step N/M' lines. Then, while D's main step runs:
#  1 cgroup layout + job memory headroom as seen from an overlap step (what npl5_roll.py checks)
#  2 an overlap step that exceeds its own --mem (step-level OOM)            -> only that step dies; D's main step keeps running
#  3 the supervisor (npl5_roll_super.sbatch, test mode) attaches its npl5-roll step; STOP_ROLL -> it scancels ONLY <D>.<its step>, exits
#  4 SIGKILL of an overlap srun CLIENT (not the step)                         -> D's main step keeps running
#  After each: D's main step must still be RUNNING and its log still advancing. -> W/safety/RESULT.txt
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; T=$W/safety/run-$(date +%s); mkdir -p $T
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "== $(pt) $*" | tee -a $T/RESULT.txt; }
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]*'); do unset $v; done
NODE=${NODE:-slinky-18}
D=$(sbatch --parsable -p all -w $NODE -x slinky-9 -N1 -n1 -c1 --mem=3G --gres=none -t 0:40:00 -J dummy811 -o $T/dummy-%j.out \
  --wrap "srun --mem=3G --kill-on-bad-exit=1 bash -c 'i=0; while [ \$i -lt 2000 ]; do echo \"[x] step \$((17850+i))/19400 68.0s (dummy)\" >> $T/trainer.log; i=\$((i+1)); sleep 1; done'")
echo $D > $T/ctl; say "dummy trainer job $D"
steps() { sacct -j $D -n -P -s R -o JobID,JobName 2>/dev/null; }
until steps | grep -q "^$D\.0|"; do sleep 2; done; sleep 3
alive() { local a b; a=$(wc -l < $T/trainer.log); sleep 4; b=$(wc -l < $T/trainer.log); steps | grep -q "^$D\.0|" && [ $b -gt $a ] && echo ALIVE || echo DEAD; }
say "0 baseline: main step $(alive)"
srun --jobid $D --overlap -N1 -n1 -c1 --mem=200M --cpu-bind=none --input=none -J cgprobe bash -c \
  'c=$(cut -d: -f3 /proc/self/cgroup); j=/sys/fs/cgroup${c%%/step_*}; echo "self $c"; echo "job dir $j max $(cat $j/memory.max) current $(cat $j/memory.current)"; s=/sys/fs/cgroup${c%/user*}; s=${s%/task_*}; echo "step mem.max $(cat /sys/fs/cgroup$(echo $c | sed "s|/user.*||")/memory.max 2>/dev/null)"' >> $T/RESULT.txt 2>&1
say "1 cgroup probe done; main step $(alive)"
srun --jobid $D --overlap -N1 -n1 -c1 --mem=300M --cpu-bind=none --input=none -J oomtest /data/home/amartyavarma/venvs/qwen-s2/bin/python -c \
  "x=bytearray(1200<<20); print('allocated 1.2 GB in a 300M step: NOT confined', flush=True)" >> $T/RESULT.txt 2>&1; say "2 step-OOM test rc $? (137/OOM expected); main step $(alive)"
mkdir -p $T/state; NPL5_ROLL_DIR=$T/roll
sbatch --parsable -p all -w $NODE -x slinky-9 -N1 -n1 -c1 --mem=1G --gres=none -t 0:20:00 -J npl5-roll-super-test -o $T/super-%j.out \
  --export=ALL,END=19400,STOP_FILE=$T/STOP_ROLL,CTL_JOB_FILE=$T/ctl,TRAIN_LOG_PAT=$T/trainer.log,NPL5_ROLL_DIR=$T/roll,NPL5_KIT_STATE=$T/kit,ROLL_MEM=3G,POLL=5,SUPER_POLL=10,ROLL_ARGS="--phys slinky-0 --key slinky-0 --root $W/prov/slinky-0 --audit13 --ckpt-dir $T/ckpt --hold-dir $T/hold --pin-dir $T/pin --log $T/trainer.log --hf-rate 5 --no-startup-hash" \
  $W/bin/npl5_roll_super.sbatch > $T/super.jobid
SJ=$(cat $T/super.jobid); say "3 supervisor test job $SJ (test mode: CTL_JOB_FILE / TRAIN_LOG_PAT / NPL5_ROLL_DIR point at this test)"
until steps | grep -q "|npl5-roll$"; do sleep 3; done
say "3 npl5-roll step attached: $(steps | grep npl5-roll); main step $(alive)"
sleep 60; say "3 daemon after 60 s: $(tail -n 1 $T/roll/slinky-0.log 2>/dev/null | cut -c1-200)"
say "3 status JSON (real job cgroup headroom read inside the step): $(python3 -c "import json;d=json.load(open('$T/roll/slinky-0.json'));print({k: d.get(k) for k in ('heartbeat','job_mem_headroom_GB','ahead_steps','alerts','phase')})")"
say "3 supervisor heartbeat: $(cat $T/roll/SUPER.json 2>/dev/null)"
S1=$(steps | grep "|npl5-roll$" | cut -d'|' -f1 | tail -1)
srun --jobid $D --overlap -N1 -n1 -c1 --mem=100M --cpu-bind=none --input=none -J crashtest /data/home/amartyavarma/venvs/qwen-s2/bin/python -c \
  "import os,signal; pat='npl5_'+'roll.py daemon'; me=os.getpid(); ps=[int(p) for p in os.listdir('/proc') if p.isdigit() and int(p)!=me and pat in open(f'/proc/{p}/cmdline','rb').read().replace(b'\\0',b' ').decode('utf-8','replace')]; [os.kill(p, signal.SIGKILL) for p in ps]; print('killed daemon pids', ps)" >> $T/RESULT.txt 2>&1
for i in $(seq 18); do [ -f $T/roll/STEP_DIED ] && [ -n "$(steps | grep '|npl5-roll$' | cut -d'|' -f1 | grep -v "^$S1$")" ] && break; sleep 5; done
say "3b daemon crash (SIGKILL inside step $S1): STEP_DIED marker $([ -f $T/roll/STEP_DIED ] && cat $T/roll/STEP_DIED || echo MISSING); re-attached as $(steps | grep '|npl5-roll$' | cut -d'|' -f1 | grep -v "^$S1$" | tr '\n' ' '); main step $(alive)"
touch $T/STOP_ROLL; say "3 STOP file touched"
until ! squeue -h -j $SJ | grep -q .; do sleep 5; done
say "3 supervisor exited: $(grep -h 'scancel own step\|supervisor exit' $T/super-$SJ.out | tr '\n' ' '); npl5-roll steps left: $(steps | grep -c npl5-roll); main step $(alive)"
srun --jobid $D --overlap -N1 -n1 -c1 --mem=100M --cpu-bind=none --input=none -J clientkill sleep 300 & CP=$!
sleep 8; kill -9 $CP; sleep 5
say "4 SIGKILL of an overlap srun client (step clientkill): main step $(alive); steps now: $(steps | tr '\n' ' ')"
srun --jobid $D --overlap -N1 -n1 -c1 --mem=100M --cpu-bind=none --input=none -J cleanup bash -c 'du -sh /scratch/amartyavarma/npl-append/npl5 2>/dev/null; rm -rf /scratch/amartyavarma/npl-append/npl5; rmdir /scratch/amartyavarma/npl-append 2>/dev/null' >> $T/RESULT.txt 2>&1
say "cleanup of the canonical npl5 dir on $NODE done; main step $(alive)"
scancel $D; say "dummy job $D cancelled (test end)"

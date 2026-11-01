#!/bin/bash
#SBATCH -J npl5-roll-super -p all -w slinky-4 -x slinky-9 -N1 -n1 -c1 --mem=1G --gres=none -t 4-00:00:00 --dependency=singleton
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/roll-super-%j.out
# npl5 ROLLING FETCHER SUPERVISOR v2 (audit round 1, S11). Start once, at launch, from C's switch script, right after the trainer job:
#   sbatch --export=ALL,END=<PPM_MAX_STEPS>,AHEAD=100,LOW=30 /data/home/amartyavarma/ppm/pose5/npl5/bin/npl5_roll_super.sbatch
# slinky-4, 1 CPU, 1G, nice; --dependency=singleton (one supervisor per user + name). The trainer job (ppm811.sbatch: ONE homogeneous component,
# not a het job; all 64 CPUs of each host) leaves no room for another job, so the per-host daemons (bin/npl5_roll.py v2; source = the HF bucket)
# run as ONE light `srun --overlap` step 'npl5-roll' inside the CURRENT trainer job (control/ppm-v1/CURRENT_JOB_811): -c1 --cpu-bind=none
# --mem=8G (its own step cgroup: an OOM there kills only the fetcher; tested), nice 19, ionice be/7, --kill-on-bad-exit=0.
#  - heartbeat: W/roll/SUPER.json every <= 30 s (job attached, step state, restarts); C alerts when it is older than 5 min
#  - the step ends while the trainer job is still RUNNING (a daemon crash / OOM) -> W/roll/STEP_DIED (+ log) and it is re-attached after 20 s;
#    the marker is cleared when a re-attached step has run 10 min
#  - trainer job ends (watchdog relaunch / run end) -> the step ends with it -> re-attach to the next CURRENT_JOB_811 job
#  - stop: touch W/state/STOP_ROLL; only a file NEWER than this supervisor's start counts (an older one is renamed .stale-<time> at start);
#    it scancels ONLY its own step (<job>.<step>, never the job; never pkill), waits <= 2 min for the srun client, then kills that client only
set -uo pipefail
W=/data/home/amartyavarma/ppm/pose5/npl5; CTL=/data/home/amartyavarma/ppm/control/ppm-v1
: "${END:?END=<PPM_MAX_STEPS>}"
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]*'); do unset $v; done      # (this job's own allocation must not leak into srun --jobid)
CTL_JOB=${CTL_JOB_FILE:-$CTL/CURRENT_JOB_811}; STOPF=${STOP_FILE:-$W/state/STOP_ROLL}; LOGPAT=${TRAIN_LOG_PAT:-/data/home/amartyavarma/ppm/logs/ppm-%s.log}
RD=${NPL5_ROLL_DIR:-$W/roll}; mkdir -p $RD; T0=$(date +%s); NRE=0; J=""; SP=""; STEP_T0=0
[ -f $STOPF ] && { mv $STOPF $STOPF.stale-$(date +%s); echo "== $(pt) stale STOP file (older than this supervisor) renamed .stale"; }
hb() { printf '{"time": "%s", "epoch": %s, "host": "%s", "job": "%s", "step_running": %s, "restarts": %s, "end": %s}\n' "$(pt)" "$(date +%s)" \
  "$(hostname -s)" "${J:-}" "$([ -n "$SP" ] && kill -0 $SP 2>/dev/null && echo true || echo false)" $NRE $END > $RD/SUPER.json.$$; mv -f $RD/SUPER.json.$$ $RD/SUPER.json; }
stopped() { [ -f $STOPF ] && [ $(stat -c %Y $STOPF) -gt $T0 ]; }
own_step() { sacct -j $1 -n -P -o JobID,JobName,State 2>/dev/null | awk -F'|' '$2=="npl5-roll" && $3=="RUNNING"{print $1}'; }
echo "== $(pt) npl5 roll supervisor v2 on $(hostname -s): END $END, ahead ${AHEAD:-100} (kit/state/AHEAD-<host> overrides), low water ${LOW:-30}"
while ! stopped; do
  hb
  J=$(awk '{print $1}' $CTL_JOB 2>/dev/null)
  if [ -n "$J" ] && [ "$(squeue -h -j $J -t R -o %T 2>/dev/null)" = RUNNING ]; then
    grep -aq "done $END steps" $(printf "$LOGPAT" $J) 2>/dev/null && { echo "== $(pt) trainer $J done $END steps: supervisor exits"; break; }
    H=$(squeue -h -j $J -o %N); N=$(scontrol show hostnames "$H" | wc -l)
    echo "== $(pt) attaching the npl5 roll daemons to trainer job $J ($N hosts: $H)"
    nice -n19 srun --jobid $J --overlap -N$N --ntasks-per-node=1 -c1 --cpu-bind=none --mem=${ROLL_MEM:-8G} --kill-on-bad-exit=0 --input=none -J npl5-roll \
      -o $W/logs/roll-$J-%N.out bash -c "P=\$(hostname -s); A=AHEAD_\${P//-/_}; exec nice -n19 ionice -c2 -n7 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I \
        $W/bin/npl5_roll.py daemon --end $END --ahead \${!A:-${AHEAD:-100}} --low-water ${LOW:-30} --hf-rate ${HF_RATE:-20} --threads 8 --poll ${POLL:-120} ${ROLL_ARGS:-}" &
    SP=$!; STEP_T0=$(date +%s)
    while kill -0 $SP 2>/dev/null; do
      hb
      if stopped; then
        s=$(own_step $J); [ -n "$s" ] && { echo "== $(pt) STOP: scancel own step $s"; scancel $s; }
        for i in $(seq 12); do kill -0 $SP 2>/dev/null || break; sleep 10; done
        kill -0 $SP 2>/dev/null && { echo "== $(pt) srun client $SP still alive 2 min after the step scancel: killing the client only"; kill $SP; }
        break
      fi
      [ -f $RD/STEP_DIED ] && [ $(( $(date +%s) - STEP_T0 )) -gt 600 ] && rm -f "${RD:?}/STEP_DIED"
      sleep 20
    done
    wait $SP 2>/dev/null; rc=$?; SP=""
    stopped && break
    if [ "$(squeue -h -j $J -t R -o %T 2>/dev/null)" = RUNNING ]; then
      NRE=$((NRE + 1)); echo "$(pt) roll step in RUNNING job $J ended rc $rc (restart $NRE)" > $RD/STEP_DIED
      echo "== $(pt) ALERT: npl5-roll step in still-RUNNING job $J ended (rc $rc): re-attaching in 20 s (restart $NRE)"
    else
      echo "== $(pt) roll step ended with trainer job $J (rc $rc)"
    fi
  fi
  sleep ${SUPER_POLL:-20}
done
hb; echo "== $(pt) supervisor exit"

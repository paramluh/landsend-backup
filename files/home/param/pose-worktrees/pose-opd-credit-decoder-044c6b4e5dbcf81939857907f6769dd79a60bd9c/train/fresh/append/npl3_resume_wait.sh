#!/bin/bash
# 10-23 DELIVERY-3 RESUME WAITER (control/OPS_PLAYBOOK.md "Delivery-3 append"). Runs in npl3_resume_wait.sbatch (CPU-only, slinky-15). Every POLL s it checks 4 gates
# and, when ALL hold, launches the resume of job 27204's 8,1,1 run from its final save dcp-17641 onto plans [plan 1, npl2, npl3] with the SAME
# launcher (ppm811.sbatch, 16 hosts nodes16_s7.txt) + a watchdog (ppm811f_watchdog.sbatch, TARGET = the new max) + an eval logger
# (eval_log_append.sbatch -> fresh/EVALS_append3.md), like 27204 / 27205 / 27206.
#  (a) plan 3 verified: the npl3 ingest finished READY (fetch + verify 0 mismatches, scan + plan, check, dry run vs dcp-17641, guard, unit
#      test all3, cfg diff), PPM_MAX_STEPS filled in prod_811_fresh_append3.env == the dry run's; at launch the plan files on every host are
#      re-hashed against the dry run's sha256.
#  (b) both Aloe FTs finished: the two jobs in aloe-v2pure-17641/control/FTB_LAUNCHED are gone from the queue AND no aloev2p* job is queued /
#      running (a relaunched FT gates too). An FT that ended not COMPLETED: wait GRACE_MIN for a relaunch, then go (nodes never idle) + note.
#      Fallback when the marker is missing / unreadable: no aloev2p* job and no GPU job on the 16 hosts for 3 polls in a row.
#  (c) node-local dcp-17641 COMPLETE (marker 17641, 20 files, train_state step 17641) on all 16 hosts.
#  (d) /scratch free >= MIN_FREE_GB on every host (npl3/ctl/MIN_FREE_GB, default 122 = 30 floor + 46 save + 46 HF hold; coordinator rule).
#      While (d) blocks: npl3/ctl/DISK_BLOCKED holds the per-host numbers.
# Also (coordinator OK): once an FT's export-200 is VERIFIED on private HF (its hf_final job COMPLETED), its node-local copy on slinky-0/1/2 is
# deleted (npl3/ctl/ft_export_rm.sh, logged to setup-logs/hfb/DELETED-1021-hfb-hold.log).
# Never launches twice: refuses if any ppm811 job is queued / running (someone else launched) and takes npl3/ctl/LAUNCH.lock. Stop: touch
# npl3/ctl/WAIT_OFF. Status line: npl3/ctl/WAIT_STATUS. Log: npl3/logs/resume_wait.log.
set -u
C=/data/home/amartyavarma/ppm/code-811; A=$C/fresh/append; W=$A/npl3; K=$W/ctl; CTL=/data/home/amartyavarma/ppm/control/ppm-v1
ENVF=$C/prod_811_fresh_append3.env; NODEF=$C/nodes16_s7.txt; NODES=$(cat $NODEF); NH=$(echo $NODES | tr , '\n' | wc -l)
FTL=/data/home/amartyavarma/ppm/aloe-v2pure-17641/control/FTB_LAUNCHED
RS=17641; RES=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$RS
HOLD=${HOLD:-27621}; POLL=${POLL:-60}; GRACE_MIN=${GRACE_MIN:-20}
LOG=$W/logs/resume_wait.log; MD=/data/home/amartyavarma/ppm/control/overnight_monitor_811f.md
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "[$(pt)] $*" | tee -a $LOG; }
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
csb() { env $UNSET sbatch "$@"; }
minfree() { local v; v=$(cat $K/MIN_FREE_GB 2>/dev/null | tr -dc 0-9); echo ${v:-122}; }
# one command on all 16 hosts: inside the CPU-only holder while it lives, else a tiny job of its own
on_hosts() {
  if squeue -h -j $HOLD -t R 2>/dev/null | grep -q .; then
    timeout 300 env $UNSET srun --jobid $HOLD --overlap --cpu-bind=none -N$NH -n$NH --ntasks-per-node=1 -w $NODES -c1 --mem=1G "$@" 2>/dev/null
  else
    timeout 600 env $UNSET srun -p all -N$NH -n$NH --ntasks-per-node=1 -w $NODES -c1 --mem=1G --gres=none -t 10 --immediate=300 -J npl3-gate "$@" 2>/dev/null
  fi
}
on_host() { local h=$1; shift
  if squeue -h -j $HOLD -t R 2>/dev/null | grep -q .; then
    timeout 600 env $UNSET srun --jobid $HOLD --overlap --cpu-bind=none -N1 -n1 -w $h -c1 --mem=2G "$@"
  else
    timeout 900 env $UNSET srun -p all -N1 -n1 -w $h -c1 --mem=2G --gres=none -t 10 --immediate=300 -J npl3-ftrm "$@"
  fi
}

gate_a() {   # -> 0 ok; echo reason
  local s; for s in fetch verify plan check dryrun_host dryrun_reduce guard unittest cfgdiff; do [ -f $W/state/$s.ok ] || { echo "ingest step $s not done"; return 1; }; done
  grep -q "READY: env $ENVF" $W/logs/ingest.log 2>/dev/null || { echo "ingest READY line missing"; return 1; }
  python3 - "$W" "$ENVF" <<'EOF' || return 1
import json, glob, sys, re
W, envf = sys.argv[1], sys.argv[2]
R, G = json.load(open(f"{W}/dryrun/RESULT.json")), json.load(open(f"{W}/dryrun/GUARD.json"))
V = [json.load(open(f)) for f in glob.glob(f"{W}/verify/*.json")]
m = [l.split("=", 1)[1].strip() for l in open(envf) if l.startswith("PPM_MAX_STEPS=")][-1]
bad = []
if G.get("guard") != "ok" or G.get("pose_append_at") != [17352, 17641]: bad.append(f"guard {G.get('guard')} {G.get('pose_append_at')}")
if not R.get("resume_is_final_save") or R.get("resume_ckpt") != [17641]: bad.append(f"resume ckpt {R.get('resume_ckpt')}")
if not m.isdigit() or int(m) != R["max_steps"] or R["ends"][:2] != [17352, 17641] or R["max_steps"] <= 17641: bad.append(f"max_steps env {m} vs {R['max_steps']} ends {R['ends']}")
if len(V) != 16 or any(v["hash_mismatch"] or v["missing_or_wrong_size"] for v in V): bad.append(f"verify {len(V)} hosts / mismatches")
print("; ".join(bad) if bad else "ok"); sys.exit(1 if bad else 0)
EOF
}
FTDONE_AT=""; FBK=0
gate_b() {
  local ids q st bad=""
  q=$(squeue -h -u amartyavarma -o "%i %j %T" | awk '$2 ~ /^aloev2p[0-9]+$/')
  [ -z "$q" ] || { echo "FT running/queued: $(echo $q | tr '\n' ' ')"; FBK=0; return 1; }
  ids=$(cat $FTL 2>/dev/null | tr -dc '0-9 ')
  if [ "$(echo $ids | wc -w)" != 2 ]; then    # marker missing: idle-node fallback (no aloev2p job, no GPU job on the 16 hosts, 3 polls in a row)
    local g; g=$(squeue -h -o "%i %b %N" -t R,PD,CG | grep -i gpu | while read -r i b n; do for h in $(scontrol show hostnames $n 2>/dev/null); do echo ",$NODES," | grep -q ",$h," && echo $i; done; done | sort -u)
    [ -z "$g" ] && FBK=$((FBK + 1)) || FBK=0
    [ $FBK -ge 3 ] && { echo "ok (fallback: FTB_LAUNCHED unreadable, hosts GPU-idle 3 polls)"; return 0; }
    echo "FTB_LAUNCHED unreadable; GPU jobs on hosts: ${g:-none} (idle polls $FBK/3)"; return 1
  fi
  for i in $ids; do
    squeue -h -j $i 2>/dev/null | grep -q . && { echo "FT job $i still in queue"; return 1; }
    st=$(sacct -n -X -j $i -o State%20 2>/dev/null | head -1 | tr -d ' ')
    [ "$st" = COMPLETED ] || bad="$bad $i:$st"
  done
  if [ -n "$bad" ]; then
    [ -n "$FTDONE_AT" ] || { FTDONE_AT=$(date +%s); say "NOTE: FT job(s) ended NOT COMPLETED:$bad; waiting $GRACE_MIN min for a relaunch (an aloev2p job gates), then launching"; echo "- $(TZ=America/Los_Angeles date '+%a %m-%d %H:%M PT'): [npl3 waiter] FT ended not COMPLETED:$bad; main resume waits $GRACE_MIN min for an FT relaunch, then launches" >> $MD; }
    [ $(( $(date +%s) - FTDONE_AT )) -ge $((GRACE_MIN * 60)) ] || { echo "FT ended not COMPLETED:$bad (grace)"; return 1; }
  fi
  echo "ok ($ids gone${bad:+; not completed:$bad})"; return 0
}
SCAN=""
gate_cd() {   # host scan -> 0 ok
  local mf; mf=$(minfree)
  SCAN=$(on_hosts bash -c "h=\$(hostname -s); c=0; [ \"\$(cat $RES/COMPLETE 2>/dev/null)\" = $RS ] && [ \$(ls $RES | wc -l) = 20 ] && grep -q '\"step\": $RS' $RES/train_state.json && c=1; p=0; [ -f /scratch/amartyavarma/ppm-fresh-pose-npl3/PLAN.json ] && [ -f /scratch/amartyavarma/ppm-fresh-pose-npl3/PLAN.npz ] && p=1; echo \"\$h ck=\$c plan=\$p free=\$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)\"" | sort -V)
  [ "$(echo "$SCAN" | grep -c ' ck=')" = $NH ] || { echo "host scan incomplete ($(echo "$SCAN" | grep -c ' ck=')/$NH)"; return 1; }
  local nc np low
  nc=$(echo "$SCAN" | grep -c ' ck=1'); np=$(echo "$SCAN" | grep -c ' plan=1')
  low=$(echo "$SCAN" | awk -v m=$mf '{split($4,a,"="); if (a[2]+0 < m) printf "%s=%s ", $1, a[2]}')
  if [ -n "$low" ]; then echo "$(pt) /scratch free < $mf GB on: $low| all: $(echo "$SCAN" | awk '{split($4,a,"="); printf "%s=%s ", $1, a[2]}')" > $K/DISK_BLOCKED; else rm -f $K/DISK_BLOCKED; fi
  [ $nc = $NH ] || { echo "dcp-$RS COMPLETE on $nc/$NH: $(echo "$SCAN" | grep ' ck=0' | awk '{print $1}' | tr '\n' ' ')"; return 1; }
  [ $np = $NH ] || { echo "npl3 plan files on $np/$NH"; return 1; }
  [ -z "$low" ] || { echo "disk < $mf GB: $low"; return 1; }
  echo "ok (min free $(echo "$SCAN" | awk '{split($4,a,"="); print a[2]}' | sort -n | head -1) GB >= $mf)"; return 0
}
ft_exports() {   # FT final export-200 off slinky-0/1/2 once VERIFIED on HF (once per host; ft_export_rm.sh checks everything again)
  local r hosts hj st h
  for r in "aloe-v2pure-17352 slinky-1" "aloe-v2pure-17641 slinky-0,slinky-2"; do
    set -- $r; hosts=$2; r=$1
    hj=$(python3 -c "import json;print(json.load(open('/data/home/amartyavarma/ppm/$r/BACKUP_JOB'))['backup_job'])" 2>/dev/null) || continue
    st=$(sacct -n -X -j $hj -o State%20 2>/dev/null | head -1 | tr -d ' ')
    [ "$st" = COMPLETED ] || continue
    for h in ${hosts//,/ }; do
      [ -f $K/ftrm-$r-$h.done ] && continue
      on_host $h nice -n19 $K/ft_export_rm.sh $r $hj >> $LOG 2>&1; local rc=$?
      if [ $rc = 0 ]; then touch $K/ftrm-$r-$h.done; say "FT $r export-200 on $h: removed / absent after HF VERIFY PASS (hf job $hj)"
      else say "FT $r export-200 on $h KEPT (rc $rc; see above)"; [ $rc = 2 ] && touch $K/ftrm-$r-$h.done; fi
    done
  done
}

say "waiter start (job ${SLURM_JOB_ID:-?}): resume dcp-$RS -> plans [plan 1, npl2, npl3], env $ENVF, hosts $NODES; holder $HOLD; MIN_FREE_GB $(minfree)"
LAST=""
while :; do
  [ -f $K/WAIT_OFF ] && { say "WAIT_OFF present: exiting without launch"; exit 0; }
  ft_exports
  ra=$(gate_a); a=$?; rb=$(gate_b); b=$?; rcd=$(gate_cd); cd=$?
  ST="a(plan3)=${ra} | b(FTs)=${rb} | c+d(ckpt,disk>=$(minfree))=${rcd}"
  echo "$(pt) $ST" > $K/WAIT_STATUS
  [ "$ST" != "$LAST" ] && say "$ST"; LAST=$ST
  if [ $a = 0 ] && [ $b = 0 ] && [ $cd = 0 ]; then break; fi
  sleep $POLL
done
# ---- launch (once)
mkdir $K/LAUNCH.lock 2>/dev/null || { say "LAUNCH.lock exists: another waiter launched; exiting"; exit 0; }
O=$(squeue -h -u amartyavarma -n ppm811 -t R,PD,CF -o %i | head -1)
[ -z "$O" ] || { say "ABORT: a ppm811 job ($O) is already queued / running: not launching a second one"; exit 1; }
say "ALL GATES OK; final plan-file hash check on every host"
HC=$(python3 -c "
import json,glob
for f in sorted(glob.glob('$W/dryrun/slinky-*.json')):
    r=json.load(open(f)); print(r['host'], r['plan_sha256']['npl3/PLAN.json'], r['plan_sha256']['npl3/PLAN.npz'], r['plan_sha256']['npl2/PLAN.npz'])")
HS=$(on_hosts bash -c 'echo "$(hostname -s) $(sha256sum /scratch/amartyavarma/ppm-fresh-pose-npl3/PLAN.json | cut -c1-64) $(sha256sum /scratch/amartyavarma/ppm-fresh-pose-npl3/PLAN.npz | cut -c1-64) $(sha256sum /scratch/amartyavarma/ppm-fresh-pose-npl2/PLAN.npz | cut -c1-64)"' | sort -V)
if [ "$(echo "$HC" | sort -V)" != "$HS" ]; then say "ABORT: plan files differ from the dry run's sha256:"; diff <(echo "$HC" | sort -V) <(echo "$HS") | tee -a $LOG; rmdir $K/LAUNCH.lock; exit 1; fi
say "plan files == dry run on all $NH hosts"
M=$(grep '^PPM_MAX_STEPS=' $ENVF | tail -1 | cut -d= -f2)
touch $K/RELEASE
for i in $(seq 1 18); do squeue -h -j $HOLD 2>/dev/null | grep -q . || break; sleep 10; done
squeue -h -j $HOLD 2>/dev/null | grep -q . && { say "holder $HOLD still up after 3 min: scancel (CPU-only holder, mine)"; scancel $HOLD; sleep 10; }
for j in $(squeue -h -u amartyavarma -n ppm811f-watch,ppm811f-evallog -o %i); do say "stale 8,1,1 watcher job $j (old target): scancel"; scancel $j; done
J=$(PPM_ENV=$ENVF PPM_RESUME_FROM=$RES csb --parsable -N $NH -w $NODES $C/ppm811.sbatch)
[[ "$J" =~ ^[0-9]+$ ]] || { say "ABORT: sbatch failed: $J"; exit 1; }
echo $J > $CTL/CURRENT_JOB_811
say "TRAIN job $J (resume $RES, env $ENVF, PPM_MAX_STEPS $M) on $NODES"
WD=$(ENVF=$ENVF TARGET=$M MIN_STEP=$RS csb --parsable $C/ppm811f_watchdog.sbatch); say "WATCHDOG $WD (TARGET $M, MIN_STEP $RS)"
EL=$(MIN_JOB=$J END=$M EVAL_OUT=EVALS_append3.md MIN_STEP=$RS csb --parsable $A/eval_log_append.sbatch); say "EVALLOG $EL -> fresh/EVALS_append3.md"
echo "$J $WD $EL $M" > $A/LAUNCHED3.txt
echo "- $(TZ=America/Los_Angeles date '+%a %m-%d %H:%M PT'): [npl3 waiter] delivery-3 resume LAUNCHED: train $J from dcp-$RS (plans plan1+npl2+npl3, max $M), watchdog $WD, eval log $EL" >> $MD
say "LAUNCHED: $J $WD $EL $M (fresh/append/LAUNCHED3.txt)"

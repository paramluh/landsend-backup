#!/bin/bash
# pose5: SAFE SWAP of the running chained hold (v1 job, e.g. 31103: kit/hold_chain.sh) v3 (kit/hold_chain_v3.sh) for kit/hold_chain_v4.sh (audit 4 R4-4/5/6).
#  check (default): is the v1 loop SLEEPING between polls? (its bash's only child = `sleep <POLL>`; no srun step, no python, no export_drop)
#  --go: re-check; if sleeping -> scancel v1, wait until gone, submit hold_chain_v2.sbatch with the SAME env (TARGET / PIN_STEP; same state dir
#        kit/state/hold_chain: held-/released-/unheld- carry over), require a numeric job id, rewrite HC= in kit/state/LAUNCHED_RUN (the
#        watchdog's HC-running alert reads it each poll), reset the 16 future-dated alert-disk-* markers (suppression until the fix), and wait
#        for v2's first poll in its log. Not sleeping -> retry every 3 s for up to 60 s, else ABORT (nothing changed).
# usage: bash swap_hold_chain_v2.sh [--go] [OLD_JOB]
set -u
K=/data/home/amartyavarma/ppm/pose5/kit; ST=$K/state/hold_chain; LR=$K/state/LAUNCHED_RUN
GO=0; [ "${1:-}" = --go ] && { GO=1; shift; }
OLD=${1:-$(grep -oE '^HC=[0-9]+' $LR | cut -d= -f2)}
M=$(grep -oE '^M=[0-9]+' $LR | cut -d= -f2); PIN=17850
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
check() {   # -> 0 if the v1 loop is sleeping between polls; prints the process tree of the job
  squeue -h -j "$OLD" -t R | grep -q . || { echo "[$(pt)] job $OLD not RUNNING"; return 2; }
  local T
  T=$(timeout 60 env $UNSET srun --jobid="$OLD" --overlap -N1 -n1 -c1 --mem=64M --cpu-bind=none ps -u amartyavarma -o pid=,ppid=,etime=,args= 2>/dev/null)
  local B; B=$(echo "$T" | awk '/bash .*kit\/hold_chain_v3\.sh/ && !/srun/ {print $1; exit}')
  [ -n "$B" ] || { echo "[$(pt)] hold_chain_v3.sh bash not found in job $OLD"; echo "$T" | head -20; return 3; }
  local DESC; DESC=$(echo "$T" | awk -v b=$B '{par[$1]=$2; line[$1]=$0} END {for (p in par) {q=p; while (q in par && q != b && q > 1) q=par[q]; if (q == b && p != b) print line[p]}}')
  echo "[$(pt)] job $OLD: hold_chain_v3.sh bash pid $B; ALL its descendants:"; echo "${DESC:-  (none)}" | sed 's/^/    /'
  local KIDS; KIDS=$(echo "$T" | awk -v b=$B '$2==b')
  [ "$(echo "$DESC" | grep -c .)" = 1 ] && echo "$KIDS" | grep -qE ' sleep [0-9]+$' && { echo "    -> SLEEPING between polls, nothing else in flight (safe)"; return 0; }
  echo "    -> NOT between polls"; return 1
}
check; RC=$?
[ $GO = 1 ] || exit $RC
for i in $(seq 1 20); do [ $RC = 0 ] && break; sleep 3; check; RC=$?; done
[ $RC = 0 ] || { echo "[$(pt)] ABORT: v1 never seen sleeping between polls in 60 s; nothing changed"; exit 1; }
scancel "$OLD"; for i in $(seq 1 30); do squeue -h -j "$OLD" | grep -q . || break; sleep 2; done
squeue -h -j "$OLD" | grep -q . && { echo "[$(pt)] ABORT: $OLD still in squeue after scancel"; exit 1; }
NEW=$(env $UNSET TARGET=$M PIN_STEP=$PIN sbatch --parsable $K/hold_chain_v4.sbatch)
[[ "$NEW" =~ ^[0-9]+$ ]] || { echo "[$(pt)] ALERT: v4 submit returned '$NEW' (v1 $OLD is cancelled: KEEP=1 net DOWN): resubmit by hand: TARGET=$M PIN_STEP=$PIN sbatch $K/hold_chain_v4.sbatch"; exit 1; }
python3 - "$LR" "$NEW" <<'EOF'
import os, re, sys
p, new = sys.argv[1], sys.argv[2]; s = open(p).read(); s2 = re.sub(r"^HC=\d+$", f"HC={new}", s, flags=re.M)
open(p + ".new", "w").write(s2); os.replace(p + ".new", p)
EOF
for f in $ST/alert-disk-slinky-*; do [ -f "$f" ] && touch -d '-2 hours' "$f"; done
echo "[$(pt)] swapped: v1 $OLD cancelled between polls -> v4 job $NEW (TARGET=$M PIN_STEP=$PIN, state dir kit/state/hold_chain); LAUNCHED_RUN HC=$NEW; alert-disk markers reset"
for i in $(seq 1 60); do squeue -h -j "$NEW" -t R | grep -q . && grep -q 'armed' $K/logs/hold_chain-$NEW.log 2>/dev/null && break; sleep 2; done
sleep 20; echo "--- v2 log:"; tail -5 $K/logs/hold_chain-$NEW.log 2>/dev/null; echo "--- chain log tail:"; tail -3 $ST/hold_chain.log

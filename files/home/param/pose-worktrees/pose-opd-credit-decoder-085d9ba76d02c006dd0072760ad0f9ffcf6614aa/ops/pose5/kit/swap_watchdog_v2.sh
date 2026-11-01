#!/bin/bash
# pose5: SAFE SWAP of the running 8,1,1 watchdog (v1 job, e.g. 31102: kit/ppm811s2p_watchdog.sh) for kit/ppm811s2p_watchdog_v2.sh (main OK
# 10-27 ~14:05). Same pattern as swap_hold_chain_v2.sh:
#  check (default): v1's bash has ONLY `sleep <POLL>` as descendant (no scan / restage / sbatch in flight); prints its launch env (from
#                   /proc/<pid>/environ: ENVF NODEF TARGET MIN_STEP RESUME_PIN PIN_STEP CODE) = what v2 is submitted with
#  --go: re-check (retry 3 s, up to 60 s) -> scancel v1 -> submit ppm811s2p_watchdog_v2.sbatch with that SAME env (same state dir
#        control/ppm-v1/watchdog_pose5_state: relaunch history / caps carry over) -> numeric id required -> WD= in kit/state/LAUNCHED_RUN
# usage: bash swap_watchdog_v2.sh [--go] [OLD_JOB]
set -u
K=/data/home/amartyavarma/ppm/pose5/kit; LR=$K/state/LAUNCHED_RUN
GO=0; [ "${1:-}" = --go ] && { GO=1; shift; }
OLD=${1:-$(grep -oE '^WD=[0-9]+' $LR | cut -d= -f2)}
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
WENV=""
check() {
  squeue -h -j "$OLD" -t R | grep -q . || { echo "[$(pt)] job $OLD not RUNNING"; return 2; }
  local T B DESC
  T=$(timeout 60 env $UNSET srun --jobid="$OLD" --overlap -N1 -n1 -c1 --mem=64M --cpu-bind=none bash -c \
      'ps -u amartyavarma -o pid=,ppid=,etime=,args=; for p in $(pgrep -u amartyavarma -f "kit/ppm811s2p_watchdog.sh"); do echo "ENV $p $(tr "\0" "\n" < /proc/$p/environ | grep -E "^(ENVF|NODEF|TARGET|MIN_STEP|RESUME_PIN|PIN_STEP|CODE)=" | tr "\n" " ")"; done' 2>/dev/null)
  B=$(echo "$T" | awk '/bash .*kit\/ppm811s2p_watchdog\.sh/ && !/srun|pgrep/ {print $1; exit}')
  [ -n "$B" ] || { echo "[$(pt)] watchdog bash not found in job $OLD"; return 3; }
  WENV=$(echo "$T" | awk -v b=$B '$1=="ENV" && $2==b {$1=""; $2=""; print}' | sed 's/^ *//')
  DESC=$(echo "$T" | grep -v '^ENV ' | awk -v b=$B '{par[$1]=$2; line[$1]=$0} END {for (p in par) {q=p; while (q in par && q != b && q > 1) q=par[q]; if (q == b && p != b) print line[p]}}')
  echo "[$(pt)] job $OLD: watchdog bash pid $B; env: ${WENV:-?}"; echo "  descendants:"; echo "${DESC:-  (none)}" | sed 's/^/    /'
  for v in ENVF NODEF TARGET MIN_STEP RESUME_PIN PIN_STEP CODE; do echo "$WENV" | grep -q "$v=" || { echo "    env $v missing: refusing"; return 3; }; done
  [ "$(echo "$DESC" | grep -c .)" = 1 ] && echo "$DESC" | grep -qE ' sleep [0-9]+$' && { echo "    -> SLEEPING between polls, nothing else in flight (safe)"; return 0; }
  echo "    -> NOT between polls"; return 1
}
check; RC=$?
[ $GO = 1 ] || exit $RC
for i in $(seq 1 20); do [ $RC = 0 ] && break; sleep 3; check; RC=$?; done
[ $RC = 0 ] || { echo "[$(pt)] ABORT: v1 never seen sleeping between polls in 60 s; nothing changed"; exit 1; }
scancel "$OLD"; for i in $(seq 1 30); do squeue -h -j "$OLD" | grep -q . || break; sleep 2; done
squeue -h -j "$OLD" | grep -q . && { echo "[$(pt)] ABORT: $OLD still in squeue after scancel"; exit 1; }
NEW=$(env $UNSET $WENV sbatch --parsable $K/ppm811s2p_watchdog_v2.sbatch)
[[ "$NEW" =~ ^[0-9]+$ ]] || { echo "[$(pt)] ALERT: v2 submit returned '$NEW' (v1 $OLD cancelled: NO watchdog): resubmit by hand: $WENV sbatch $K/ppm811s2p_watchdog_v2.sbatch"; exit 1; }
python3 - "$LR" "$NEW" <<'EOF'
import os, re, sys
p, new = sys.argv[1], sys.argv[2]; s = open(p).read(); s2 = re.sub(r"^WD=\d+$", f"WD={new}", s, flags=re.M)
open(p + ".new", "w").write(s2); os.replace(p + ".new", p)
EOF
echo "[$(pt)] swapped: v1 watchdog $OLD cancelled between polls -> v2 job $NEW ($WENV); LAUNCHED_RUN WD=$NEW"
for i in $(seq 1 60); do grep -q 'armed' /data/home/amartyavarma/ppm/logs/ppm811s2p-watchdog-$NEW.log 2>/dev/null && break; sleep 2; done
echo "--- v2 log:"; tail -3 /data/home/amartyavarma/ppm/logs/ppm811s2p-watchdog-$NEW.log 2>/dev/null

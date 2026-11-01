#!/bin/bash
# monosam go/ (10-26): STAGED node-local /scratch cleanup for the aloe-pose-mono-8020-sam FT hosts. NOT RUN BY ANYONE YET.
# Deletes ONLY rows of go/scratch_candidates.tsv with class SAFE, only on the chosen hosts, only via scratch_guard.sh guard_rm
# (refuses anything protected / any parent of a protected path / anything holding a *keep* entry). CHECK and KEEP rows are never touched.
#
# Default is --dry-run: prints what it WOULD delete; runs no srun, touches no node.
# Usage:
#   scratch_cleanup.sh [--option 12|8] [--hosts slinky-a,slinky-b] [--all-safe] [--jobid J] [--execute]
#     --option 12   (default) the 12-host plan, target >= 240 GiB free per host
#     --option 8    the 8-host plan, target >= 300 GiB free per host
#     --hosts       override the host list (MoT hosts only; a host outside the option's plan needs --all-safe)
#     --all-safe    delete every SAFE row on those hosts (default: only rows whose `pick` column has the option = minimal set to reach the target)
#     --jobid J     run the per-host step as `srun --overlap --jobid J` (e.g. a held FT/MoT allocation); default = new tiny allocation per host
#     --execute     really delete. Needs MAIN_GO=1 in the environment (main's relayed approval of SCRATCH_CANDIDATES.md).
# Each host: 1 CPU, --mem=512M, nice 19 + ionice idle; re-runs guard_check on the node for every path first and aborts that host
# if any is refused; skips hfb/xet-cache while an HF upload process is alive on the node; logs to go/SCRATCH_CLEANUP.log.
set -u
GO=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/go
TSV=$GO/scratch_candidates.tsv
GUARD=/data/home/amartyavarma/ppm/scratch_guard.sh
LOG=$GO/SCRATCH_CLEANUP.log
H12="slinky-15,slinky-14,slinky-5,slinky-7,slinky-1,slinky-2,slinky-17,slinky-3,slinky-8,slinky-0,slinky-19,slinky-6"
MOTH="slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19"
H8="slinky-15,slinky-14,slinky-5,slinky-7,slinky-1,slinky-2,slinky-17,slinky-6"
OPT=12; HOSTS=""; ALL=0; JOB=""; EXEC=0
while [ $# -gt 0 ]; do
  case "$1" in
    --option) OPT=$2; shift 2;;
    --hosts) HOSTS=$2; shift 2;;
    --all-safe) ALL=1; shift;;
    --jobid) JOB=$2; shift 2;;
    --execute) EXEC=1; shift;;
    --dry-run) EXEC=0; shift;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
case "$OPT" in 12) DEF=$H12;; 8) DEF=$H8;; *) echo "--option must be 12 or 8" >&2; exit 2;; esac
[ -z "$HOSTS" ] && HOSTS=$DEF
for h in ${HOSTS//,/ }; do
  case ",$MOTH," in *",$h,"*) ;; *) echo "REFUSED: $h is not one of MoT's 16 hosts" >&2; exit 2;; esac
  case ",$DEF," in *",$h,"*) ;; *) [ $ALL = 1 ] || { echo "REFUSED: $h is not in the $OPT-host plan ($DEF): it has no pick rows; use --all-safe (e.g. slinky-13 as the swap for slinky-3)" >&2; exit 2; };; esac
  case "$h" in slinky-4|slinky-9|slinky-10|slinky-18) echo "REFUSED: $h is off-limits" >&2; exit 2;; esac
done
[ -r "$TSV" ] || { echo "missing $TSV" >&2; exit 2; }
[ -r "$GUARD" ] || { echo "missing $GUARD (fail closed)" >&2; exit 2; }
if [ $EXEC = 1 ] && [ "${MAIN_GO:-0}" != 1 ]; then echo "REFUSED: --execute needs MAIN_GO=1 (main's relayed approval)" >&2; exit 2; fi
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|ILP|MOT)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }

rc_all=0
for h in ${HOSTS//,/ }; do
  # rows: host path GiB class proof pick ; SAFE only; pick filter unless --all-safe
  sel=$(awk -F'\t' -v h="$h" -v o="$OPT" -v all=$ALL 'NR>1 && $1==h && $4=="SAFE" { if (all==1 || index(","$6",", ","o",")>0) print $2"\t"$3 }' "$TSV")
  n=$(grep -c . <<<"$sel"); gib=$(awk -F'\t' '{s+=$2} END{printf "%.1f", s}' <<<"$sel")
  echo "== $h: $n SAFE paths, ${gib} GiB (option $OPT, $([ $ALL = 1 ] && echo all-safe || echo minimal-pick))"
  [ "$n" -gt 0 ] || continue
  while IFS=$'\t' read -r p g; do
    case "$p" in /scratch/amartyavarma/?*) ;; *) echo "REFUSED odd path $p" >&2; exit 2;; esac
    echo "   $([ $EXEC = 1 ] && echo DELETE || echo WOULD) guard_rm $p  ($g GiB)"
  done <<<"$sel"
  [ $EXEC = 1 ] || continue
  paths=$(cut -f1 <<<"$sel" | tr '\n' ' ')
  remote='
set -u; h=$(hostname -s); source '"$GUARD"'
b=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
for p in '"$paths"'; do guard_check "$p" || { echo "$h ABORT: guard refused $p; nothing deleted on $h"; exit 3; }; done
up=$(pgrep -u "$(id -u)" -f "[h]f_xet|[u]pload_large_folder|[u]pload_folder|[h]fb_onhost|[h]fb_train_node" | head -1)
for p in '"$paths"'; do
  [ -e "$p" ] || { echo "$h gone already $p"; continue; }
  case "$p" in */hfb/xet-cache) [ -n "$up" ] && { echo "$h SKIP $p (HF upload pid $up alive)"; continue; };; esac
  nice -n 19 ionice -c3 bash -c "source '"$GUARD"'; guard_rm \"$p\"" && echo "$h deleted $p" || echo "$h FAILED $p"
done
a=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); echo "$h /scratch free ${b}G -> ${a}G"'
  if [ -n "$JOB" ]; then
    out=$(timeout 3600 env $UNSET srun --overlap --jobid "$JOB" -N1 -w "$h" --ntasks-per-node=1 -c1 --mem=512M --cpu-bind=none -t 60 bash -c "$remote" 2>&1); rc=$?
  else
    out=$(timeout 3900 env $UNSET srun -p all -N1 -w "$h" --ntasks-per-node=1 -c1 --mem=512M --gres=none -t 60 --immediate=300 -J scratch-clean bash -c "$remote" 2>&1); rc=$?
  fi
  echo "$out" | sed 's/^/   /'
  { echo "[$(pt)] option $OPT host $h rc $rc"; echo "$out" | sed 's/^/    /'; } >> "$LOG"
  [ $rc = 0 ] || rc_all=$rc
done
[ $EXEC = 1 ] || echo "(dry run: nothing deleted, no srun issued; add --execute with MAIN_GO=1 after main's approval)"
exit $rc_all

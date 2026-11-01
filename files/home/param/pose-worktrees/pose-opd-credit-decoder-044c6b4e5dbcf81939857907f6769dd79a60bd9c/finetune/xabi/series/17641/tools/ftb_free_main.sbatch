#!/bin/bash
# NOT USED (coordinator 10-22 ~12:10 PT: the user gave no path-specific OK; no protected deletions; ftb_chain.sh no longer calls this).
#SBATCH -J ftb-freemain -p all -N16 --ntasks-per-node=1 -c 1 --mem=2G --gres=none -t 0:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/logs/ftb_freemain-%j.log
# aloe-v2pure-17641 (10-22): free PPM-host /scratch for FT-(b) (coordinator 10-22 ~11:55 PT). BOTH targets lie in PROTECTED_SCRATCH.txt
# (/scratch/amartyavarma/ppm/* and *keep*), whose rule is the USER's explicit, path-specific OK -> runs ONLY if control/USER_OK_FREE_MAIN exists and
# is non-empty (the coordinator writes the user's OK there, quoted). One task per PPM host; per deletion one line -> setup-logs/hfb/DELETED-1021-hfb-hold.log:
#  (1) /scratch/amartyavarma/ppm/ckpt-keep-ft/dcp-17352 iff dcp-17352 is VERIFY PASS on HF (hfb watch.log: 'finalize: bundle rc 0 PASS True ... bad []' +
#      'release ... 16/16 hosts') AND /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352/COMPLETE exists;
#  (2) /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-N, N < FTB_STEP, N != 16861, iff dcp-FTB_STEP is COMPLETE in ckpt-keep-ft (hard links) AND dcp-N is
#      VERIFY PASS on HF (same two watch.log lines). Never dcp-16861, never the final dcp-FTB_STEP, never the HF watcher's ckpt-keep holds.
set -uo pipefail
: "${FTB_STEP:?}"
OKF=/data/home/amartyavarma/ppm/aloe-v2pure-17641/control/USER_OK_FREE_MAIN
[ -s $OKF ] || { echo "ABORT: no user OK ($OKF): the targets are PROTECTED_SCRATCH paths"; exit 1; }
export FTB_STEP OKQ="$(head -c 300 $OKF | tr '\n' ' ')"
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c 1 --mem=1G --cpu-bind=none bash -c '
H=$(hostname -s); W=/data/home/amartyavarma/setup-logs/hfb/logs/watch.log; LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
K=/scratch/amartyavarma/ppm/ckpt-keep-ft; C=/scratch/amartyavarma/ppm/ckpt/ppm-v1; M=$FTB_STEP
pt() { TZ=America/Los_Angeles date "+%Y-%m-%d %H:%M:%S PT"; }
hfok() { grep -q "dcp-$1 finalize: bundle rc 0 PASS True .*bad \[\]" $W && grep -q "release dcp-$1 (uploaded + verified on HF): 16/16 hosts" $W; }
del() { local p=$1 why=$2 sz f0 f1 rc; [ -d "$p" ] || return 0
  case "$p" in $K/dcp-*|$C/dcp-*) ;; *) echo "$H REFUSE $p"; return 1;; esac
  case "$p" in */dcp-16861|*/dcp-16861/*|$C/dcp-$M) echo "$H REFUSE $p (switch / final)"; return 1;; esac
  sz=$(du -s -BG "$p" | cut -f1); f0=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
  rm -rf --one-file-system "$p"; rc=$?; f1=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
  echo "$(pt) DEL $H $p size $sz rc $rc free ${f0}G -> ${f1}G :: $why; user OK: $OKQ [aloe-v2pure-17641 FT-(b) host space]" | tee -a $LOG; }
if [ -d $K/dcp-17352 ] && hfok 17352 && [ -f /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17352/COMPLETE ]; then
  del $K/dcp-17352 "PPM 8,1,1 dcp-17352 keep-ft hard links (FT-(a) init source): VERIFY PASS on private HF InternalCan/ppm-v1-c (watch.log finalize PASS + release 16/16) and /data ckpt-infer/ppm-v1/local-17352 COMPLETE"
fi
if [ -f $K/dcp-$M/COMPLETE ]; then
  for d in $C/dcp-*; do n=${d##*/dcp-}; [[ "$n" =~ ^[0-9]+$ ]] || continue
    [ "$n" = 16861 ] && continue; [ "$n" -ge "$M" ] && continue
    hfok $n || { echo "$H keep $d (no HF VERIFY PASS record)"; continue; }
    del $d "PPM 8,1,1 non-final dcp-$n, superseded by the final dcp-$M of the delivery-2 run (COMPLETE, hard-linked in ckpt-keep-ft); VERIFY PASS on private HF (watch.log finalize PASS + release 16/16)"
  done
fi
echo "$H SCRATCHFREE $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)"' | sort -V

#!/bin/bash
#SBATCH -p all -n8 --ntasks-per-node=1 -c 1 --mem=2G --gres=none -t 12:00:00 -J si20kvox-bkfill
# si20k-vox: back up the intermediate node-local exports the watcher did not reach (it handles one at a time and stopped at the final):
# for each export-N (in order): every node copies its shard files into <run>/exports/export-N on /data (sizes verified), COMPLETE, then the
# private-HF upload (hf_up.sbatch ... delete-after-verify: the /data copy goes once HF verifies; node-local shards stay). Waits for /data
# >= 300 + size + 10 GB before each copy and for the HF verify before the next. usage: sbatch -N8 -w <the run's 8 hosts> backfill_exports.sbatch <run dir> <GB> <N...>
Q=$1; GB=$2; shift 2
RUN=$(basename $Q); EX=$Q/exports; LOC=/scratch/amartyavarma/$RUN/exports; BL=$EX/BACKUP_LOG.txt; HFS=/data/home/amartyavarma/ppm/si20k-vox-ppm/tools/hf_up.sbatch
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
lg() { echo "[$(pt)] backfill: $*" | tee -a $BL; }
for N in "$@"; do
  while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt $((300 + GB + 10)) ]; do sleep 60; done
  D=$EX/export-$N; mkdir -p $D
  out=$(srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 -c1 --mem=1G bash -c "cd $LOC/export-$N && for f in rank*.pt rep.pt meta.json; do [ -f \$f ] || continue; nice -n 10 cp -p \$f $D/\$f.tmp && mv $D/\$f.tmp $D/\$f && echo \"\$f \$(stat -c %s \$f)\"; done" 2>&1)
  ok=1; nf=0
  while read -r f s; do case "$f" in rank*.pt|rep.pt|meta.json) nf=$((nf+1)); [ "$(stat -c %s $D/$f 2>/dev/null)" = "$s" ] || ok=0;; esac; done <<< "$out"
  nr=$(ls $D | grep -c '^rank[0-9]*\.pt$')
  if [ $ok -eq 1 ] && [ $nr -eq 64 ] && [ $nf -eq 66 ]; then
    echo $N > $D/COMPLETE
    h=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $D export-$N delete-after-verify); lg "export-$N copied + verified ($(du -sb $D | cut -f1) B) -> HF job $h"
    while squeue -h -j $h 2>/dev/null | grep -q .; do sleep 30; done
    lg "export-$N HF job $h ended: $(cat $Q/hf_backup/HF_BACKUP-export-$N.json 2>/dev/null | cut -c1-120 || echo 'NO VERIFY RECORD')"
  else
    lg "export-$N copy NOT verified (files $nf, rank files $nr, ok $ok)"; echo "$out" | tail -3 >> $BL
  fi
done
lg "done $*"

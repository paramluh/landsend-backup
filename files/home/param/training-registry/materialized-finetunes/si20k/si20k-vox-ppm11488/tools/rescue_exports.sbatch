#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=8G --gres=none -t 6:00:00 -J si20k11488-rescue --exclude=slinky-4,slinky-18
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-ppm11488/logs/rescue-%j.out
# si20k-vox-ppm11488 RESCUE of the node-local intermediate exports left on /scratch when the FT job died writing its final node-local export
# (/scratch full: the watcher's /data gate 330 + 80 GB kept every intermediate on the nodes). One export at a time (newest first): node-local
# -> /data copy (sizes verified, COMPLETE), the node-local copy deleted, theta check (already fp32), HF upload delete-after-verify; the next
# export starts only after that HF job ended (/data stays >= 320 GB: TTS pause line).
Q=/data/home/amartyavarma/ppm/si20k-vox-ppm11488; EX=$Q/exports; LOC=/scratch/amartyavarma/si20k-vox-ppm11488/exports; BL=$EX/BACKUP_LOG.txt
H=slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17; HFS=/data/home/amartyavarma/ppm/si20k-txt-ppm/tools/hf_up.sbatch
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; FIX=/data/home/amartyavarma/ppm/si20k-txt-ppm/tools/theta_fixup.py; MET=$Q/runs/si20k-vox-ppm11488-j25245/metrics.jsonl
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
lg() { echo "[$(pt)] $*" | tee -a $BL; }
on8() { env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER srun -p all -w $H -N8 -n8 --ntasks-per-node=1 -c1 --mem=2G --gres=none -t 30 -J si20k11488-rcopy "$@"; }
lg "rescue job $SLURM_JOB_ID: node-local exports ${RESCUE_STEPS:-132 99 66} -> /data -> HF (export-165 done by 25288)"
for N in ${RESCUE_STEPS:-132 99 66}; do                         # 165 already rescued by job 25288
  while [ "$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)" -lt 400 ]; do sleep 30; done
  D=$EX/export-$N; mkdir -p $D
  out=$(on8 bash -c "cd $LOC/export-$N && for f in rank*.pt rep.pt meta.json COMPLETE; do [ -f \$f ] || continue; nice -n 10 cp -p \$f $D/\$f.tmp && mv $D/\$f.tmp $D/\$f && echo \"\$f \$(stat -c %s \$f)\"; done" 2>&1)
  ok=1; nf=0
  while read -r f s; do case "$f" in rank*.pt|rep.pt|meta.json) nf=$((nf + 1)); [ "$(stat -c %s $D/$f 2>/dev/null)" = "$s" ] || ok=0;; esac; done <<< "$out"
  nr=$(ls $D | grep -c '^rank[0-9]*\.pt$')
  if [ $ok -ne 1 ] || [ $nr -ne 64 ] || [ ! -f $D/rep.pt ] || [ ! -f $D/meta.json ]; then lg "export-$N: copy NOT verified (files $nf, ranks $nr, ok $ok): STOP"; echo "$out" | tail -5 >> $BL; exit 1; fi
  echo $N > $D/COMPLETE
  lg "export-$N -> $D: COPIED + VERIFIED ($nf files, $(du -sb $D | cut -f1) B); /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"
  r=$($PY -I $FIX $D --metrics $MET --step $N --run-dir $Q 2>&1 | grep THETA_FIXUP | grep -o '"status": "[^"]*"'); lg "export-$N: theta check $r"
  rmo=$(on8 bash -c "source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_rm $LOC/export-$N && echo \"\$(hostname -s) \$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G\"" 2>&1 | grep G | sort -V | tr '\n' ' ')
  lg "export-$N: node-local copies DELETED (verified on /data); /scratch free: $rmo"
  j=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $D export-$N delete-after-verify); lg "export-$N: HF upload job $j -> private InternalCan/si20k-vox-ppm11488/export-$N/ (then the /data copy is deleted)"
  while squeue -h -j $j 2>/dev/null | grep -q .; do sleep 20; done
  lg "export-$N: HF job $j ended: $(grep "export-$N HF upload rc" $BL | tail -1 | grep -o 'rc [0-9]')"
done
lg "rescue done"

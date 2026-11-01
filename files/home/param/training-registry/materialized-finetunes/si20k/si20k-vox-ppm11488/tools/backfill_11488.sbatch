#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=8G --gres=none -t 6:00:00 -J si20k11488-bkfill --exclude=slinky-4,slinky-18
#SBATCH -o /data/home/amartyavarma/ppm/si20k-vox-ppm11488/logs/backfill-%j.out
# si20k-vox-ppm11488 BACKFILL after the job crashed writing its final node-local export (/scratch full: the watcher's /data gate had held
# export-66..165 node-local). (1) the PARTIAL node-local export-198 dirs (incomplete rank files, no COMPLETE: not a checkpoint) are deleted;
# (2) export-66, 99, 132, 165 (complete, node-local only = unexported) are copied one at a time to <run>/exports (sizes verified, COMPLETE,
# fp32-theta check), their node-local copies deleted, then the private-HF upload (delete-after-verify); /data gate 305 + 80 GB;
# (3) once every PPM host has >= 45 GB free /scratch, the no-corruption run (si20k-vox-ppm-nocorrupt) is launched + its watcher (once).
Q=/data/home/amartyavarma/ppm/si20k-vox-ppm11488; N2=/data/home/amartyavarma/ppm/si20k-vox-ppm-nocorrupt; EX=$Q/exports; BL=$EX/BACKUP_LOG.txt
LOC=/scratch/amartyavarma/si20k-vox-ppm11488/exports; HP=slinky-0,slinky-1,slinky-3,slinky-5,slinky-8,slinky-9,slinky-16,slinky-17
HFS=/data/home/amartyavarma/ppm/si20k-txt-ppm/tools/hf_up.sbatch; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
FIX=/data/home/amartyavarma/ppm/si20k-txt-ppm/tools/theta_fixup.py; MET=$Q/runs/si20k-vox-ppm11488-j25245/metrics.jsonl
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
lg() { echo "[$(pt)] $*" | tee -a $BL; }
onnodes() { env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER srun -p all -w $HP -N8 -n8 --ntasks-per-node=1 -c1 --mem=2G --gres=none -t 60 -J si20k11488-bkcopy "$@"; }
launch_noc() {
  [ -f $N2/LAUNCH_DONE ] && return 0
  mn=$(onnodes bash -c 'df -BG --output=avail /scratch | tail -1 | tr -dc 0-9' 2>/dev/null | sort -n | head -1)
  [ -n "$mn" ] && [ "$mn" -ge 45 ] || { lg "nocorrupt launch: min /scratch free ${mn:-?} GB < 45, wait"; return 1; }
  j=$(sbatch --parsable -w $HP $N2/si20k_vox_ppm_nocorrupt.sbatch) || return 1
  w=$(sbatch --parsable --exclude=slinky-4,slinky-18 -o $N2/logs/watch-%j.out $N2/tools/watch_noc.sbatch $N2 $j si20kvoxnocorrupt 80 600 -)
  echo $j > $N2/LAUNCHED_JOB; echo "$(pt) nocorrupt $j (watch $w): relaunch after the 11488 scratch backfill freed /scratch (min $mn GB)" | tee $N2/LAUNCH_DONE
  echo "- $(pt): launched job $j (after the 11488 scratch backfill; the first chained launch 25283 aborted at preflight: /scratch full)." >> $N2/README.md
  lg "nocorrupt launched: job $j (min /scratch free $mn GB)"
}
lg "backfill job $SLURM_JOB_ID: partial final export-198 dirs + export-66/99/132/165 node-local -> /data -> HF"
out=$(onnodes bash -c "source /data/home/amartyavarma/ppm/scratch_guard.sh; d=$LOC/export-198; [ -d \$d ] && [ ! -f \$d/COMPLETE ] && n=\$(ls \$d | wc -l) && guard_rm \$d && echo \"\$(hostname -s) deleted partial \$d (\$n files, no COMPLETE)\"" 2>&1 | grep deleted | sort -V | tr '\n' ';')
lg "partial node-local export-198 (failed final write) deleted: $out"
launch_noc
for N in 66 99 132 165; do
  while :; do FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); [ "$FD" -ge $((305 + 80)) ] && break; lg "export-$N: waiting for /data space ($FD GB < 385)"; sleep 60; done
  D=$EX/export-$N; mkdir -p $D; t0=$(date +%s)
  out=$(onnodes bash -c "cd $LOC/export-$N && for f in rank*.pt rep.pt meta.json; do [ -f \$f ] || continue; nice -n 10 cp -p \$f $D/\$f.tmp && mv $D/\$f.tmp $D/\$f && echo \"\$f \$(stat -c %s \$f)\"; done" 2>&1)
  ok=1; nf=0
  while read -r f s; do case "$f" in rank*.pt|rep.pt|meta.json) ;; *) continue;; esac; nf=$((nf + 1)); [ "$(stat -c %s $D/$f 2>/dev/null)" = "$s" ] || { ok=0; lg "export-$N: size mismatch $f"; }; done <<< "$out"
  nr=$(ls $D | grep -c '^rank[0-9]*\.pt$')
  if [ $ok -eq 1 ] && [ $nr -eq 64 ] && [ -f $D/rep.pt ] && [ -f $D/meta.json ] && [ $nf -eq 66 ]; then
    echo $N > $D/COMPLETE
    lg "export-$N -> Weka $D: COPIED + VERIFIED ($nf files, $(du -sb $D | cut -f1) B; $(( $(date +%s) - t0 ))s)"
    fx=$($PY -I $FIX $D --metrics $MET --step $N --run-dir $Q 2>&1 | grep THETA_FIXUP | grep -o '"status": "[^"]*"')
    lg "export-$N: theta check $fx"
    rmo=$(onnodes bash -c "source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_rm $LOC/export-$N && echo \"\$(hostname -s) \$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)G\"" 2>&1 | grep -v srun | sort -V | tr '\n' ' ')
    lg "export-$N: node-local copies DELETED (verified on /data); /scratch free: $rmo"
    h=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $D export-$N delete-after-verify); lg "export-$N: HF upload job $h -> private InternalCan/si20k-vox-ppm11488/export-$N/"
  else
    lg "export-$N: copy NOT verified (files $nf, rank files $nr, ok $ok): node-local kept; stop"; exit 1
  fi
  launch_noc
done
launch_noc || { sleep 120; launch_noc; }
lg "backfill done"

#!/bin/bash
# scaling rerun driver (orchestration only; copies / backups run on compute nodes): for each export of job 21832 as it lands:
#   GCS backup (verified) + copy to the eval node's /scratch (sizes checked) -> delete the /data copy (except the final 223) -> launch its QA eval.
# 14 -> slinky-4 (GPUs 4,5, during training) + a 2nd copy on slinky-18; 28 -> slinky-7; 59 -> slinky-15; 117 -> slinky-18; 223 stays on /data.
set -u
Q=/data/home/amartyavarma/ppm/qqa-ft; S=$Q/scale; QE=$Q/qa_eval; SC=/scratch/amartyavarma/ppm-qqa-scale; LOG=$Q/logs/scale_driver.log
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
say() { echo "[$(pt)] $*" | tee -a $LOG; }
conv() { case $1 in 14) echo 5913;; 28) echo 11925;; 59) echo 25126;; 117) echo 49865;; 223) echo 95192;; esac; }
copy_to() {   # step node
  srun -p all -w $2 -N1 -n1 -c4 --mem=8G --gres=none -t 60 -J qqa-scale-cp bash -c "mkdir -p $SC && rm -rf $SC/export-$1.tmp && cp -r $S/export-$1 $SC/export-$1.tmp && \
    a=\$(cd $S/export-$1 && find . -type f -printf '%P %s\n' | sort | md5sum); b=\$(cd $SC/export-$1.tmp && find . -type f -printf '%P %s\n' | sort | md5sum); \
    [ \"\$a\" = \"\$b\" ] && mv $SC/export-$1.tmp $SC/export-$1 && echo COPY_OK" 2>&1 | tail -1
}
handle() {   # step
  s=$1; E=$S/export-$s
  until [ -f $E/COMPLETE ]; do sleep 15; done
  say "export-$s landed ($(du -sh $E | cut -f1))"
  if [ $s = 223 ]; then
    GCS_BK_WAIT=1 bash $Q/gcs_backup.sh $E > $Q/logs/scale-gcs-$s.log 2>&1 && say "export-$s GCS verified" || say "!!! export-$s GCS backup FAILED"
    return
  fi
  ( GCS_BK_WAIT=1 bash $Q/gcs_backup.sh $E > $Q/logs/scale-gcs-$s.log 2>&1; echo $? > $S/.gcs-$s ) &
  case $s in 14) nodes="slinky-4 slinky-18";; 28) nodes=slinky-7;; 59) nodes=slinky-15;; 117) nodes=slinky-18;; esac
  ok=1
  for n in $nodes; do r=$(copy_to $s $n); [ "$r" = COPY_OK ] && say "export-$s copied to $n:$SC/export-$s" || { ok=0; say "!!! copy export-$s -> $n FAILED: $r"; }; done
  wait
  g=$(cat $S/.gcs-$s 2>/dev/null)
  if [ $ok = 1 ] && [ "$g" = 0 ] && [ -f $E/GCS_BACKUP.json ]; then rm -rf $E; say "export-$s: GCS-verified + scratch copies -> /data copy deleted"; else say "!!! export-$s kept on /data (copy ok $ok, gcs rc $g)"; fi
  first=${nodes%% *}
  if [ $first = slinky-4 ]; then J=$(bash $QE/launch_eval.sh ppmqqa-scale-$(conv $s) $SC/export-$s slinky-4 "4 5"); else J=$(bash $QE/launch_eval.sh ppmqqa-scale-$(conv $s) $SC/export-$s $first "0 1 2 3 4 5 6 7"); fi
  say "eval ppmqqa-scale-$(conv $s) job $J on $first"
}
for s in 14 28 59 117; do handle $s; done
handle 223
say "driver done"

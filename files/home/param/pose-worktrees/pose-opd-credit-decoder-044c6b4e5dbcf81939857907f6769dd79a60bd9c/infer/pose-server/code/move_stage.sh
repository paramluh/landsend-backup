#!/bin/bash
# move the gathered exports from /data stage -> slinky-15 node-local /scratch, verify sizes, then delete from /data
set -u
S=/data/home/amartyavarma/ppm/ppm-pose-server/stage; D=/scratch/amartyavarma/ppm-pose-server/stage
for m in 7932 13533; do
  mkdir -p $D/$m
  for f in $S/$m/*; do b=$(basename $f); cp $f $D/$m/.$b.tmp && mv $D/$m/.$b.tmp $D/$m/$b; done
  ok=1
  for f in $S/$m/*; do b=$(basename $f); [ "$(stat -c %s $f)" = "$(stat -c %s $D/$m/$b 2>/dev/null)" ] || { echo "$m size mismatch $b"; ok=0; }; done
  n=$(ls $D/$m/rank*.pt | wc -l)
  [ "$n" = 64 ] && [ -f $D/$m/rep.pt ] && [ -f $D/$m/meta.json ] || { echo "$m incomplete: $n rank files"; ok=0; }
  if [ $ok = 1 ]; then echo ok > $D/$m/COMPLETE; rm -rf $S/$m; echo "$m moved: $(du -sh $D/$m | cut -f1), 64 rank files + rep.pt + meta.json; /data copy deleted"; fi
done
rmdir $S 2>/dev/null; df -h /scratch | tail -1

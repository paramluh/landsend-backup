#!/bin/bash
# read-only survey of node-local s3-rows stores on this host
H=$(hostname -s)
for d in /scratch/amartyavarma/s3-rows /scratch/amartyavarma/s3-rows-extra/*; do
  if [ -e "$d/NODE.json" ]; then
    node=$(grep -o '"node": *"[^"]*"' $d/NODE.json | head -1)
    nc=$(ls $d 2>/dev/null | grep -c 'codes_user.npy$'); nd=$(ls $d | grep -c '\.DONE$'); np_=$(ls $d | grep -c 'persona.arrow$'); nw=$(ls $d | grep -c 'words.arrow$')
    echo "$H|$d|$node|done=$nd codes_user=$nc words=$nw persona=$np_|rows.parquet=$( [ -f $d/rows.parquet ] && echo y || echo n)"
  else
    echo "$H|$d|MISSING"
  fi
done
ls -d /scratch/amartyavarma/s3-rows* 2>/dev/null | tr '\n' ' '; echo
for f in $(find /scratch/amartyavarma -maxdepth 5 -name NODE.json 2>/dev/null); do
  if grep -q 's3-duplex-rows-v1' $f 2>/dev/null; then echo "$H|ALT|$f|$(grep -o '"node": *"[^"]*"' $f | head -1)|codes=$(ls $(dirname $f) | grep -c codes_user.npy)"; fi
done
df -h /scratch | tail -1 | sed "s/^/$H|DF|/"

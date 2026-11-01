#!/bin/bash
# pose5 D: merge G0COND into workstream A's code-811-s2p (run ONLY after code-811-s2p/tests/s2p/S2P_PASS exists).
# The hand-resolved merge was built from A's files whose md5s are in $M/BASE_MD5.txt. If A's three files still have exactly those md5s,
# the merged files are copied in (originals kept as *.bak-pre-g0cond-1027); otherwise it stops (re-resolve by hand).
# Writes pose5/g0cond_onto_s2p.patch (A's files -> merged) and code-811-s2p/tests/g0/MERGE.json.
set -euo pipefail
S=/data/home/amartyavarma/ppm/pose5/code-811-s2p
M=/tmp/claude-243004069/-data-home-amartyavarma/cb50b1a9-f392-41e9-94e8-9155bd3fd8c9/scratchpad/merge1
[ -e $S/tests/s2p/S2P_PASS ] || { echo "A has not written S2P_PASS"; exit 1; }
for f in data_ppm.py model_ppm.py train_ppm.py; do
  a=$(md5sum < $S/$f | cut -d' ' -f1); b=$(grep " $M/base/$f\$" $M/BASE_MD5.txt | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "A's $f changed since the merge was built ($a != $b): re-resolve"; exit 2; }
done
mkdir -p $S/tests/g0/out
: > /data/home/amartyavarma/ppm/pose5/g0cond_onto_s2p.patch
for f in data_ppm.py model_ppm.py train_ppm.py; do
  cp -p $S/$f $S/$f.bak-pre-g0cond-1027
  diff -u $S/$f $M/work/$f | sed "s#^--- $S/#--- a/#; s#^+++ $M/work/#+++ b/#" >> /data/home/amartyavarma/ppm/pose5/g0cond_onto_s2p.patch || true
  cp $M/work/$f $S/$f
done
/data/home/amartyavarma/venvs/qwen-s2/bin/python -m py_compile $S/data_ppm.py $S/model_ppm.py $S/train_ppm.py
md5sum $S/data_ppm.py $S/model_ppm.py $S/train_ppm.py > $S/tests/g0/MERGED_MD5.txt
echo "{\"merged_at_pt\": \"$(TZ=America/Los_Angeles date '+%F %T PT')\", \"base_md5\": \"$M/BASE_MD5.txt\", \"patch\": \"pose5/g0cond_onto_s2p.patch\", \"backups\": \"*.bak-pre-g0cond-1027\"}" > $S/tests/g0/MERGE.json
cp $M/BASE_MD5.txt $S/tests/g0/BASE_MD5_A.txt
echo "merged into $S"

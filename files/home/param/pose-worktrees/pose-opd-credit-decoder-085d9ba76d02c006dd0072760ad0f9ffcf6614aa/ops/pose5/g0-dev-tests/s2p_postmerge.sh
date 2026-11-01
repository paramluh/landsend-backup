#!/bin/bash
# pose5 D: re-run workstream A's s2p suite on the MERGED code-811-s2p (G0COND merged, PPM_G0COND unset = off), without touching A's evidence:
# A's harness is copied to code-811-s2p/tests/s2p_postmerge (paths -> that dir, own out/, own node scratch); its micros step is skipped
# (A's pickles in pose5/s2p-test-micros are read as they are). Submits: stage1 cfg, stage1 tiny, stage2 resume (A's job scripts as written).
set -euo pipefail
S=/data/home/amartyavarma/ppm/pose5/code-811-s2p
P=$S/tests/s2p_postmerge
[ -e $S/tests/g0/MERGE.json ] || { echo "not merged yet"; exit 1; }
mkdir -p $P/out $P/logs
for f in $S/tests/s2p/*.py $S/tests/s2p/*.sh $S/tests/s2p/*.sbatch; do cp -p "$f" $P/; done
sed -i -e 's#/tests/s2p\b#/tests/s2p_postmerge#g' \
       -e "s#/scratch/amartyavarma/pose5-s2p-test#/scratch/amartyavarma/pose5-s2p-postmerge-test#g" $P/*.py $P/*.sh $P/*.sbatch
sed -i -e 's#OUTD = f"{COPY}/tests/s2p/out"#OUTD = f"{COPY}/tests/s2p_postmerge/out"#' $P/s2p_cpu.py
grep -n "tests/s2p[^_]" $P/*.py $P/*.sh $P/*.sbatch | grep -v "^.*#" || true
grep -n "OUTD\|SCR =" $P/s2p_cpu.py
echo "harness copied to $P"

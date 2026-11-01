#!/bin/bash
# (b-ii) step 1: the reduced-depth REAL weight cache on this node's /scratch (read-only source ckpt-infer/ppm-v1/local-17641)
source /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge/stage3_common.sh
[ -f $WC ] || $N $PY $TD/real_weights.py $WC $LB $LT $LD || exit 1
ls -la $WC; echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

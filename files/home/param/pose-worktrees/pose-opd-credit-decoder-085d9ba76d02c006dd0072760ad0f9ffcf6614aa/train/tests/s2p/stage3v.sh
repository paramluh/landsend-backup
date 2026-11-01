#!/bin/bash
# (b-ii) step 2: one variant (args: ORIG|COPY off|full|cb0 port) on the real widths + real weights, every real micro
source /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p/stage3_common.sh
tag=real_$(echo $1 | tr A-Z a-z)_$2
$N $PY $TD/run_variant.py --code $1 --variant $2 --model real --lb $LB --lt $LT --ld $LD --wcache $WC --micros $MIC --port $3 --out $OUT/$tag.json || exit 1
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

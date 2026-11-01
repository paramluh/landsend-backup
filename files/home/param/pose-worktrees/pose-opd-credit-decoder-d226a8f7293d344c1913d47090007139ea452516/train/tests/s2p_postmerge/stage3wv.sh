#!/bin/bash
# (b-ii) on a node without the weight cache yet: build it on this node's /scratch (if missing), then run one variant (args as stage3v.sh)
source /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge/stage3_common.sh
df -BG --output=avail /scratch | tail -1
mkdir -p $(dirname $WC); ( flock 9; [ -f $WC ] || $N $PY $TD/real_weights.py $WC $LB $LT $LD ) 9>$WC.lock || exit 1   # one builder per node
tag=real_$(echo $1 | tr A-Z a-z)_$2
$N $PY $TD/run_variant.py --code $1 --variant $2 --model real --lb $LB --lt $LT --ld $LD --wcache $WC --micros $MIC --port $3 --out $OUT/$tag.json || exit 1
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

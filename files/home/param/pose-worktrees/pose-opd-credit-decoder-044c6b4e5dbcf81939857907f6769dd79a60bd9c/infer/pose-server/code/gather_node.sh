#!/bin/bash
# copy this node's share of a node-local PPM export (ppm-local-bf16-shards-v1: rank<r>.pt for its 8 ranks [+ rep.pt, meta.json]) to a /data stage dir
# usage: gather_node.sh SRC_EXPORT_DIR STAGE_DIR FIRST(0|1)
S=$1; D=$2; FIRST=$3
ls $S/rank*.pt > /dev/null || { echo "$(hostname) ABORT: no rank files in $S"; exit 1; }
for f in $S/rank*.pt; do b=$(basename $f); ionice -c3 nice -n 19 cp $f $D/.$b.tmp && mv $D/.$b.tmp $D/$b; done
if [ "$FIRST" = 1 ]; then ionice -c3 nice -n 19 cp $S/rep.pt $D/.rep.pt.tmp && mv $D/.rep.pt.tmp $D/rep.pt; cp $S/meta.json $D/meta.json; fi
echo "$(hostname) done $(ls $S/rank*.pt | wc -l) rank files"

#!/bin/bash
# on the SOURCE host of SLOT: list the tt-v3 store, check NODE.json node == SLOT == hostname, copy rows.parquet + NODE.json to plan/ (small)
set -euo pipefail
source /data/home/amartyavarma/ppm/ttconvo-ft/tools/log.sh
SLOT=$1; SRC=/scratch/amartyavarma/tt-v3/rows; P=/data/home/amartyavarma/ppm/ttconvo-ft/plan
[ "$(hostname)" == "$SLOT" ] || { log "FAIL plan_copy $SLOT: host $(hostname)"; exit 1; }
node=$(/data/home/amartyavarma/venvs/qwen-s2/bin/python -I -c "import json;print(json.load(open('$SRC/NODE.json'))['node'])")
[ "$node" == "$SLOT" ] || { log "FAIL plan_copy $SLOT: NODE.json node=$node"; exit 1; }
nonreg=$(find $SRC -mindepth 1 -maxdepth 1 ! -type f | wc -l)
nf=$(find $SRC -mindepth 1 -maxdepth 1 -type f | wc -l)
sz=$(du -sb $SRC | cut -f1)
cp $SRC/rows.parquet $P/rows-$SLOT.parquet.tmp; mv $P/rows-$SLOT.parquet.tmp $P/rows-$SLOT.parquet
cp $SRC/NODE.json $P/NODE-$SLOT.json.tmp; mv $P/NODE-$SLOT.json.tmp $P/NODE-$SLOT.json
log "plan_copy $SLOT: node=$node files=$nf nonregular=$nonreg bytes=$sz; copied rows.parquet+NODE.json to plan/"

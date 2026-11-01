#!/bin/bash
# build s3-persona-sidecar-v1 sidecars for the 5 tt-v3 slots + evalv3 with conv-pool-v2's build_pool_persona.py (prefix 'p' in a temp dir),
# then place them as persona/<slot>/<shard>.persona.arrow (no prefix) + STATS.json
set -euo pipefail
source /data/home/amartyavarma/ppm/ttconvo-ft/tools/log.sh
B=/data/home/amartyavarma/ppm/ttconvo-ft; TMP=$B/persona-tmp; P=$B/plan
rm -rf $TMP; mkdir -p $TMP
log "persona_build start (build_pool_persona.py -> $TMP)"
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/qwen-1ch-data/conv-pool-v2/scripts/build_pool_persona.py $TMP \
  slinky-4:p:$P/rows-slinky-4.parquet slinky-7:p:$P/rows-slinky-7.parquet slinky-11:p:$P/rows-slinky-11.parquet \
  slinky-13:p:$P/rows-slinky-13.parquet slinky-18:p:$P/rows-slinky-18.parquet \
  evalv3:p:/data/home/amartyavarma/qwen-1ch-data/tt-v3-rows/evalstore/rows/rows.parquet 2>&1 | tee $TMP/build.log
for tag in slinky-4 slinky-7 slinky-11 slinky-13 slinky-18 evalv3; do
  O=$B/persona/$tag; rm -rf $O; mkdir -p $O
  for f in $TMP/$tag/p-*.persona.arrow; do b=$(basename $f); mv $f $O/${b#p-}; done
  mv $TMP/$tag/STATS-p.json $O/STATS.json
  log "persona_build placed $tag: $(ls $O/*.persona.arrow | wc -l) sidecars + STATS.json in $O"
done
log "persona_build done"

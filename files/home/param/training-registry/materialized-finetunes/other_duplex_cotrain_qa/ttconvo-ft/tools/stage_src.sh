#!/bin/bash
# on the SOURCE host of SLOT: single-stream copy of every file of /scratch/amartyavarma/tt-v3/rows to Weka xfer/<slot>/, hashing the
# source bytes on the way (cat | tee | sha256sum = one read); writes plan/sha256-<slot>.txt (sha256sum format, bare names)
set -euo pipefail
source /data/home/amartyavarma/ppm/ttconvo-ft/tools/log.sh
SLOT=$1; SRC=/scratch/amartyavarma/tt-v3/rows; B=/data/home/amartyavarma/ppm/ttconvo-ft; X=$B/xfer/$SLOT; L=$B/plan/sha256-$SLOT.txt
[ "$(hostname)" == "$SLOT" ] || { log "FAIL stage $SLOT: host $(hostname)"; exit 1; }
free=$(df -B1G --output=avail /data | tail -1 | tr -dc 0-9); need=$(( $(du -sb $SRC | cut -f1) / 1073741824 + 1 ))
[ $((free - need)) -ge 300 ] || { log "FAIL stage $SLOT: /data free ${free}G - ${need}G < 300G"; exit 1; }
log "stage $SLOT start: /data free ${free}G, staging ${need}G -> $X"
rm -rf $X; mkdir -p $X; : > $L.tmp
for f in $(find $SRC -mindepth 1 -maxdepth 1 -type f -printf '%f\n' | sort); do
  h=$(cat $SRC/$f | tee $X/$f | sha256sum | cut -d' ' -f1)
  [ "$(stat -c %s $SRC/$f)" == "$(stat -c %s $X/$f)" ] || { log "FAIL stage $SLOT: size mismatch $f"; exit 1; }
  echo "$h  $f" >> $L.tmp
done
mv $L.tmp $L
# plan copies of rows.parquet / NODE.json must match the source hashes
for pair in "rows.parquet rows-$SLOT.parquet" "NODE.json NODE-$SLOT.json"; do set -- $pair
  [ "$(grep "  $1\$" $L | cut -d' ' -f1)" == "$(sha256sum $B/plan/$2 | cut -d' ' -f1)" ] || { log "FAIL stage $SLOT: plan copy $2 hash"; exit 1; }
done
log "stage $SLOT done: $(wc -l < $L) files hashed+staged ($(du -sh $X | cut -f1)); plan copies match; /data free $(df -B1G --output=avail /data | tail -1 | tr -dc 0-9)G"

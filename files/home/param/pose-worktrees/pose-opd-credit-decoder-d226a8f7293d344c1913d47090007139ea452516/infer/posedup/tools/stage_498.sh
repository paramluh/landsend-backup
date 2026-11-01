#!/bin/bash
# copy the final DCP export from /data to slinky-10 node-local (8 parallel copies, sizes checked, COMPLETE last, then rename), mark it SWAP_OK
SRC=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/export-poseduplex-v5-498; DST=/scratch/amartyavarma/aloe-posedup-live/export-498-dcp
set -u; [ -f $SRC/COMPLETE ] || { echo "ABORT: no $SRC/COMPLETE"; exit 1; }; [ -e $DST ] && { echo "ABORT: $DST exists"; exit 1; }
t0=$(date +%s); mkdir -p $DST.part
( cd $SRC && find . -type f ! -name COMPLETE -printf '%P\n' ) | sort > $DST.list
xargs -a $DST.list -P 8 -I{} sh -c 'nice -n 10 cp -p "'$SRC'/{}" "'$DST.part'/{}"'
bad=0; while read f; do [ "$(stat -c %s "$SRC/$f")" = "$(stat -c %s "$DST.part/$f" 2>/dev/null)" ] || { echo "SIZE MISMATCH $f"; bad=1; }; done < $DST.list
[ $bad = 0 ] || { echo "VERIFY FAIL"; exit 1; }
cp -p $SRC/COMPLETE $DST.part/COMPLETE && echo "copied from $SRC, sizes == source, $(date)" > $DST.part/SWAP_OK && mv $DST.part $DST && rm -f $DST.list
echo "STAGED $(find $DST -type f | wc -l) files $(du -sh $DST | cut -f1) in $(( $(date +%s) - t0 )) s -> $DST"; df -h /scratch | tail -1

#!/bin/bash
# stage the 17850 monoplex FT export-200 node-local (COMPLETE marker last), verify every file size against the source
S=/data/home/amartyavarma/ppm/aloe-v2pure-main-17850/export-200; D=/scratch/amartyavarma/xabi-samples/w17850/export-200
mkdir -p $D; t0=$(date +%s)
ls -A $S | grep -v '^COMPLETE$' | xargs -P 24 -I{} cp -p $S/{} $D/{}
bad=0; for f in $(ls -A $S | grep -v '^COMPLETE$'); do [ "$(stat -c %s $S/$f)" = "$(stat -c %s $D/$f 2>/dev/null)" ] || { echo "SIZE MISMATCH $f"; bad=1; }; done
[ $bad = 0 ] && cp -p $S/COMPLETE $D/COMPLETE && echo "$(hostname -s) staged $(ls -A $D | wc -l) files $(du -sh $D | cut -f1) in $(( $(date +%s) - t0 )) s"

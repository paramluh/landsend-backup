#!/bin/bash
# on each DESTINATION host: copy xfer/<slot>/ -> /scratch/amartyavarma/ttconvo-ft/stores/<slot>/, verify sha256 of every file vs the source
# list, exact file set, and the s3rows load check
set -euo pipefail
source /data/home/amartyavarma/ppm/ttconvo-ft/tools/log.sh
SLOT=$1; B=/data/home/amartyavarma/ppm/ttconvo-ft; X=$B/xfer/$SLOT; L=$B/plan/sha256-$SLOT.txt; D=/scratch/amartyavarma/ttconvo-ft/stores/$SLOT
[ "$(hostname)" != "$SLOT" ] || { log "FAIL fanout $SLOT: destination is the source host"; exit 1; }
free=$(df -B1G --output=avail /scratch | tail -1 | tr -dc 0-9); need=$(( $(du -sb $X | cut -f1) / 1073741824 + 1 ))
[ $((free - need)) -ge 30 ] || { log "FAIL fanout $SLOT: /scratch free ${free}G - ${need}G < 30G"; exit 1; }
[ ! -L $D ] || { log "FAIL fanout $SLOT: $D is a symlink"; exit 1; }
rm -rf $D; mkdir -p $D
cut -c67- $L | xargs -P 8 -I{} cp --preserve=mode,timestamps $X/{} $D/{}
diff <(ls -A $D | sort) <(cut -c67- $L | sort) > /dev/null || { log "FAIL fanout $SLOT: file set differs"; exit 1; }
T=$(mktemp); sed "s#  #  $D/#" $L > $T
sha256sum -c --quiet $T || { log "FAIL fanout $SLOT: sha256 mismatch"; rm -f $T; exit 1; }
rm -f $T
out=$(/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/ttconvo-ft/tools/verify_store.py $D $SLOT $B/persona/$SLOT)
log "fanout $SLOT OK -> $D: $(wc -l < $L) files sha256 match; $out; /scratch free $(df -B1G --output=avail /scratch | tail -1 | tr -dc 0-9)G"

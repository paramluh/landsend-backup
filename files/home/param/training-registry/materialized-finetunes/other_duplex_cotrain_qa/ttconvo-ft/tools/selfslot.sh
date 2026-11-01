#!/bin/bash
# on slinky-11 / slinky-13: stores/<slot> = SYMLINK to the source store; load + persona checks via the link; slinky-11 also cross-checks b- sidecars
set -euo pipefail
source /data/home/amartyavarma/ppm/ttconvo-ft/tools/log.sh
SLOT=$(hostname); B=/data/home/amartyavarma/ppm/ttconvo-ft; S=/scratch/amartyavarma/ttconvo-ft/stores; D=$S/$SLOT; PY="/data/home/amartyavarma/venvs/qwen-s2/bin/python -I"
mkdir -p $S
if [ -L $D ]; then [ "$(readlink $D)" == /scratch/amartyavarma/tt-v3/rows ] || { log "FAIL selfslot: $D -> $(readlink $D)"; exit 1; }
elif [ -e $D ]; then log "FAIL selfslot: $D exists and is not a symlink"; exit 1
else ln -s /scratch/amartyavarma/tt-v3/rows $D; fi
out=$($PY $B/tools/verify_store.py $D $SLOT $B/persona/$SLOT)
log "selfslot $SLOT OK: $D -> $(readlink $D); $out"
if [ "$SLOT" == slinky-11 ]; then
  out=$($PY $B/tools/crosscheck_b.py $B/persona/$SLOT $(ls /scratch/amartyavarma/tt-v3/rows/ | grep words.arrow | sed 's/.words.arrow//'))
  log "selfslot $SLOT $out"
fi

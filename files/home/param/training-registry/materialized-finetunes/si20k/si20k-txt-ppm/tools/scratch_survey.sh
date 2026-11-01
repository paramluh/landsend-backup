#!/bin/bash
# per-host /scratch survey (read only): free GB, si20k stores staged, podcast FLAC count, export dirs
H=$(hostname -s)
F=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
S=$(for k in 0 1 2 3; do [ -f /scratch/amartyavarma/si20k-vox/stores/s20n$k/NODE.json ] && echo -n 1 || echo -n 0; done)
N=$(find /scratch/amartyavarma/si20k-podcast-flac -name "*.flac" 2>/dev/null | wc -l)
E=$(du -sh /scratch/amartyavarma/si20k-vox-ppm/exports /scratch/amartyavarma/si20k-vox-intl/exports 2>/dev/null | tr '\n\t' '  ')
D=$(ls -d /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-12829 2>/dev/null | wc -l)
echo "$H free=${F}G stores=$S flac=$N dcp12829=$D exports: $E"

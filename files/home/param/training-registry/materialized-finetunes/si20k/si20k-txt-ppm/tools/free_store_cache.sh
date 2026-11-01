#!/bin/bash
# si20k-txt: delete this host's node-local si20k store CACHE (/scratch/amartyavarma/si20k-vox/stores = byte copies of the kept /data stores,
# re-creatable by stage_si20k.py in ~1 min); the text-out runs read the stores from /data in place. Logs before / after free space.
source /data/home/amartyavarma/ppm/scratch_guard.sh
H=$(hostname -s); P=/scratch/amartyavarma/si20k-vox/stores
B=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
[ -d $P ] || { echo "$H: $P absent (free ${B} GB)"; exit 0; }
SZ=$(du -s --block-size=1G $P | cut -f1)
guard_check $P && guard_rm $P
A=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
echo "$H: deleted $P (${SZ} GB store cache; /data copy kept): /scratch free ${B} -> ${A} GB"

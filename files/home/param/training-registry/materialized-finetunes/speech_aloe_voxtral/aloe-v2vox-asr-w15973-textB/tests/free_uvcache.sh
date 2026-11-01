#!/bin/bash
# textB pre-flight disk: delete ONLY the uv package cache (regenerable download cache, not a checkpoint / data / protected path) on this node.
U=/scratch/amartyavarma/uv-cache; H=$(hostname -s)
[ -d "$U" ] || { echo "$H: no $U"; exit 0; }
if pgrep -u "$USER" -f '(^|/)uv( |$)' > /dev/null; then echo "$H: a uv process is running: KEEP $U"; exit 1; fi
L=$(find /scratch/amartyavarma/venvs /data/home/amartyavarma/venvs -maxdepth 7 -type l -lname "$U/*" 2>/dev/null | head -3)
[ -z "$L" ] || { echo "$H: venv symlinks into $U ($L): KEEP"; exit 1; }
B=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
source /data/home/amartyavarma/ppm/scratch_guard.sh
guard_rm "$U" || { echo "$H: scratch_guard refused $U"; exit 1; }
echo "$H: deleted $U; /scratch free ${B} -> $(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) GiB"

#!/bin/bash
#SBATCH -p all -w slinky-15 -N1 -n1 -c 1 --mem=2G -t 7-00:00:00 -J gb-ppm-watch
#SBATCH -o /data/home/amartyavarma/ppm/gb/watch-%j.log
# CPU watcher: every 5 min, each new trainer export <PPM_EXPORT_ROOT>/<run>/w-N and each new inference copy ckpt-infer/<run>/local-N
# (ckpt_local_export.sh) with COMPLETE gets 20 GenghisBench games (seeds 100-119) via gb_ppm.sh (GPU workers per its GB_PLAN). Marker: /data/home/amartyavarma/ppm/gb/<run>-w-N/.auto-launched.
#   sbatch /data/home/amartyavarma/ppm/code/gb_ppm_watch.sbatch      (one instance: it exits if another gb-ppm-watch job is running)
ROOT=${PPM_EXPORT_ROOT:-/data/home/amartyavarma/ppm/ckpt-weka}
n=$(squeue -u "$USER" -h -n gb-ppm-watch -t R -o %i | grep -vx "$SLURM_JOB_ID" | wc -l)
[ "$n" -gt 0 ] && { echo "another gb-ppm-watch is running"; exit 0; }
while true; do
  for d in "$ROOT"/*/w-* /data/home/amartyavarma/ppm/ckpt-infer/*/local-*; do
    [ -f "$d/COMPLETE" ] || continue
    tag=$(basename "$(dirname "$d")")-$(basename "$d")
    m=/data/home/amartyavarma/ppm/gb/$tag/.auto-launched
    [ -f "$m" ] && continue
    mkdir -p "$(dirname "$m")"
    echo "[$(TZ=America/Los_Angeles date '+%F %T PT')] new export $d -> GB 20 games"
    bash /data/home/amartyavarma/ppm/code/gb_ppm.sh "$d" 20 100 && date > "$m"
  done
  sleep 300
done

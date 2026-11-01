#!/bin/bash
#SBATCH -J ttconvo-hfb-watch -p all -w slinky-15 -N1 -n1 -c 1 --mem=4G --gres=none -t 7-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/ttconvo-ft/logs/hfb_watch-job-%j.out
# CPU-only HF backup watcher for the ttconvo-ft exports (hf_backup_watch.py; uploads via hf_local_exports.sh). Start: sbatch this file.
# Idles until plan/EXPORT_STEPS.txt, RUNNING_JOB.txt (a RUNNING job named ttconvo-ft) and a COMPLETE export exist; exits by itself after
# the final export is verified on all 16 hosts. Like setup-logs/hfb/hfb_watch.sbatch it resubmits itself 30 min before the time limit
# (--dependency=singleton: the new watcher starts only after this one ended; && so a scancel does not resubmit). Stop: scancel <job>.
B=/data/home/amartyavarma/ppm/ttconvo-ft
( sleep $((7*24*3600 - 1800)) && sbatch --dependency=singleton $B/hf_backup_watch.sbatch ) &
exec /data/home/amartyavarma/venvs/duplex/bin/python -u $B/hf_backup_watch.py

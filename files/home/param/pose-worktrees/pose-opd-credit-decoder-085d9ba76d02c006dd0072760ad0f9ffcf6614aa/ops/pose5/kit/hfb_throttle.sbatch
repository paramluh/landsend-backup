#!/bin/bash
#SBATCH -J hfb-throttle -p all -w slinky-4 -N1 -n1 -c1 --mem=2G -t 7-00:00:00 -o /data/home/amartyavarma/setup-logs/hfb/logs/job-throttle-%j.out
# pose5: start the HF upload step-time controller (setup-logs/hfb/hfb_throttle.py; = the 10-27 job 31153 the coordinator started by hand).
# Without it every host's control/<host>.workers stays 0 -> hfb uploads stall (10-27 ~13:00). switch_to_pose.sh starts this if none runs.
exec /data/home/amartyavarma/venvs/duplex/bin/python -u /data/home/amartyavarma/setup-logs/hfb/hfb_throttle.py

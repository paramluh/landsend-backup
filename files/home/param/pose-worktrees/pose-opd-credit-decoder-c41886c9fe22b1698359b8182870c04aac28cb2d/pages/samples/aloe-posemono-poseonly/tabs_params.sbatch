#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=none --mem=16G -t 20 -J pmpo-tabs2
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/tabs2-%j.out
P=/data/home/amartyavarma/venvs/qwen-s2/bin/python; W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/work
$P -u $W/tabs_page.py && $P -u $W/add_params.py && echo "TABS2 DONE"

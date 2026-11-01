#!/bin/bash
# runs podcast_plan_intl.py host on every ppm-next host, via srun --overlap into a running job there (CPU only, nice, small memory).
# usage: run_podcast_plan_hosts.sh "slinky-0:24475 slinky-1:24492 ..."
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
T=/data/home/amartyavarma/ppm/si20k-vox-intl/tools/podcast_plan_intl.py
L=/data/home/amartyavarma/ppm/si20k-vox-intl/plan/podcast_intl_work/logs
for hj in $1; do
  h=${hj%%:*}; j=${hj##*:}
  srun --jobid=$j --overlap -N1 -n1 -c1 --mem=8G --gres=none --cpu-bind=none -w $h nice -n 10 env OMP_NUM_THREADS=1 $PY $T host > $L/$h.log 2>&1 &
done
wait

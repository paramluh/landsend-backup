#!/bin/bash
# slinky-18 side step inside motasr 30741 (coordinator-approved shape): 6 prep shards x 4 threads, exits on STOP_S18 / timeout
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
for s in 1 2 3 4 5 6; do
  ( /data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/prep2.py train 150 3750 $s/7 --stopfile /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/STOP_S18 && /data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/prep2.py eval 60 3750 $s/7 --stopfile /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/STOP_S18 && echo "PREP_DONE_s18 shard $s" ) > /data/home/amartyavarma/ppm/agents1026/gesturemetric/judge/logs/prep_s18_$s.log 2>&1 &
done
wait
echo ALL_S18_PREP_EXIT

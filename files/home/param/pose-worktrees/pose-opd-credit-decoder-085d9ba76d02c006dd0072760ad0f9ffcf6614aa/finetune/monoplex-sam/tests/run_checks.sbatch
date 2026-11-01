#!/bin/bash
#SBATCH -J samchecks -p all -N1 -n1 -c12 --mem=30G -t 60 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/logs/checks-%j.log
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; T=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/tests/plan_sam_check.py; P=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/tests/plan
for k in 0 1 2 3 4 5 6 7; do nice $PY -I /data/home/amartyavarma/ppm/aloe-pose-ft-sam/code/verify_sam_stores.py $k > /data/home/amartyavarma/ppm/aloe-pose-ft-sam/logs/verify-p$k.log 2>&1 & done
nice $PY $T one /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env /data/home/amartyavarma/ppm/aloe-pose-ft/stores 64 $P/orig_w64.json /data/home/amartyavarma/ppm/aloe-pose-mono-8020/code > $P/orig_w64.log 2>&1 &
nice $PY $T one /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/aloe_mono8020.env /data/home/amartyavarma/ppm/aloe-pose-ft-sam/stores 64 $P/sam_w64.json > $P/sam_w64.log 2>&1 &
nice $PY $T one /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/aloe_mono8020.env /data/home/amartyavarma/ppm/aloe-pose-ft-sam/stores 96 $P/sam_w96.json > $P/sam_w96.log 2>&1 &
nice $PY $T one /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env /data/home/amartyavarma/ppm/aloe-pose-ft/stores 96 $P/orig_w96.json /data/home/amartyavarma/ppm/aloe-pose-mono-8020/code > $P/orig_w96.log 2>&1 &
wait
echo "== verify"; for k in 0 1 2 3 4 5 6 7; do tail -1 /data/home/amartyavarma/ppm/aloe-pose-ft-sam/logs/verify-p$k.log | cut -c1-400; done
echo "== orig w64 vs sam w64"; $PY $T diff $P/orig_w64.json $P/sam_w64.json
echo "== sam w64 vs sam w96"; $PY $T diff $P/sam_w64.json $P/sam_w96.json
echo "== orig w96 vs sam w96"; $PY $T diff $P/orig_w96.json $P/sam_w96.json
grep -h '"step_micros_ranks_ge64_all_dummy"\|"dummy_micros"\|"micro_layout"' $P/*_w*.json

#!/bin/bash
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly
srun -p all -w slinky-4 -N1 -n1 -c 2 --cpu-bind=none --gres=none --mem=48G -t 20 -J pmpo-rowsst \
  bash -c "export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2; \
    source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; set -a; source /data/home/amartyavarma/ppm/aloe-v2pure-ft/aloe_v2pure.env; set +a; \
    exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $W/work/build_rows_studio.py $W/rows-studio"
echo "rows-studio exit $?"

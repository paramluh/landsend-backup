#!/bin/bash
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly
srun -p all -w slinky-4 -N1 -n1 -c 2 --cpu-bind=none --gres=none --mem=40G -t 30 -J pmpo-cands \
  bash -c "export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 PPM_CD=2; \
    source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a; \
    exec /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $W/work/build_cands.py"
echo "cands exit $? $(TZ=America/Los_Angeles date '+%F %T PT')"

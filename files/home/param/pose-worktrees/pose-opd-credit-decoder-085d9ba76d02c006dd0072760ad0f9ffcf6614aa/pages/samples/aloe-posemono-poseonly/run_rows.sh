#!/bin/bash
# rows (CPU, compute node): the run's env + code -> rows/R-xxx.pkl + rows.json. Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly
srun -p all -w slinky-4 -N1 -n1 -c 4 --cpu-bind=none --gres=none --mem=48G -t 30 -J pmpo-rows \
  bash -c "export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2; \
    source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a; \
    exec nice /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $W/work/build_rows.py $W/rows"
echo "rows exit $? $(TZ=America/Los_Angeles date '+%F %T PT')"

#!/bin/bash
# candidate stick-figure previews (CPU, slinky-4): one process per candidate, all 20 in parallel; then the source-frame copy
# (on-disk app first frames only) and the page rewrite. Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly
srun -p all -w slinky-4 -N1 -n1 -c 8 --cpu-bind=none --gres=none --mem=64G -t 20 -J pmpo-skel \
  bash -c "export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PPM_CD=2; \
    source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; \
    for k in \$(seq 1 20); do nice /data/home/amartyavarma/venvs/qwen-s2/bin/python -u $W/work/cand_skel.py \$k > $W/logs/skel-\$k.log 2>&1 & done; wait; \
    flock $W/work/.cand_page.lock python3 $W/work/cand_page.py"
echo "skel exit $? $(TZ=America/Los_Angeles date '+%F %T PT')"; grep -L skeleton $W/logs/skel-*.log

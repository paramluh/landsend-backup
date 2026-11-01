#!/bin/bash
#SBATCH -J aloepose-build -p all -N1 -n1 -c 64 --mem=400G -t 2:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-ft/logs/build-%x-%j.log
# one store of the aloe-pose-ft row build (code/build_poseB_rows.py build --store $K), 64 workers on this node.
echo "== store $K on $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false S3_MIN_FRAMES=125   # = the trainer env (aloe_pose.env)
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-pose-ft/code/build_poseB_rows.py build --out /data/home/amartyavarma/ppm/aloe-pose-ft/stores --store $K --workers 64
echo "== exit $? $(TZ=America/Los_Angeles date '+%T PT')"

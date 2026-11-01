#!/bin/bash
#SBATCH -J s2p-drycfg -p all -w slinky-18 -x slinky-9 -N1 -n1 -c 4 --mem=24G -t 0-00:40:00
#SBATCH -o /data/home/amartyavarma/ppm/pose-s2p/tests/logs/dry_cfg-%j.log
# pose-s2p (e): CPU dry resolved config. (1) the ORIGINAL mono8020 code + env at world 64 (must reproduce export-357's meta cfg = the dry mode
# is faithful); (2) the branch code + s2p_mono8020.env at world 8 (+ the world-8 plan). Env exactly as the run scripts export it.
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; S=/data/home/amartyavarma/ppm/pose-s2p; O=$S/tests/out; mkdir -p $O
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8
( set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env; set +a
  export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 WORLD_SIZE=64 QQA_RUN=ppm-aloe-pose-mono-8020-j24229
  $PY -u $S/tests/dry_cfg.py /data/home/amartyavarma/ppm/aloe-pose-mono-8020/code $O/drycfg_mono8020_w64.json )
( set -a; source $S/s2p_mono8020.env; set +a
  export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 WORLD_SIZE=8 QQA_RUN=ppm-pose-s2p-mono8020-jDRY
  $PY -u $S/tests/dry_cfg.py $S/code $O/drycfg_s2p_w8.json plan )
echo "rc $? $(TZ=America/Los_Angeles date '+%T PT')"

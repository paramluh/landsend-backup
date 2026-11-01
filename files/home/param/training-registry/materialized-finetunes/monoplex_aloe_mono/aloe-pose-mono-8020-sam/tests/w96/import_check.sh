#!/bin/bash
# module-level check of the -sam trainer's env knobs (CPU import only): srun -w slinky-18 -n1 -c2 --mem=16G bash import_check.sh
set -a; source /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/aloe_mono8020.env; set +a
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/home/amartyavarma/.cache/huggingface
C=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code; P=/data/home/amartyavarma/venvs/qwen-s2/bin/python
run() {  # world norm stop lossdummy
  env RANK=0 LOCAL_RANK=0 WORLD_SIZE=$1 ALOE_NORM_WORLD=$2 ALOE_STOP_AT_STEP=$3 ALOE_LOSSCHECK_DUMMY=$4 $P -c "
import sys; sys.path.insert(0, '$C')
import train_aloe_pose as T
print('world', T.WORLD, 'NORM_W', T.NORM_W, 'STOP_AT', T.STOP_AT, 'LOSSCHECK_DUMMY', T.A.LOSSCHECK_DUMMY, 'layout', T.CFG['micro_layout'], 'rows/step', T.A.ROWS_PER_STEP)
" 2>&1 | grep -v -i warn | tail -1
}
echo "w96 recommended:   $(run 96 64 0 1)"
echo "w96 idcheck:       $(run 96 64 1 1)"
echo "w96 plain:         $(run 96 '' 0 1)"
echo "w64 defaults:      $(run 64 '' 0 0)"
echo "w96 bad NORM 50:   $(run 96 50 0 1)"

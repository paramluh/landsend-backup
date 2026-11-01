set -a; source /data/home/amartyavarma/ppm/aloe-v2-ft/aloe_v2.env; set +a
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 QQA_RUN=ppm-aloe-v2-ft-j22383 RANK=0 WORLD_SIZE=64 LOCAL_RANK=0 PPM_CD=2
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
cd /tmp && /data/home/amartyavarma/venvs/qwen-s2/bin/python -c "
import sys; sys.path.insert(0, '/data/home/amartyavarma/ppm/aloe-v2-ft/code')
import train_aloe_v2 as T
T.cfg_diff()
print(T.CFG['sched'], T.CFG['dec_sched'])
"

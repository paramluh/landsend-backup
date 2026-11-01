#!/bin/bash
# aloe-v2vox-raw: the trainer's resolved CFG (env file + launch env of a 3-node job) diffed against 22420's logged config, on CPU (import only).
set -a; source /data/home/amartyavarma/ppm/aloe-v2vox-raw/aloe_v2vox_raw.env; set +a
export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 QQA_RUN=ppm-aloe-v2vox-raw-j0 RANK=0 WORLD_SIZE=24 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 ALOE_LNODES="0,1,2;3,4,5;6,7"
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh >/dev/null 2>&1
cd /tmp
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I -c "
import sys; sys.path.insert(0, '/data/home/amartyavarma/ppm/aloe-v2vox-raw/code'); sys.path.insert(0, '/data/home/amartyavarma/ppm/aloe-v2vox-raw/code/lib')
import train_aloe_v2vox_raw as T
n, bad = T.cfg_diff_v2vox()
print('CFGDIFF', n, 'keys differ; outside allowed:', bad)
"

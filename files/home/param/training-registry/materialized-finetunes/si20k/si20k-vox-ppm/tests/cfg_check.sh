#!/bin/bash
# resolved-config diff of both trainers vs their reference runs' logged cfgs, with the REAL run env + plan/PLAN.env (CPU, import only)
for r in ppm intl; do
  Q=/data/home/amartyavarma/ppm/si20k-vox-$r
  ( set -a; source $Q/si20k_vox_$r.env; source $Q/plan/PLAN.env; set +a
    export RANK=0 WORLD_SIZE=64 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8 ALOE_LNODES="0;1;2;3;4;5;6;7" QQA_RUN=si20k-vox-$r-jCFGCHECK QQA_LOG_ROOT=/tmp/si20k-cfgcheck-$r PPM_CD=2
    export QQA_INIT=$([ $r = ppm ] && echo /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12829 || echo /data/home/amartyavarma/ppm/next-arch/ckpt-weka/ppm-next/w-6242)
    source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh >/dev/null 2>&1
    cd /tmp; /data/home/amartyavarma/venvs/qwen-s2/bin/python -I -c "
import sys; sys.path.insert(0, '$Q/code'); sys.path.insert(0, '$Q/code/lib')
import train_si20k_vox as T
n, bad = T.cfg_diff_si20k()
c = T.CFG
print('$r', 'CFGDIFF', n, 'BAD', bad, '| dec_k', c['dec_k'], 'light si20k', c['data']['light_frac'], 'podcast', c['podcast']['data']['light_frac'], '| gain', c['data']['a_gain'], 'corrupt', c['data']['a_corrupt'], 'podcast corrupt', c['podcast']['data']['a_corrupt'], '| eval', c['eval_data']['a_gain'], c['eval_data']['a_corrupt'], 'low', (c['eval_low_data'] or {}).get('a_gain'), '| steps', c['steps'], 'warmup', c['warmup'], 'init', c['init'])
" 2>&1 | grep -E "CFGDIFF|Error|error" | tail -3 )
done

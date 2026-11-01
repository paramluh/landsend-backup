#!/bin/bash
# step-0 check 3 (smoke): one full GA optimizer step (pose, speech x2, text) with nanosteps on 1 node, FSDP x 8, reduced depth
# (backbone 4 layers, towers 2, decoders 2), then checkpoint save -> fresh model -> load round trip (node-local /scratch, deleted after).
export PPM_RUN=${PPM_RUN:-ppm-smoke1} PPM_LAYERS=${PPM_LAYERS:-4} PPM_TOWER_LAYERS=${PPM_TOWER_LAYERS:-2} PPM_DEC_LAYERS=${PPM_DEC_LAYERS:-2}
export PPM_MAX_STEPS=${PPM_MAX_STEPS:-1} PPM_EVAL_EVERY=0 PPM_CKPT_ROUNDTRIP=1 PPM_SMOKE=1 PPM_DL_WORKERS=${PPM_DL_WORKERS:-2} PPM_DEC_K=${PPM_DEC_K:-25}
export PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/ckpt-smoke
bash /data/home/amartyavarma/ppm/code/run_local.sh train_ppm.py
rc=$?
rm -rf /scratch/amartyavarma/ppm/ckpt-smoke/$PPM_RUN
echo "smoke rc $rc (smoke checkpoint dir removed)"
exit $rc

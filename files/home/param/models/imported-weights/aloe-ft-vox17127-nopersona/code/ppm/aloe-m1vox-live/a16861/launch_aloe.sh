#!/bin/bash
# 10-25 (user: 'where is the Aloe FT no persona?' + 'use more backends'): "Aloe FT (no persona)" = vox17127 export-249 straight from the /data DCP
# export, 3 backends on slinky-4 GPUs 0-5 (6-7 throttled: never), server bound 0.0.0.0 for the slinky-10 front proxy (lb.json model 'aloe').
# Own front + tunnel per backend (a16861/URL_aloe<k>.txt). Submitted one at a time (first-fit GPU pairs); check the GPU ids after.
D=${ALOE_ROOT}/ppm/aloe-m1vox-live; A=$D/a16861; E=${ALOE_WEIGHTS}   # RELEASE: aloe.env (live demo: the cluster's /data DCP export-vox17127-249)
for k in ${@:-1 2 3}; do
  F=$((8975 + 10 * k)); B=$((F + 1))
  sbatch --parsable -J m1vox-aloe-$k ${NODE:+-w $NODE} -c 14 --mem=100G --gres=gpu:2 -o $D/logs/demo-aloe-$k-%j.out \
    --export=ALL,FRONT=$F,BACK=$B,PPMLIVE_STATE_DIR=$A/aloe$k,URLF=$A/URL_aloe$k.txt,CK=$E,PPMLIVE_SWAP_TO=$E,PPMLIVE_DEFAULTS=unbiased,PPMLIVE_BIND=0.0.0.0 $D/demo.sbatch
  sleep 3
done

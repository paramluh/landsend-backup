#!/bin/bash
# 10-12 dependence-probe test (codependence + xdep) on ONE spare node: prod_ppm.env + reduced depth, fresh init, eval only (max_steps 0),
# mini pose / text stores staged by tests/stage_abl_mini.py, xdep conditions incl. 'none' + PPM_ABL_SELFTEST=1 (both self-tests). No saves.
set -a; source /data/home/amartyavarma/ppm/code/prod_ppm.env; set +a
export PPM_RUN=ppm-abltest PPM_LAYERS=${PPM_LAYERS_T:-4} PPM_TOWER_LAYERS=${PPM_TOWER_LAYERS_T:-2} PPM_DEC_LAYERS=${PPM_DEC_LAYERS_T:-2} PPM_MAX_STEPS=0
export PPM_DL_WORKERS=0 PPM_RESUME_FROM= PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1 PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1
export PPM_POSE_ROOT=/data/home/amartyavarma/ppm/teststore/pose-mini PPM_TEXT_ROOT=/data/home/amartyavarma/ppm/teststore/txt-mini
export PPM_LOG_ROOT=/data/home/amartyavarma/ppm/teststore/runs PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/ckpt-abltest PPM_CONTROL_DIR=/data/home/amartyavarma/ppm/teststore/control
export PPM_EVAL_EVERY_MIN=0 PPM_SAVE_EVERY_MIN=0 PPM_EXPORT_EVERY_MIN=0 PPM_FINAL_SAVE=0
export PPM_XDEP_CONDS=${PPM_XDEP_CONDS_T:-none,silent,absent} PPM_ABL_SELFTEST=1
mkdir -p $PPM_CONTROL_DIR
bash /data/home/amartyavarma/ppm/code/run_local.sh train_ppm.py
echo "abl test rc $?"

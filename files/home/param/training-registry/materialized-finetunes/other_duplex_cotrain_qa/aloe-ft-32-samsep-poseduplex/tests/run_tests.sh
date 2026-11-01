#!/bin/bash
# POSEDUP CPU tests (run on a CPU slot of slinky-4 / slinky-18 via srun; never on the login node). Writes poseduplex/tests/out/*.json + DUMP_pose.txt
#   ref   : job 29345's data path (its code dir, vox/run_vox.env) -> out/ref_29345.json
#   p0    : poseduplex code, POSEDUP_POSE=0 (QQA_STEPS 249)          -> out/posedup_p0.json   (compare: identity)
#   p1    : poseduplex code, POSEDUP_POSE=1 (QQA_STEPS 249) + report -> out/posedup_p1.json   (compare: pose) + DUMP_pose.txt + multi-epoch checks
#   plan S: tools/plan_posedup.py for QQA_STEPS=S                    -> plans/PLAN_<digest>.json
#   dry   : trainer cfg resolution (rank 0 of world 64) + diffs vs 26232 / 26467 / 28511 / 29345 -> tests/dry/
P=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex; Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $P/tests/out $P/plans; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 NPROC=${NPROC:-16} CUDA_VISIBLE_DEVICES=
export POSEDUP_HOSTS=${POSEDUP_HOSTS:-slinky-h0,slinky-h1,slinky-h2,slinky-h3,slinky-h4,slinky-h5,slinky-h6,slinky-h7}
cd /tmp
case "$1" in
  ref)  ( set -a; source $Q/vox/run_vox.env; set +a; $PY -I -B $P/tests/identity.py $Q/code data_aloe_v2vox $P/tests/out/ref_29345.json );;
  p0)   ( set -a; source $P/run_poseduplex.env; set +a; export POSEDUP_POSE=0 QQA_STEPS=249; $PY -I -B $P/tests/identity.py $P/code data_aloe_posedup $P/tests/out/posedup_p0.json );;
  p1)   ( set -a; source $P/run_poseduplex.env; set +a; export POSEDUP_POSE=1 QQA_STEPS=${2:-498} DUMP_PATH=$P/DUMP_pose.txt
          $PY -I -B $P/tests/identity.py $P/code data_aloe_posedup $P/tests/out/posedup_p1.json report );;
  cmp)  $PY -I -B $P/tests/compare.py $P/tests/out/ref_29345.json $P/tests/out/posedup_p0.json identity
        $PY -I -B $P/tests/compare.py $P/tests/out/ref_29345.json $P/tests/out/posedup_p1.json pose;;
  plan) ( set -a; source $P/run_poseduplex.env; set +a; export QQA_STEPS=$2 ${3:+POSEDUP_ITEMS=$3}; $PY -I -B $P/tools/plan_posedup.py $P/code $P/plans );;
  dry)  ( set -a; source $P/run_poseduplex.env; set +a; export QQA_RUN=$QQA_RUN-jDRY
          for f in $P/code/train_samsep_posedup.py $P/code/data_aloe_posedup.py; do $PY -m py_compile $f && echo "compile ok $f"; done
          source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
          $PY -I -B $P/tests/cfg_dry.py $P/code $P/tests/dry );;
esac

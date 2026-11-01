#!/bin/bash
# D32 test 5 on ONE spare node (inside srun): tests/test_d32_ckpt.py phases a (head off) -> b (head on, resume a pre-D32 checkpoint) -> c (resume
# with the head) -> ckpt_local_export -> infer loaders. Tiny width, node-local /scratch only; everything deleted at the end.
O=${1:-/data/home/amartyavarma/ppm/teststore/d32}
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; R=/scratch/amartyavarma/ppm/d32test/ck
set -a; source $C/prod_ppm.env; set +a
export PPM_RUN=d32ck PPM_RUN_BASE=d32ck PPM_LAYERS=1 PPM_TOWER_LAYERS=1 PPM_DEC_LAYERS=1 PPM_DEC_FAST=0 PPM_DL_WORKERS=0 PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1
export PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1 PPM_LOG_ROOT=/scratch/amartyavarma/ppm/d32test/runs PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/d32test/ckpt
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/ppm/d32test/inductor
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf $R; mkdir -p $R
TR="$PY -m torch.distributed.run --standalone --nproc_per_node 8 $C/tests/test_d32_ckpt.py"
PPM_POSE_HEAD=0 $TR a > $O/ck_a.log 2>&1 && echo "ck_a ok" &&
PPM_POSE_HEAD=1 $TR b > $O/ck_b.log 2>&1 && echo "ck_b ok" &&
PPM_POSE_HEAD=1 $TR c > $O/ck_c.log 2>&1 && echo "ck_c ok" &&
$PY $C/ckpt_local_export.py --src $R/dcp-5 --out $R/local-5 --min-free-gb -100 > $O/ck_export.log 2>&1 &&
$PY $C/ckpt_local_export.py --out $R/local-5 --finalize >> $O/ck_export.log 2>&1 && echo "ck_export ok" &&
CUDA_VISIBLE_DEVICES=0 $PY $C/tests/test_d32_ckpt.py infer > $O/ck_infer.log 2>&1 && echo "ck_infer ok"
echo "test_d32_ckpt rc $?"
du -sh $R; rm -rf $R /scratch/amartyavarma/ppm/d32test/ckpt /scratch/amartyavarma/ppm/d32test/export

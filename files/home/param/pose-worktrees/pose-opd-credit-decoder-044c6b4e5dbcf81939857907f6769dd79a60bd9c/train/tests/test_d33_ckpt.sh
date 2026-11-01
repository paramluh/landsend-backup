#!/bin/bash
# D33 test 6 on ONE node (inside srun): tests/test_d33_ckpt.py phases a -> b -> c -> neg1 -> neg2 -> ckpt_local_export -> infer. Tiny width,
# node-local /scratch only; everything deleted at the end.
O=${1:-/data/home/amartyavarma/ppm/teststore/d33}
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; R=/scratch/amartyavarma/ppm/d33test/ck
set -a; source $C/prod_ppm.env; set +a
export PPM_RUN=d33ck PPM_RUN_BASE=d33ck PPM_LAYERS=1 PPM_TOWER_LAYERS=1 PPM_DEC_LAYERS=1 PPM_DEC_FAST=0 PPM_DL_WORKERS=0 PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1
export PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1 PPM_LOG_ROOT=/scratch/amartyavarma/ppm/d33test/runs PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/d33test/ckpt
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/ppm/d33test/inductor
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf $R; mkdir -p $R
TR="$PY -m torch.distributed.run --standalone --nproc_per_node 8 $C/tests/test_d33_ckpt.py"
PPM_DECP_BB=0 $TR a > $O/ck_a.log 2>&1 && echo "ck_a ok" &&
PPM_DECP_BB=1 $TR b > $O/ck_b.log 2>&1 && echo "ck_b ok" &&
PPM_DECP_BB=1 PPM_DECP_BB_ALPHA=0.5 $TR c > $O/ck_c.log 2>&1 && echo "ck_c ok" &&
PPM_DECP_BB=0 $TR neg1 > $O/ck_neg1.log 2>&1 && echo "ck_neg1 ok" &&
PPM_DECP_BB=1 PPM_DECP_BB_ALPHA=0.5 PPM_LR_DECP_BB=3e-4 $TR neg2 > $O/ck_neg2.log 2>&1 && echo "ck_neg2 ok" &&
$PY $C/ckpt_local_export.py --src $R/dcp-5 --out $R/local-5 --min-free-gb -100 > $O/ck_export.log 2>&1 &&
$PY $C/ckpt_local_export.py --out $R/local-5 --finalize >> $O/ck_export.log 2>&1 && echo "ck_export ok" &&
CUDA_VISIBLE_DEVICES=0 $PY $C/tests/test_d33_ckpt.py infer > $O/ck_infer.log 2>&1 && echo "ck_infer ok"
echo "test_d33_ckpt rc $?"
cp $R/*.json $O/ 2>/dev/null; du -sh $R; rm -rf $R /scratch/amartyavarma/ppm/d33test/ckpt

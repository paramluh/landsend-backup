#!/bin/bash
# D33 GPU tests on ONE node (inside srun): prod_ppm.env + reduced depth, pre-dumped real micros (tests/dump_d33_micros.py), fresh init, no saves.
# usage: bash tests/test_d33.sh <code dir> <mode grads|align|learn> <out.json>   (env: PPM_DECP_BB, PPM_DECP_BB_ALPHA, D33_NSTEP, D33_FAST=1 for
# the production compiled decoder path + real FA3 (learn / timing), else the deterministic eager path)
_BB=${PPM_DECP_BB:-0}; _AL=${PPM_DECP_BB_ALPHA:-1.0}                       # the caller's D33 settings win over prod_ppm.env's
set -a; source /data/home/amartyavarma/ppm/code/prod_ppm.env; set +a
export PPM_DECP_BB=$_BB PPM_DECP_BB_ALPHA=$_AL
if [ "${D33_FAST:-0}" = "1" ]; then export D33_DET=0; else export D33_DET=1 PPM_DEC_FAST=0 PPM_DEC_ATTN=hf; fi
export PPM_RUN=ppm-d33test PPM_LAYERS=${PPM_LAYERS_T:-2} PPM_TOWER_LAYERS=${PPM_TOWER_LAYERS_T:-2} PPM_DEC_LAYERS=${PPM_DEC_LAYERS_T:-2}
export PPM_DL_WORKERS=0 PPM_RESUME_FROM= PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1 PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1
export PPM_LOG_ROOT=/scratch/amartyavarma/ppm/d33test/runs PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/d33test/ckpt PPM_CONTROL_DIR=/scratch/amartyavarma/ppm/d33test/control
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/ppm/d33test/inductor
mkdir -p $PPM_LOG_ROOT $PPM_CONTROL_DIR
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -m torch.distributed.run --standalone --nproc_per_node ${NPROC:-8} /data/home/amartyavarma/ppm/code/tests/test_d33.py "$1" "$2" "$3"
echo "test_d33 rc $? ($1 $2 decp_bb=$PPM_DECP_BB alpha=${PPM_DECP_BB_ALPHA:-1} fast=${D33_FAST:-0})"

#!/bin/bash
# D32 GPU tests on ONE spare node (slinky-7): prod_ppm.env + reduced depth + this node's own stores, fresh init, no saves.
# usage (inside srun on the node): bash tests/test_d32.sh <code dir> <mode grads|align> <out.json> [PPM_POSE_HEAD 0|1]
set -a; source /data/home/amartyavarma/ppm/code/prod_ppm.env; set +a
export PPM_POSE_HEAD=${4:-0}
# determinism (test only): the compiled T25 decoder path (dynamic-shape inductor graph + fused attn32 kernel) has a run-to-run nondeterministic
# backward; the bit-for-bit comparisons use the eager HF decoder path (same maths; the D32 change touches no decoder code)
export PPM_DEC_FAST=${PPM_DEC_FAST_T:-0} PPM_DEC_ATTN=${PPM_DEC_ATTN_T:-hf}
export PPM_RUN=ppm-d32test PPM_LAYERS=${PPM_LAYERS_T:-2} PPM_TOWER_LAYERS=${PPM_TOWER_LAYERS_T:-2} PPM_DEC_LAYERS=${PPM_DEC_LAYERS_T:-2}
export PPM_DL_WORKERS=0 PPM_RESUME_FROM= PPM_EXTRA_PLAN= PPM_EXTRA_FROM_STEP=-1 PPM_TRAIN_EXCLUDE_SALTS= PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1
export PPM_LOG_ROOT=/scratch/amartyavarma/ppm/d32test/runs PPM_SAVE_ROOT=/scratch/amartyavarma/ppm/d32test/ckpt PPM_CONTROL_DIR=/scratch/amartyavarma/ppm/d32test/control
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/ppm/d32test/inductor
mkdir -p $PPM_LOG_ROOT $PPM_CONTROL_DIR
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -m torch.distributed.run --standalone --nproc_per_node ${NPROC:-8} /data/home/amartyavarma/ppm/code/tests/test_d32.py "$1" "$2" "$3"
echo "test_d32 rc $? ($1 $2 head=$PPM_POSE_HEAD)"

#!/bin/bash
# one task of ppm.sbatch's srun: rank = 8 x (host index in PPM_HOSTS) + SLURM_LOCALID.
H=$(hostname -s)
IDX=$(echo "$PPM_HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
[ -n "$IDX" ] || { echo "ABORT: $H not in PPM_HOSTS"; exit 1; }
NN=$(echo "$PPM_HOSTS" | tr , '\n' | wc -l)
export RANK=$((8 * (IDX - 1) + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$((8 * NN)) LOCAL_WORLD_SIZE=8
export MASTER_ADDR=$(echo "$PPM_HOSTS" | cut -d, -f1) MASTER_PORT=${MASTER_PORT:-29815}
export HF_HOME=${ALOE_ROOT}/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none} NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${ALOE_SCRATCH}/triton-cache-ppm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${ALOE_SCRATCH}/inductor-cache-ppm/$H}
source ${ALOE_ROOT}/qwen-s2-imend/hera3/env.sh
exec ${ALOE_ROOT}/venvs/qwen-s2/bin/python "$@"

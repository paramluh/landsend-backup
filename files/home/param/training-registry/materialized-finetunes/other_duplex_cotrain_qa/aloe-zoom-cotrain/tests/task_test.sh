#!/bin/bash
# TEST task wrapper (= code/task_ppm.sh with the ranks per node from the allocation instead of a fixed 8).
H=$(hostname -s)
IDX=$(echo "$PPM_HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
[ -n "$IDX" ] || { echo "ABORT: $H not in PPM_HOSTS"; exit 1; }
NN=$(echo "$PPM_HOSTS" | tr , '\n' | wc -l)
LWN=${SLURM_NTASKS_PER_NODE:?}
export RANK=$((LWN * (IDX - 1) + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$((LWN * NN)) LOCAL_WORLD_SIZE=$LWN
export MASTER_ADDR=$(echo "$PPM_HOSTS" | cut -d, -f1) MASTER_PORT=${MASTER_PORT:-29883}
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm
export TORCHINDUCTOR_CACHE_DIR=/scratch/amartyavarma/inductor-cache-ppm/$H
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
exec nice -n 5 /data/home/amartyavarma/venvs/qwen-s2/bin/python "$@"

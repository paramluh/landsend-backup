#!/bin/bash
# one task of ppm.sbatch's srun: rank = 8 x (host index in PPM_HOSTS) + SLURM_LOCALID.
H=$(hostname -s)
IDX=$(echo "$PPM_HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
[ -n "$IDX" ] || { echo "ABORT: $H not in PPM_HOSTS"; exit 1; }
NN=$(echo "$PPM_HOSTS" | tr , '\n' | wc -l)
export RANK=$((8 * (IDX - 1) + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$((8 * NN)) LOCAL_WORLD_SIZE=8
export MASTER_ADDR=$(echo "$PPM_HOSTS" | cut -d, -f1) MASTER_PORT=${MASTER_PORT:-29815}
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none} NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/amartyavarma/triton-cache-ppm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/scratch/amartyavarma/inductor-cache-ppm/$H}
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
# 10-19 hang diagnostics (no maths / data effect): faulthandler on SIGUSR2 + py-spy allowed (ppm/diag/sitecustomize.py); NCCL flight recorder
# (last 20000 collectives per rank), dumped per rank on a collective timeout to restore-1019/hang/nccl_fr/rank_<r>
export PYTHONPATH=/data/home/amartyavarma/ppm/diag${PYTHONPATH:+:$PYTHONPATH} TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000} \
  TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1} TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-/data/home/amartyavarma/ppm/restore-1019/hang/nccl_fr/rank_}
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python "$@"

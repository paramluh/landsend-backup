#!/bin/bash
# one task of ppm.sbatch's srun: rank = 8 x (host index in PPM_HOSTS) + SLURM_LOCALID.
# aloe-v2pure-17352 (10-22): + an optional UNEVEN launch layout (both env unset = 26089's task_ppm.sh exactly, backup task_ppm.sh.bak-26089):
#   PPM_HOST_NTASKS = per-host task counts aligned with PPM_HOSTS (e.g. 6,8,6): rank = the counts of the hosts before this one + SLURM_LOCALID,
#   WORLD_SIZE = their sum, LOCAL_WORLD_SIZE = this host's count. PPM_HOST_GPUS = 'host:i,j,..;host:..' physical GPU list per host ->
#   CUDA_VISIBLE_DEVICES (this cluster's Slurm sets none; slinky-18 GPUs 0-1 = the user's live demo, slinky-4 GPUs 6-7 throttled).
#   Launch layout only: the FSDP world is flat (1-D) and the v2pure plan spreads each step's 64 rows over WORLD x R micros (maths-neutral).
H=$(hostname -s)
IDX=$(echo "$PPM_HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
[ -n "$IDX" ] || { echo "ABORT: $H not in PPM_HOSTS"; exit 1; }
NN=$(echo "$PPM_HOSTS" | tr , '\n' | wc -l)
if [ -n "${PPM_HOST_NTASKS:-}" ]; then
  mapfile -t CNT < <(echo "$PPM_HOST_NTASKS" | tr , '\n')
  [ "${#CNT[@]}" = "$NN" ] || { echo "ABORT: PPM_HOST_NTASKS $PPM_HOST_NTASKS has ${#CNT[@]} entries, PPM_HOSTS $NN"; exit 1; }
  OFF=0; for ((i = 0; i < IDX - 1; i++)); do OFF=$((OFF + CNT[i])); done
  W=0; for c in "${CNT[@]}"; do W=$((W + c)); done
  LW=${CNT[$((IDX - 1))]}
  [ "$SLURM_LOCALID" -lt "$LW" ] || { echo "ABORT: $H local task $SLURM_LOCALID >= its count $LW"; exit 1; }
  export RANK=$((OFF + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$W LOCAL_WORLD_SIZE=$LW
else
  export RANK=$((8 * (IDX - 1) + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$((8 * NN)) LOCAL_WORLD_SIZE=8
fi
if [ -n "${PPM_HOST_GPUS:-}" ]; then
  G=$(echo "$PPM_HOST_GPUS" | tr ';' '\n' | grep "^$H:" | cut -d: -f2)
  [ -n "$G" ] || { echo "ABORT: $H has no entry in PPM_HOST_GPUS"; exit 1; }
  [ "$(echo "$G" | tr , '\n' | wc -l)" = "$LOCAL_WORLD_SIZE" ] || { echo "ABORT: $H GPUs $G != $LOCAL_WORLD_SIZE local ranks"; exit 1; }
  export CUDA_VISIBLE_DEVICES=$G
fi
export MASTER_ADDR=$(echo "$PPM_HOSTS" | cut -d, -f1) MASTER_PORT=${MASTER_PORT:-29815}
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none} NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/scratch/amartyavarma/triton-cache-ppm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/scratch/amartyavarma/inductor-cache-ppm/$H}
source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh
exec /data/home/amartyavarma/venvs/qwen-s2/bin/python "$@"

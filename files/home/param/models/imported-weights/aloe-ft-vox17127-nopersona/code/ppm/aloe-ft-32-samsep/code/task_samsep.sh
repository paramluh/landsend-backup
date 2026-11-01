#!/bin/bash
# aloe-ft-32-samsep: one task of run.sbatch's per-host srun step = task_ppm.sh with UNEVEN hosts: PPM_HOST_NRANKS (ranks per host, in PPM_HOSTS
# order) and PPM_HOST_GPUS (';'-separated CUDA_VISIBLE_DEVICES per host, e.g. "0,1,2,3,4,5,6,7;0,1,2,3,4,5": slinky-4's throttled GPUs 6-7
# are never visible). rank = (ranks of the hosts before this one) + SLURM_LOCALID; LOCAL_RANK indexes this host's visible GPUs.
H=$(hostname -s)
IDX=$(echo "$PPM_HOSTS" | tr , '\n' | grep -nx "$H" | cut -d: -f1)
[ -n "$IDX" ] || { echo "ABORT: $H not in PPM_HOSTS"; exit 1; }
IFS=, read -ra NR <<< "$PPM_HOST_NRANKS"
IFS=';' read -ra GP <<< "$PPM_HOST_GPUS"
[ ${#NR[@]} = $(echo "$PPM_HOSTS" | tr , '\n' | wc -l) ] && [ ${#GP[@]} = ${#NR[@]} ] || { echo "ABORT: PPM_HOST_NRANKS / PPM_HOST_GPUS vs PPM_HOSTS"; exit 1; }
BASE=0; for ((i = 0; i < IDX - 1; i++)); do BASE=$((BASE + NR[i])); done
W=0; for n in "${NR[@]}"; do W=$((W + n)); done
N=${NR[$((IDX - 1))]}
[ "$SLURM_LOCALID" -lt "$N" ] || { echo "ABORT: $H local id $SLURM_LOCALID >= $N ranks"; exit 1; }
[ $(echo "${GP[$((IDX - 1))]}" | tr , '\n' | wc -l) = "$N" ] || { echo "ABORT: $H GPU list ${GP[$((IDX - 1))]} != $N ranks"; exit 1; }
export CUDA_VISIBLE_DEVICES=${GP[$((IDX - 1))]}
export RANK=$((BASE + SLURM_LOCALID)) LOCAL_RANK=$SLURM_LOCALID WORLD_SIZE=$W LOCAL_WORLD_SIZE=$N
export MASTER_ADDR=$(echo "$PPM_HOSTS" | cut -d, -f1) MASTER_PORT=${MASTER_PORT:-29815}
export HF_HOME=${ALOE_ROOT}/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5} NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none} NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4} NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-16}
export CUDA_DEVICE_MAX_CONNECTIONS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${ALOE_SCRATCH}/triton-cache-ppm}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${ALOE_SCRATCH}/inductor-cache-ppm/$H}
export TORCH_NCCL_TRACE_BUFFER_SIZE=${TORCH_NCCL_TRACE_BUFFER_SIZE:-2000} TORCH_NCCL_DUMP_ON_TIMEOUT=1 TORCH_NCCL_DEBUG_INFO_TEMP_FILE=${ALOE_SCRATCH}/aloe-ft-32-samsep/nccl_trace_r
source ${ALOE_ROOT}/qwen-s2-imend/hera3/env.sh
exec ${ALOE_ROOT}/venvs/qwen-s2/bin/python "$@"

#!/bin/bash
# PPM inference wrapper: runs infer_ppm.py on ONE GPU node via srun (default slinky-15, 2 GPUs: ~84 GB bf16 weights + caches).
#   bash infer_ppm.sh --ckpt <ckpt> --task speech|pose|text [infer_ppm.py options]
# env: PPM_INFER_NODE (slinky-15), PPM_INFER_GPUS (2), PPM_INFER_TIME (minutes, 90), PPM_INFER_MEM (160G), PPM_INFER_JOBID (run as an
#      --overlap step inside an existing allocation instead of a new one; then PPM_INFER_DEVS picks the GPUs, e.g. 6,7)
set -uo pipefail
C=/data/home/amartyavarma/ppm/code
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
NODE=${PPM_INFER_NODE:-slinky-15}; NG=${PPM_INFER_GPUS:-2}
DEVS=${PPM_INFER_DEVS:-$(seq -s, 0 $((NG - 1)))}
ENVS="source /data/home/amartyavarma/qwen-s2-imend/hera3/env.sh; export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PPM_CD=2 GEMM_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-ppm-infer"
ARGS=$(printf ' %q' "$@")
PTY=""; case " $* " in *" --interactive "*) PTY="--pty";; esac      # live chat needs a terminal
if [ -n "${PPM_INFER_JOBID:-}" ]; then
  exec srun $PTY --jobid "$PPM_INFER_JOBID" --overlap -N1 -n1 -w "$NODE" -c 8 --cpu-bind=none --mem=${PPM_INFER_MEM:-160G} -J ppm-infer \
    bash -c "$ENVS; exec $PY -u $C/infer_ppm.py --devs $DEVS $ARGS"
fi
exec srun $PTY -p all -w "$NODE" -N1 -n1 -c 8 --gres=gpu:$NG --mem=${PPM_INFER_MEM:-160G} -t ${PPM_INFER_TIME:-90} -J ppm-infer \
  bash -c "$ENVS; exec $PY -u $C/infer_ppm.py --devs $DEVS $ARGS"

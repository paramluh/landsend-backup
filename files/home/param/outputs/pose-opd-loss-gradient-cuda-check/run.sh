#!/bin/bash
set -euo pipefail
cd /home/param/runs/pose-runs/loss-gradient-cuda-smoke-1793463266
exec 9>/home/param/runs/pose-runs/.duplex-distill-gpus.lock
flock -n 9
if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
  echo 'Refusing occupied node'; exit 74
fi
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
set +e
timeout --signal=TERM --kill-after=10s 110s /home/param/pose-repo/.venv/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m finetune.pose_opd_credit.tests.loss_gradient_cuda_check
code=$?
echo "$code" > exit-code.txt
exit "$code"

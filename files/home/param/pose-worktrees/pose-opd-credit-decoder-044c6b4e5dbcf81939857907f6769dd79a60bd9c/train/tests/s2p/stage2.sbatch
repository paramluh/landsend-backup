#!/bin/bash
#SBATCH -J pose5-s2p-t2 -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 2 --mem=2900M -t 0-02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p/logs/stage2-%j.log
# pose5 s2p tests, stage 2 (CPU only): (c) resume / LR group / AdamW / strictness phases A..I of test_resume.py (tiny FSDP2 model, real trainer
# save + load on node-local /scratch/amartyavarma/pose5-s2p-test/ckpt; needs stage 1's micros on the same node).
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
N="nice -n 19"
export S2P_RESUME_RUN=$SLURM_JOB_ID                           # synthetic checkpoints -> SCR/ckpt/run-<job>/<phase> (node-local)
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job $SLURM_JOB_ID"
i=0
ON=${S2P_ON:-1}; OTHER=cb0; [ "$ON" = cb0 ] && OTHER=1                     # S2P_ON=cb0 (+ S2P_RESUME_TAG=_cb0): the same suite for PPM_S2P=cb0
for ph in "A ORIG 0" "B COPY 0" "C COPY $ON" "D COPY $ON" "E COPY 0" "F COPY $OTHER" "G COPY $ON" "H COPY $ON" "I ORIG 0" "J COPY 0" "K COPY $ON" "L COPY $ON"; do
  set -- $ph; i=$((i + 1))
  echo "== phase $1 ($2, PPM_S2P=$3)"
  PPM_S2P=$3 $N $PY $TD/test_resume.py --phase $1 --code $2 --port $((29950 + i)) 2>&1 | grep -v "^\[rank0\]:\[W" ; echo "   rc ${PIPESTATUS[0]}"
done
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

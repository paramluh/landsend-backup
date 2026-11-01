#!/bin/bash
#SBATCH -J pose5-g0-eval -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 2 --mem=6G -t 0-01:30:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0/logs/eval-%j.log
# pose5 D, audit S9: the trainer's own evaluate() (start eval) on CPU, tiny config, real eval micros of every family (test_eval.py):
# ORIG flag off (base) -> merged copy flag off, G0COND + S2P cb0, G0COND + S2P full (zero init): no assert, every metric == base.
set -uo pipefail
export G0CODE=${G0CODE:-/data/home/amartyavarma/ppm/pose5/code-811-s2p}; export G0OUT=${G0OUT:-$G0CODE/tests/g0/out}; mkdir -p $G0OUT
TD=/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0; OUT=$G0OUT; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P PPM_G0COND
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
export S2P_TEST_CE_CHUNK=${S2P_TEST_CE_CHUNK:-256}
N="nice -n 19"
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID:-none} code $G0CODE"
$N $PY $TD/test_eval.py --code ORIG --g0 0 --s2p none --out $OUT/eval_orig_off.json --port 29881 || exit 1
for v in "0 none eval_off 29882" "1 cb0 eval_g0cb0 29883" "1 full eval_g0full 29884"; do
  set -- $v
  $N $PY $TD/test_eval.py --code COPY --g0 $1 --s2p $2 --out $OUT/$3.json --base $OUT/eval_orig_off.json --port $4 || echo "!! $3 FAILED (rc $?)"
done
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

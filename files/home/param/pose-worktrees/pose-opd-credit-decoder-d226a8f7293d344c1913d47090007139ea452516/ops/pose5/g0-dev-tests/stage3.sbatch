#!/bin/bash
#SBATCH -J pose5-g0-t3 -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 2 --mem=56G -t 0-06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0/logs/stage3-%j.log
# pose5 workstream D (G0COND) tests, stage 3 (CPU only, NOT YET RUN when written: needs ~56 GB on a spare): REAL widths + REAL 8,1,1 weights
# (step 17641, reduced depth) on the real micros of every row type: ORIG flag off vs COPY g0 zero (identical) and g0 nz (pose-only).
# Weight cache: workstream A's (same file format / tensors; real_weights.py) if present on this node, else built here (node-local scratch).
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0; OUT=${G0OUT:-$TD/out}; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P PPM_G0COND
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
N="nice -n 19"; LB=${LB:-2}; LT=${LT:-2}; LD=${LD:-1}; MIC=${MIC:-pose_full,pose_frozen,speech,gemma,chat}
WA=/scratch/amartyavarma/pose5-s2p-test/real_w_${LB}_${LT}_${LD}.pt
WC=/scratch/amartyavarma/pose5-g0-test/real_w_${LB}_${LT}_${LD}.pt
[ -f $WA ] && WC=$WA
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID:-none}; lb $LB lt $LT ld $LD; micros $MIC; weights $WC"
[ -f $WC ] || $N $PY /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p/real_weights.py $WC $LB $LT $LD || exit 1
i=0
for v in "ORIG 0 real_orig_off" "COPY 1 real_copy_on" "COPY nz real_copy_nz"; do
  set -- $v; i=$((i + 1))
  $N $PY $TD/run_variant.py --code $1 --g0 $2 --model real --lb $LB --lt $LT --ld $LD --wcache $WC --micros $MIC --port $((29830 + i)) --out $OUT/$3.json || exit 1
done
$PY $TD/compare.py var $OUT/real_orig_off.json $OUT/real_copy_on.json g0_zero $OUT/CMP_real_orig_vs_on0.json
$PY $TD/compare.py var $OUT/real_orig_off.json $OUT/real_copy_nz.json g0_nz $OUT/CMP_real_orig_vs_nz.json
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

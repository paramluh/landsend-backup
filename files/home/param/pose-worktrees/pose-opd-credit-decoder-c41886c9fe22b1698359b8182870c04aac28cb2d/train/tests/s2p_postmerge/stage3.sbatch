#!/bin/bash
#SBATCH -J pose5-s2p-t3 -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 4 --mem=56G -t 0-06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge/logs/stage3-%j.log
# pose5 s2p tests, stage 3 (CPU only): (b-ii) REAL widths + REAL 8,1,1 weights (step 17641, reduced depth) on the real micros of every row
# type: ORIG flag off vs COPY full / cb0 (zero s_emb). Weight cache + micros on node-local /scratch/amartyavarma/pose5-s2p-postmerge-test.
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p_postmerge; OUT=$TD/out; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
N="nice -n 19"; LB=${LB:-2}; LT=${LT:-2}; LD=${LD:-1}; MIC=${MIC:-pose_full,pose_frozen,speech,gemma,chat}
WC=/scratch/amartyavarma/pose5-s2p-postmerge-test/real_w_${LB}_${LT}_${LD}.pt
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job $SLURM_JOB_ID; lb $LB lt $LT ld $LD; micros $MIC"
[ -f $WC ] || $N $PY $TD/real_weights.py $WC $LB $LT $LD || exit 1
i=0
for v in "ORIG off" "COPY full" "COPY cb0"; do
  set -- $v; i=$((i + 1)); tag=real_$(echo $1 | tr A-Z a-z)_$2
  $N $PY $TD/run_variant.py --code $1 --variant $2 --model real --lb $LB --lt $LT --ld $LD --wcache $WC --micros $MIC --port $((29930 + i)) --out $OUT/$tag.json || exit 1
done
$PY $TD/compare.py var $OUT/real_orig_off.json $OUT/real_copy_full.json s2p_zero $OUT/CMP_real_orig_vs_full0.json
$PY $TD/compare.py var $OUT/real_orig_off.json $OUT/real_copy_cb0.json s2p_zero $OUT/CMP_real_orig_vs_cb00.json
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

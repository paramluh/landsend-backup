#!/bin/bash
#SBATCH -J pose5-s2p-t1 -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 2 --mem=2900M -t 0-03:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p/logs/stage1-%j.log
# pose5 s2p tests, stage 1 (CPU only, no GPU requested; any spare node (slinky-4 / 10 / 18; later stages pin to it), nice): (a1) resolved cfg + real-size param table, ORIG vs
# COPY flag off / full / cb0; real micros (build_micros.py); (a2, b-i, optional nz) tiny-config training micros through run_micro.
# Writes only: tests/s2p/{out,logs} (small json / logs) and node-local /scratch/amartyavarma/pose5-s2p-test (micros).
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p; OUT=$TD/out; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0
N="nice -n 19"
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job $SLURM_JOB_ID"
STEP=${STEP:-all}
if [ "$STEP" = all ] || [ "$STEP" = cfg ]; then
  echo "== (a1) resolved cfg + param table"
  WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py ORIG $OUT/cfg_orig.json || exit 1
  WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py COPY $OUT/cfg_copy_off.json || exit 1
  PPM_S2P=1 WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py COPY $OUT/cfg_copy_full.json || exit 1
  PPM_S2P=cb0 WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py COPY $OUT/cfg_copy_cb0.json || exit 1
  $PY $TD/compare.py cfg $OUT/cfg_orig.json $OUT/cfg_copy_off.json $OUT/cfg_copy_full.json $OUT/cfg_copy_cb0.json $OUT/CMP_cfg.json
fi
if [ "$STEP" = all ] || [ "$STEP" = micros ]; then
  echo "== real micros"
  WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/build_micros.py || exit 1
fi
if [ "$STEP" = all ] || [ "$STEP" = tiny ]; then
  echo "== (a2 / b-i) tiny-config training micros"
  i=0
  for v in "ORIG off" "ORIG off" "COPY off" "COPY full" "COPY cb0" "COPY full_nz" "COPY cb0_nz"; do
    set -- $v; i=$((i + 1)); tag=tiny${TAGSFX:-}_$(echo $1 | tr A-Z a-z)_$2; [ $i = 2 ] && tag=tiny${TAGSFX:-}_orig_off_rerun
    WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/run_variant.py --code $1 --variant $2 --model tiny --port $((29910 + i)) --out $OUT/$tag.json || exit 1
  done
  $PY $TD/compare.py var $OUT/tiny${TAGSFX:-}_orig_off.json $OUT/tiny${TAGSFX:-}_orig_off_rerun.json same $OUT/CMP_tiny${TAGSFX:-}_orig_vs_orig.json
  $PY $TD/compare.py var $OUT/tiny${TAGSFX:-}_orig_off.json $OUT/tiny${TAGSFX:-}_copy_off.json same $OUT/CMP_tiny${TAGSFX:-}_orig_vs_copyoff.json
  $PY $TD/compare.py var $OUT/tiny${TAGSFX:-}_orig_off.json $OUT/tiny${TAGSFX:-}_copy_full.json s2p_zero $OUT/CMP_tiny${TAGSFX:-}_orig_vs_full0.json
  $PY $TD/compare.py var $OUT/tiny${TAGSFX:-}_orig_off.json $OUT/tiny${TAGSFX:-}_copy_cb0.json s2p_zero $OUT/CMP_tiny${TAGSFX:-}_orig_vs_cb00.json
  $PY $TD/compare.py var $OUT/tiny${TAGSFX:-}_orig_off.json $OUT/tiny${TAGSFX:-}_copy_full_nz.json s2p_nz $OUT/CMP_tiny${TAGSFX:-}_orig_vs_fullnz.json
  $PY $TD/compare.py var $OUT/tiny${TAGSFX:-}_orig_off.json $OUT/tiny${TAGSFX:-}_copy_cb0_nz.json s2p_nz $OUT/CMP_tiny${TAGSFX:-}_orig_vs_cb0nz.json
fi
echo "== done rc 0 $(TZ=America/Los_Angeles date '+%T PT')"

#!/bin/bash
#SBATCH -J pose5-g0-t1 -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 2 --mem=8G -t 0-03:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0/logs/stage1-%j.log
# pose5 workstream D (G0COND) tests, stage 1, CPU only (no GPU requested), nice 19, on a spare node (slinky-4 / 10 / 18):
#   cfg (resolved cfg + real-size param table: ORIG vs COPY off vs COPY on) | micros (real micros, byte checks) | fstats (f on real clips)
#   | tiny (tiny-config 8,1,1 training micros: ORIG off, COPY off, COPY on (zero), COPY nz) | resume (phases A..I) | compare.
# Writes only tests/g0/{out,logs}, pose5/g0-test-micros (small pickles) and node-local /scratch/amartyavarma/pose5-g0-test (test ckpts).
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0; OUT=${G0OUT:-$TD/out}; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P PPM_G0COND
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0
N="nice -n 19"
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID:-none} code ${G0CODE:-code-811-g0}"
ls -la /scratch/amartyavarma/pose5-s2p-test/ 2>/dev/null | head; ls -d /scratch/amartyavarma/pose-s3-samdata* 2>/dev/null
STEPS=${STEPS:-cfg,micros,fstats,tiny,resume}
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
if has cfg; then
  echo "== (a1) resolved cfg + param table $(TZ=America/Los_Angeles date '+%T PT')"
  WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py ORIG $OUT/cfg_orig.json || exit 1
  WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py COPY $OUT/cfg_copy_off.json || exit 1
  PPM_G0COND=1 WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py COPY $OUT/cfg_copy_on.json || exit 1
  $PY $TD/compare.py cfg $OUT/cfg_orig.json $OUT/cfg_copy_off.json $OUT/cfg_copy_on.json $OUT/CMP_cfg.json
fi
if has micros; then
  echo "== real micros $(TZ=America/Los_Angeles date '+%T PT')"
  WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/build_micros.py || exit 1
fi
if has fstats; then
  echo "== f on real clips $(TZ=America/Los_Angeles date '+%T PT')"
  WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 G0_MAX_FILES=${G0_MAX_FILES:-150} $N $PY $TD/f_stats.py || echo "!! f_stats rc $?"
fi
if has tiny; then
  echo "== (a2 / b-i) tiny-config training micros $(TZ=America/Los_Angeles date '+%T PT')"
  i=0
  for v in "ORIG 0 tiny_orig_off" "COPY 0 tiny_copy_off" "COPY 1 tiny_copy_on" "COPY nz tiny_copy_nz"; do
    set -- $v; i=$((i + 1))
    WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/run_variant.py --code $1 --g0 $2 --model tiny --port $((29810 + i)) --out $OUT/$3.json || exit 1
  done
  $PY $TD/compare.py var $OUT/tiny_orig_off.json $OUT/tiny_copy_off.json same $OUT/CMP_tiny_orig_vs_copyoff.json
  $PY $TD/compare.py var $OUT/tiny_orig_off.json $OUT/tiny_copy_on.json g0_zero $OUT/CMP_tiny_orig_vs_on0.json
  $PY $TD/compare.py var $OUT/tiny_orig_off.json $OUT/tiny_copy_nz.json g0_nz $OUT/CMP_tiny_orig_vs_nz.json
fi
if has nzorder; then                                               # non-zero inp.g0: pose-only effect. Speech / text micros FIRST (the harness
  echo "== (b-iii) non-zero g0, speech/text first $(TZ=America/Los_Angeles date '+%T PT')"   # carries decoder nanostep state from micro to micro)
  MO=speech,gemma,chat,pose_full,pose_frozen
  WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/run_variant.py --code ORIG --g0 0 --model tiny --micros $MO --port 29821 --out $OUT/tiny_orig_off_stfirst.json || exit 1
  WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/run_variant.py --code COPY --g0 nz --model tiny --micros $MO --port 29822 --out $OUT/tiny_copy_nz_stfirst.json || exit 1
  $PY $TD/compare.py var $OUT/tiny_orig_off_stfirst.json $OUT/tiny_copy_nz_stfirst.json g0_nz $OUT/CMP_tiny_orig_vs_nz_stfirst.json
fi
if has resume; then
  echo "== (c) resume / LR group / strictness $(TZ=America/Los_Angeles date '+%T PT')"
  export WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
  i=0
  for ph in ${PHASES:-A:ORIG:0 B:COPY:0 C:COPY:1 D:COPY:1 E:COPY:0 G:COPY:1 H:COPY:1 I:ORIG:0}; do
    set -- ${ph//:/ }; i=$((i + 1))
    echo "== phase $1 ($2, PPM_G0COND=$3)"
    PPM_G0COND=$3 $N $PY $TD/test_resume.py --phase $1 --code $2 --port $((29850 + i)) 2>&1 | grep -v "^\[rank0\]:\[W" ; echo "   rc ${PIPESTATUS[0]}"
  done
  du -sh /scratch/amartyavarma/pose5-g0-test/ckpt 2>/dev/null       # synthetic tiny test checkpoints (node-local; kept: no-deletion rule)
fi
echo "== done rc 0 $(TZ=America/Los_Angeles date '+%T PT')"

#!/bin/bash
#SBATCH -J pose5-g0-cmb -p all -x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19 -N1 -n1 -c 2 --mem=8G -t 0-01:30:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0/logs/combo-%j.log
# pose5 D: COMBINED G0COND + POSE-S2P proofs on the MERGED copy (G0CODE, default code-811-s2p), CPU only, nice 19. STEPS (comma list):
#   cfg       resolved cfg + real-size params: ORIG | off | g0 | cb0 | full | g0+cb0 | g0+full  -> compare2 cfg
#   tA..tE    tiny-config 8,1,1 training micros on the real micros (zero-init variants vs ORIG; non-zero variants speech/text first)
#   resume    test_resume_combo phases: A (ORIG stand-in, neither key) -> C / D for g0, cb0, full, g0+cb0, g0+full -> X refusals
#   cmp       every comparison (after the variant steps)
# Outputs: $G0OUT (default code-811-s2p/tests/g0/out); node-local test checkpoints /scratch/amartyavarma/pose5-g0-test (kept).
set -uo pipefail
export G0CODE=${G0CODE:-/data/home/amartyavarma/ppm/pose5/code-811-s2p}
export G0OUT=${G0OUT:-$G0CODE/tests/g0/out}
TD=/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0; OUT=$G0OUT; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $OUT
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P PPM_G0COND
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0
export S2P_TEST_CE_CHUNK=${S2P_TEST_CE_CHUNK:-256}                  # tiny steps: every compared variant uses the same value
N="nice -n 19"
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID:-none} code $G0CODE out $OUT steps ${STEPS:-}"
has() { case ",${STEPS:-}," in *",$1,"*) return 0;; *) return 1;; esac; }
rv() {  # code g0 s2p tag port [micros]
  WORLD_SIZE=1 LOCAL_WORLD_SIZE=1 $N $PY $TD/run_variant.py --code $1 --g0 $2 --s2p $3 --model tiny --port $5 --out $OUT/$4.json \
    ${6:+--micros $6} || { echo "!! variant $4 FAILED"; return 1; }
}
SF=speech,gemma,chat,pose_full,pose_frozen
if has cfg; then
  for v in "ORIG 0 none orig" "COPY 0 none off" "COPY 1 none g0" "COPY 0 cb0 cb0" "COPY 0 full full" "COPY 1 cb0 g0cb0" "COPY 1 full g0full"; do
    set -- $v
    S2=$3; [ $S2 = none ] && S2=
    env PPM_G0COND=$2 ${S2:+PPM_S2P=$S2} WORLD_SIZE=128 LOCAL_WORLD_SIZE=8 $N $PY $TD/test_cfg.py $1 $OUT/cfg_$4.json || echo "!! cfg $4 FAILED"
  done
  $PY $TD/compare2.py cfg $OUT/CMP_cfg_combo.json $OUT/cfg_orig.json off=$OUT/cfg_off.json g0=$OUT/cfg_g0.json cb0=$OUT/cfg_cb0.json \
    full=$OUT/cfg_full.json g0cb0=$OUT/cfg_g0cb0.json g0full=$OUT/cfg_g0full.json
fi
has tA && { rv ORIG 0 none tiny_orig_off 29901; rv COPY 0 none tiny_off 29902; }
has tB && { rv COPY 1 none tiny_g0 29903; rv COPY 0 cb0 tiny_cb0 29904; }
has tC && { rv COPY 0 full tiny_full 29905; rv COPY 1 cb0 tiny_g0cb0 29906; }
has tD && { rv COPY 1 full tiny_g0full 29907; rv ORIG 0 none tiny_orig_off_stfirst 29908 $SF; }
has tE && { rv COPY nz cb0_nz tiny_g0cb0_nz_stfirst 29909 $SF; rv COPY nz full_nz tiny_g0full_nz_stfirst 29910 $SF; }
if has resume; then
  export WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
  R() {  # phase g0 s2p port [X source]
    echo "== resume phase $1 g0 $2 s2p $3 ${5:-}"
    S2=$3; [ $S2 = none ] && S2=
    env PPM_G0COND=$2 ${S2:+PPM_S2P=$S2} ${5:+G0_X_SRC=$5} $N $PY $TD/test_resume_combo.py --phase $1 \
      --code $( [ $1 = A ] && echo ORIG || echo ${XCODE:-COPY} ) --port $4 2>&1 | grep -v "^\[rank0\]:\[W"; echo "   rc ${PIPESTATUS[0]}"
  }
  CKT=/scratch/amartyavarma/pose5-g0-test/ckpt
  R A 0 none 29920
  p=29921
  for c in "1 none" "0 cb0" "0 full" "1 cb0" "1 full"; do set -- $c; R C $1 $2 $p; R D $1 $2 $((p + 1)); p=$((p + 2)); done
  R X 1 full 29940 $CKT/combo-g01_s2pcb0/C/dcp-17851             # mode change cb0 -> full: refuse
  R X 1 none 29941 $CKT/combo-g01_s2pcb0/C/dcp-17851             # s2p removed: refuse
  R X 0 cb0 29942 $CKT/combo-g01_s2pcb0/C/dcp-17851              # g0 removed: refuse
  R X 0 none 29943 $CKT/combo-g01_s2pfull/C/dcp-17851            # both removed: refuse
  XCODE=ORIG R X 0 none 29944 $CKT/combo-g01_s2pfull/C/dcp-17851  # original code: refuse (unknown keys)
  du -sh $CKT 2>/dev/null
fi
if has cmp; then
  C2="$PY $TD/compare2.py var"
  $PY $TD/compare.py var $OUT/tiny_orig_off.json $OUT/tiny_off.json same $OUT/CMP_tiny_off.json
  $C2 $OUT/tiny_orig_off.json $OUT/tiny_g0.json z:g0 $OUT/CMP_tiny_g0.json
  $C2 $OUT/tiny_orig_off.json $OUT/tiny_cb0.json z:cb0 $OUT/CMP_tiny_cb0.json
  $C2 $OUT/tiny_orig_off.json $OUT/tiny_full.json z:full $OUT/CMP_tiny_full.json
  $C2 $OUT/tiny_orig_off.json $OUT/tiny_g0cb0.json z:g0+cb0 $OUT/CMP_tiny_g0cb0.json
  $C2 $OUT/tiny_orig_off.json $OUT/tiny_g0full.json z:g0+full $OUT/CMP_tiny_g0full.json
  $C2 $OUT/tiny_orig_off_stfirst.json $OUT/tiny_g0cb0_nz_stfirst.json nz:g0+cb0 $OUT/CMP_tiny_g0cb0_nz.json
  $C2 $OUT/tiny_orig_off_stfirst.json $OUT/tiny_g0full_nz_stfirst.json nz:g0+full $OUT/CMP_tiny_g0full_nz.json
fi
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

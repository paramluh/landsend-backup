#!/bin/bash
# pose5 D (G0COND) real-width test (b-ii), ONE variant per job: args <ORIG|COPY> <g0: 0|1|nz> <s2p: none|full|cb0|full_nz|cb0_nz> <tag> <port>.
# REAL widths + REAL 8,1,1 weights (step 17641, lb 2 / lt 2 / ld 1) from workstream A's cache (shared Weka copy; same file A's runs use);
# settings = A's stage3 (CE chunk = production 1024, threads = SLURM_CPUS_PER_TASK), so A's real_orig_off.json is a valid common baseline.
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0; OUT=${G0OUT:-$TD/out}; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P PPM_G0COND S2P_TEST_CE_CHUNK
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
LB=2; LT=2; LD=1; MIC=${MIC:-pose_full,pose_frozen,speech,gemma,chat}     # (non-zero runs: MIC=speech,gemma,chat,pose_full,pose_frozen for both sides)
WC=/data/home/amartyavarma/ppm/pose5/s2p-test-weights/real_w_${LB}_${LT}_${LD}.pt     # A's shared cache (12.5 GB, any node)
[ -f $WC ] || WC=/scratch/amartyavarma/pose5-s2p-test/real_w_${LB}_${LT}_${LD}.pt
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID:-} code ${G0CODE:-code-811-g0} args $*"
[ -f $WC ] || { echo "!! no weight cache $WC on $(hostname -s)"; exit 1; }
nice -n 19 $PY $TD/run_variant.py --code $1 --g0 $2 --s2p $3 --model real --lb $LB --lt $LT --ld $LD --wcache $WC --micros $MIC --port $5 --out $OUT/$4.json || exit 1
echo "== done $(TZ=America/Los_Angeles date '+%T PT')"

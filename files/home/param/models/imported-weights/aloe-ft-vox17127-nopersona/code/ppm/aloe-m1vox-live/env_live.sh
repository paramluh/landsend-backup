# sourced by every aloe-m1vox live launcher (= infer_ppm.sh's ENVS line + the live front's Voxtral path).
# RELEASE: every path comes from the ONE env file aloe.env at the repo root (sourced here unless the caller already did; ALOE_ENV may
# point at another copy). HF_HOME, VOXTRAL_DIR, ALOE_MIMI_DIR, ALOE_FA3_DIR, ALOE_QUIET_BANK, ALOE_PRESET_BANK, ALOE_PY, ... are set there.
if [ -z "${ALOE_ROOT:-}" ]; then source "${ALOE_ENV:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../aloe.env}"; fi
source ${ALOE_ROOT}/qwen-s2-imend/hera3/env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PPM_CD=2 GEMM_PRECISION=bf16 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=${ALOE_SCRATCH}/triton-cache-ppm-live
export PY=${ALOE_PY} LV=${ALOE_ROOT}/ppm/aloe-m1vox-live
mkdir -p $TRITON_CACHE_DIR

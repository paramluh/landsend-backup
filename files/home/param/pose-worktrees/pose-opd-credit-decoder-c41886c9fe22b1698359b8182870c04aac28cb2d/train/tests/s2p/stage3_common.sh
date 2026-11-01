# sourced by stage3w.sh / stage3v.sh / stage3c.sh: env of the real-width test (b-ii)
TD=/data/home/amartyavarma/ppm/pose5/code-811-s2p/tests/s2p; OUT=$TD/out; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source /data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env; set +a
unset PPM_S2P S2P_TEST_CE_CHUNK
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 LOCAL_WORLD_SIZE=1
N="nice -n 19"; LB=${LB:-2}; LT=${LT:-2}; LD=${LD:-1}; MIC=${MIC:-pose_full,pose_frozen,speech,gemma,chat}
WC=${WC:-/data/home/amartyavarma/ppm/pose5/s2p-test-weights/real_w_${LB}_${LT}_${LD}.pt}   # shared (slinky-4 run 30947 used its node-local /scratch copy: same builder + source)
echo "== host $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT') job ${SLURM_JOB_ID:-} lb $LB lt $LT ld $LD micros $MIC"

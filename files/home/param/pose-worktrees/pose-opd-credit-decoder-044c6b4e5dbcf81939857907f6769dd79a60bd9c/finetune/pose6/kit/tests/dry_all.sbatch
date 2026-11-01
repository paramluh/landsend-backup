#!/bin/bash
#SBATCH -J pose6-kit-dry -p all -N1 -n1 -c2 --mem=16G -t 0-00:15:00 -x slinky-9,slinky-18
#SBATCH -o /data/home/amartyavarma/ppm/pose6/kit/tests/logs/dry_all-%j.log
# pose6 kit (B3): resolved CFG with the trainer's own code (tools/dry_cfg.py; no model build, CPU only, nothing written to /scratch) for every
# kit variant vs job 24272's resolved CFG (export-357 meta.json); input to CFG_DIFF.txt. Submit with -w slinky-4 (never slinky-18 / slinky-9).
# The env is built exactly as ft_s2pg0_run.sh builds it: source ft_s2pg0.env (FT_NODES) + tools/ft_overrides.sh (FT_LINK / FT_SAVES / FT_EXPORTS).
#  D0  ORIGINAL mono8020 code + env, world 64 (dry mode faithful: must be 0 keys)
#  D1  64-rank kit = DEFAULT (8 nodes, link off, --saves final)   D4  64-rank --link-on     D5  64-rank --saves hourly (recipe)     D6  64-rank --saves none
#  D2  FALLBACK one-node kit (world 8)      D3  control: s2p / G0COND / link-off all OFF + the recipe's hourly saves, 64 ranks (names / base only)
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; K=/data/home/amartyavarma/ppm/pose6/kit; C=/data/home/amartyavarma/ppm/pose6/code
M8=/data/home/amartyavarma/ppm/aloe-pose-mono-8020; REF=$K/ref/ref_cfg_24272.json; O=$K/tests/out
BASE_EX=/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-STEP
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8
CLR='unset $(env | grep -E "^(QQA_|ALOE_|PPM_|S3_MIN|GEMM|FT_)" | cut -d= -f1)'
echo "== code md5 $(md5sum $C/train_aloe_pose.py $C/data_aloe_pose.py $C/model_ppm.py | awk '{print substr($1,1,8)}' | paste -sd' ') $(TZ=America/Los_Angeles date +%T) on $(hostname -s)"
( eval "$CLR"; set -a; source $M8/aloe_mono8020.env; set +a
  export QQA_INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488 WORLD_SIZE=64 QQA_RUN=ppm-aloe-pose-mono-8020-j24229
  $PY -u $K/tools/dry_cfg.py $M8/code $O/dry_D0_mono8020_w64.json --ref $REF ) 2>&1 | grep -v Warning
for V in D1:8:off:final D4:8:on:final D5:8:off:hourly D6:8:off:none D2:1:off:none D3:8:on:hourly; do
  IFS=: read N NN LK SV <<< "$V"; W=$((8 * NN))
  echo "== $N nodes $NN world $W FT_LINK=$LK FT_SAVES=$SV $(TZ=America/Los_Angeles date +%T)"
  ( eval "$CLR"; export FT_NODES=$NN FT_LINK=$LK FT_SAVES=$SV FT_EXPORTS=all; set -a; source $K/ft_s2pg0.env; set +a; . $K/tools/ft_overrides.sh
    [ $N = D3 ] && { export PPM_S2P=0 PPM_G0COND=0; unset ALOE_G0_SIDECAR; }
    echo "   env: QQA_DECP_BB_OFF=$QQA_DECP_BB_OFF ALOE_SAVE_EVERY_MIN=$ALOE_SAVE_EVERY_MIN ALOE_SKIP_FINAL_FULL=${ALOE_SKIP_FINAL_FULL:-} ALOE_OPT_OFFLOAD=${ALOE_OPT_OFFLOAD:-} QQA_EXPORT_STEPS=$QQA_EXPORT_STEPS"
    export QQA_INIT=$BASE_EX WORLD_SIZE=$W QQA_RUN=ppm-pose6-ft-s2pg0-jNNNNN FT_ENV_FILE=$K/ft_s2pg0.env
    $PY -u $K/tools/dry_cfg.py $C $O/dry_${N}_w$W.json plan --ref $REF ) 2>&1 | grep -v Warning
done
echo "== done $(TZ=America/Los_Angeles date +%T)"

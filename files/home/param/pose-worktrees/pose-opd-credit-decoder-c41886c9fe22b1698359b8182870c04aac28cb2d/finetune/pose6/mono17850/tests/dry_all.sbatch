#!/bin/bash
#SBATCH -J pose6-m17850-dry -p all -N1 -n1 -c2 --mem=16G -t 0-00:20:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/pose6/mono17850/tests/logs/dry_all-%j.log
# pose6 mono17850: resolved CFG (trainer's own code, tools/dry_cfg.py; CPU; writes NOTHING to /scratch) vs job 24272's, + preflight --mods none.
# STAND-IN BASE until C1's local-17850 exists: QQA_INIT = the target path string (the cfg only records it); preflight positives on the 8,1,1
# local-17641 (same run lineage / format / world 128 as local-17850) and export-357.
#  M1 64-rank DEFAULT (recipe hourly saves, link OFF)   M2 64-rank --saves none (flagged deviation for placement (a))   M3 FALLBACK world 8
#  M6 = main's GO (14:41): 64-rank --saves none --final-export-only --link-on
#  M5 placement (a'): 64-rank --saves none --final-export-only
#  M4 control: 64-rank --link-on (= mono8020's knobs exactly: only init + names / paths may differ)
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; M=/data/home/amartyavarma/ppm/pose6/mono17850; KT=/data/home/amartyavarma/ppm/pose6/kit
C=/data/home/amartyavarma/ppm/pose6/code; REF=$KT/ref/ref_cfg_24272.json; O=$M/tests/out; B=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 RANK=0 LOCAL_RANK=0 LOCAL_WORLD_SIZE=8
CLR='unset $(env | grep -E "^(QQA_|ALOE_|PPM_|S3_MIN|GEMM|FT_)" | cut -d= -f1)'
echo "== code md5 $(md5sum $C/train_aloe_pose.py $C/data_aloe_pose.py $C/model_ppm.py | awk '{print substr($1,1,8)}' | paste -sd' ') $(TZ=America/Los_Angeles date +%T) on $(hostname -s)"
for V in M6:8:on:none:final M1:8:off:hourly:all M2:8:off:none:all M5:8:off:none:final M3:1:off:none:all M4:8:on:hourly:all; do
  IFS=: read N NN LK SV EX <<< "$V"; W=$((8 * NN))
  echo "== $N nodes $NN world $W FT_LINK=$LK FT_SAVES=$SV $(TZ=America/Los_Angeles date +%T)"
  ( eval "$CLR"; export FT_NODES=$NN FT_LINK=$LK FT_SAVES=$SV FT_EXPORTS=$EX; set -a; source $M/mono17850.env; set +a; . $KT/tools/ft_overrides.sh
    echo "   env: PPM_S2P=${PPM_S2P:-unset} PPM_G0COND=${PPM_G0COND:-unset} QQA_DECP_BB_OFF=$QQA_DECP_BB_OFF ALOE_SAVE_EVERY_MIN=$ALOE_SAVE_EVERY_MIN ALOE_SKIP_FINAL_FULL=${ALOE_SKIP_FINAL_FULL:-} ALOE_OPT_OFFLOAD=${ALOE_OPT_OFFLOAD:-}"
    export QQA_INIT=$B WORLD_SIZE=$W QQA_RUN=ppm-pose6-mono17850-jNNNNN
    $PY -u $KT/tools/dry_cfg.py $C $O/dry_${N}_w$W.json plan --ref $REF ) 2>&1 | grep -v Warning
done
for X in /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641 /data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357; do
  echo "== preflight --mods none $X (expect PASS) $(TZ=America/Los_Angeles date +%T)"
  ( eval "$CLR"; export FT_NODES=8; set -a; source $M/mono17850.env; set +a
    $PY -u $KT/tools/preflight.py --mods none --skip-sidecar --base $X --code $C --json $O/preflight_$(basename $X).json ) 2>&1 | grep -E 'RESULT'
done
echo "== done $(TZ=America/Los_Angeles date +%T)"

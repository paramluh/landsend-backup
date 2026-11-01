#!/bin/bash
#SBATCH -J pds-gen -p all -N1 -n1 -t 01:30:00
#SBATCH --gres-flags=disable-binding
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/logs/gen-%j-%N.log
# usage: PDS_CK=<export dir> sbatch -w <host> --gres=gpu:N -c C --mem=M gen.sbatch <conv[+conv]>:<gpu,gpu> [...]  (gpu = index into THIS job's allocation)
S=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples; CK=${PDS_CK:-/scratch/amartyavarma/pdsamples/export-498}; RUNS=${PDS_RUNS:-$S/runs}
source /data/home/amartyavarma/ppm/aloe-m1vox-live/env_live.sh
export TRITON_CACHE_DIR=/scratch/amartyavarma/triton-cache-pdsamples; mkdir -p $TRITON_CACHE_DIR $RUNS
[ -f $CK/COMPLETE ] || { echo "no staged export $CK"; exit 3; }
nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu,memory.used --format=csv,noheader; echo "CVD=${CUDA_VISIBLE_DEVICES:-}"
IFS=, read -ra AL <<< "${CUDA_VISIBLE_DEVICES:?no GPUs allocated}"
PIDS=()
for spec in "$@"; do
  cs=${spec%%:*}; gi=${spec#*:}
  g=$(for i in ${gi//,/ }; do [ -n "${AL[$i]:-}" ] || exit 9; printf '%s,' "${AL[$i]}"; done) || { echo "GPU index $gi not in allocation"; exit 9; }; g=${g%,}
  cv=""; od=""; sd=""
  for c in ${cs//+/ }; do n=$(printf %03d $c); cv="$cv,$S/convs/c$c.npz"; od="$od,$RUNS/B-$n"; sd="$sd,$((20261025 + 100 * c))"; done
  CUDA_VISIBLE_DEVICES=$g nice -n 5 $PY -u $S/code/gen_pd.py --ck $CK --conv ${cv#,} --out ${od#,} --seed ${sd#,} ${PDS_FRAMES:+--frames $PDS_FRAMES} ${PDS_PARAMS:+--params "$PDS_PARAMS"} \
    > $S/logs/duo-$(basename $RUNS)-${cs//+/_}-$SLURM_JOB_ID.log 2>&1 &
  PIDS+=($!); echo "$(date +%T) convs $cs on GPUs $g pid ${PIDS[-1]}"
done
for p in "${PIDS[@]}"; do wait $p; echo "$(date +%T) pid $p exit $?"; done

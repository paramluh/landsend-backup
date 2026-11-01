#!/bin/bash
#SBATCH -J genweb-render2
#SBATCH -x slinky-9
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
# SHARED render queue: one render_service instance per GPU, playground jobs FIRST (pg:), then posefix's batches, all with atomic claims.
# 10-26 (user via main / videort): Vansh's SEQUENTIAL sampler. Each GPU runs videort's rt/render_multi.sh exactly as videort calls it
# (render_multi.sh <gpu> <tag>: RS_ENGINE=seq, RS_SPECS=SPECS_seq.txt, its own env), so no env of ours changes its maths; a restart loop
# per GPU. Logs: video/dmd-rt/logs/render-pg<job>-g<gpu>.log. Old engine version: render2.sbatch.bak-pre-seq-1026.
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
bash $CANOPY_ROOT/ppm/aloe-posedup-live/tools/gpu_guard.sh "playground renderer pair" || { echo "GPU guard refused playground renderer pair; exiting"; exit 3; }   # 10-27 main: never slinky-4 GPUs 6-7 / SM < 1500 MHz for live / user-facing work
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES host $(hostname) $(date)"
IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"
for g in "${GP[@]}"; do
  if [ "$(hostname)" = slinky-4 ] && { [ $g = 6 ] || [ $g = 7 ]; }; then   # 10-26 main: user jobs NEVER on the throttled slinky-4 GPUs 6/7
    ( while true; do bash ${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}/render_flex.sh $g posefix pg$SLURM_JOB_ID
        echo "render_flex posefix g$g exited rc=$? $(date)"; sleep 10; done ) &
  else
    ( while true; do bash $CANOPY_ROOT/video/dmd-rt/rt/render_multi.sh $g pg$SLURM_JOB_ID
        echo "render_multi g$g exited rc=$? $(date)"; sleep 10; done ) &
  fi
done
wait

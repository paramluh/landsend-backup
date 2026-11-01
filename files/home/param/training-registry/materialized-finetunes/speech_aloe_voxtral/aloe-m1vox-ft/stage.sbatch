#!/bin/bash
#SBATCH -J m1vox-stage -p all -N8 --ntasks-per-node=1 -c 48 --mem=200G --gres=gpu:1 -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/stage-%j.log
# aloe-m1vox-ft staging on the 8 training hosts (node i = i-th host sorted -V): tools/stage_m1vox.py --node i (21980's 4 stores, the node's A
# FLACs verified, Mimi decodes of purged rows); then the superseded aloe-voxall-ft data caches (22303, discarded) are removed, and on
# slinky-0 the node-local cache copy of PPM local-11488 (/scratch/amartyavarma/asrppm/bb-11488; the /data copy stays) to keep >= 100 GB free.
set -uo pipefail
export STAGE_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
echo "== stage $SLURM_JOB_ID on $STAGE_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT')"
srun --kill-on-bad-exit=0 bash -c '
H=$(hostname -s); I=$(echo "$STAGE_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); K=$((I - 1)); cd /tmp
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 S3_MIN_FRAMES=125 PPM_CD=2
echo "$H node $K start, scratch free $(df -BG --output=avail /scratch | tail -1)"
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-m1vox-ft/tools/stage_m1vox.py --node $K --root /scratch/amartyavarma/aloe-m1vox-ft/flac --workers 64 > /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/stage-node$K-$H.log 2>&1 || { echo "$H node $K STAGE FAILED"; tail -5 /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/stage-node$K-$H.log; exit 1; }
tail -2 /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/stage-node$K-$H.log
rm -rf /scratch/amartyavarma/aloe-voxall-ft/flac /scratch/amartyavarma/aloe-voxall-ft/stores /scratch/amartyavarma/aloe-voxall-build
if [ "$H" = slinky-0 ] && [ -f /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488/COMPLETE ]; then rm -rf /scratch/amartyavarma/asrppm/bb-11488 && echo "$H removed node-local cache copy asrppm/bb-11488 (63 GB; /data local-11488 kept)"; fi
# SPEED FIX 10-16 (SPEED_FIX.md): decode the A rows of this node once into the node-local row cache (disk-guarded; misses decode on the fly, bit-identical)
ALOE_ROWCACHE=/scratch/amartyavarma/aloe-m1vox-ft/rowcache /data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-m1vox-ft/tools/stage_rowcache.py --node $K --workers 32 --verify 4 > /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/rowcache-node$K-$H.log 2>&1 && tail -1 /data/home/amartyavarma/ppm/aloe-m1vox-ft/logs/rowcache-node$K-$H.log || echo "$H node $K ROWCACHE FAILED (training still runs: on-the-fly decode)"
echo "$H node $K done, scratch free $(df -BG --output=avail /scratch | tail -1)"'
rc=$?; echo "== stage exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc

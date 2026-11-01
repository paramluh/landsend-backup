#!/bin/bash
#SBATCH -J v2vox-stage -p all -N8 --ntasks-per-node=1 -c 32 --mem=120G --gres=gpu:1 -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/stage-%j.log
# aloe-v2vox-ft: verify / complete each training node's A audio for the v2 plan (tools/stage_v2vox.py --node i; node i = i-th host sorted -V).
set -uo pipefail
export STAGE_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
echo "== stage $SLURM_JOB_ID on $STAGE_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT')"
srun --kill-on-bad-exit=0 bash -c '
H=$(hostname -s); I=$(echo "$STAGE_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); K=$((I - 1)); cd /tmp
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 S3_MIN_FRAMES=125 PPM_CD=2
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-v2vox-ft/tools/stage_v2vox.py --node $K --root /scratch/amartyavarma/aloe-m1vox-ft/flac --workers 32 > /data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/stage-node$K-$H.log 2>&1 || { echo "$H node $K STAGE FAILED"; tail -5 /data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/stage-node$K-$H.log; exit 1; }
echo "$H node $K: $(tail -1 /data/home/amartyavarma/ppm/aloe-v2vox-ft/logs/stage-node$K-$H.log); scratch free $(df -BG --output=avail /scratch | tail -1)"'
rc=$?; echo "== stage exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc

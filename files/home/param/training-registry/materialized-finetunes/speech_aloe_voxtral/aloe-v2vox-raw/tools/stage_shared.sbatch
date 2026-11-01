#!/bin/bash
#SBATCH -J v2vraw-stage -p all -N1 -n1 -c 16 --mem=48G -t 2:00:00 -w slinky-4
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/stage-shared-%j.log
# aloe-v2vox-raw: ALL 8 logical nodes' channel-A tracks (238 GB, GCS @ manifest generation, crc32c + md5 verified) -> the shared /data copy;
# + the 4 stores on this node's /scratch. /data keeps >= 700 GB free (exports + the user's 300 GB floor).
cd /tmp
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONWARNINGS=ignore OMP_NUM_THREADS=4 S3_MIN_FRAMES=125 PPM_CD=2
echo "== stage $SLURM_JOB_ID on $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
nice -n 5 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I /data/home/amartyavarma/ppm/aloe-v2vox-raw/tools/stage_raw.py --lnodes 0,1,2,3,4,5,6,7 \
  --root /data/home/amartyavarma/ppm/aloe-v2vox-raw/flac --workers 64 --min-free-gb 700
rc=$?; echo "== stage exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; df -h /data | tail -1; exit $rc

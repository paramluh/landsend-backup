#!/bin/bash
#SBATCH -J v2vraw-tests -p all -N1 -n1 -c 16 --mem=64G -t 0:45:00 -w slinky-4
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2vox-raw/logs/tests-%j.log
# aloe-v2vox-raw CPU tests on the shared A-track copy: raw-gain waveform test, accumulation-stream test, loudness report (sample + full population).
cd /tmp
export VOX_A_GAIN=raw ALOE_FLAC_ROOT=/data/home/amartyavarma/ppm/aloe-v2vox-raw/flac ALOE_ROWCACHE= PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 S3_MIN_FRAMES=125 PPM_CD=2
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; T=/data/home/amartyavarma/ppm/aloe-v2vox-raw/tests
echo "== tests $SLURM_JOB_ID on $(hostname -s) $(TZ=America/Los_Angeles date '+%F %T PT')"
bash $T/test_cfgdiff.sh 2>&1 | grep -v Warning | tail -25; r0=${PIPESTATUS[0]}
$PY -I $T/test_raw_gain.py --n 6 2>&1 | grep -v Warning; r1=${PIPESTATUS[0]}
$PY -I $T/test_accds.py 2>&1 | grep -v Warning; r2=${PIPESTATUS[0]}
$PY -I $T/loudness_report.py --n 360 --workers 14 --out $T/loudness_report.json 2>&1 | grep -v Warning; r3=${PIPESTATUS[0]}
echo "== rc cfgdiff $r0 raw_gain $r1 accds $r2 loudness $r3 $(TZ=America/Los_Angeles date '+%T PT')"

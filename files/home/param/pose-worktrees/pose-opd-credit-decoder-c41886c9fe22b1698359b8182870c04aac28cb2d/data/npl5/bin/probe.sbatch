#!/bin/bash
#SBATCH -J npl5-probe -p all -w slinky-10 -x slinky-9 -N1 -n1 -c 6 --mem=24G --gres=none -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/probe-%j.out
# npl5 measure-first probe on a spare (CPU/net/scratch only, nice 19 + ionice idle): 2,000 random delivery-5 releases ->
# HF fetch rate (src_fetch.py, HF_RATE 120, 64 threads), xet verify time, fresh_pose.py scan rate (6 workers), spare->Weka copy rate.
set -uo pipefail
A=/data/home/amartyavarma/ppm/code-811/fresh/append; W=/data/home/amartyavarma/ppm/pose5/npl5/probe; DEST=/scratch/amartyavarma/npl5-stage/probe
PYV=/data/home/amartyavarma/venvs/voicebank/bin/python; PYQ=/data/home/amartyavarma/venvs/qwen-s2/bin/python
N="nice -n19 ionice -c3"
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
echo "== $(pt) host $(hostname -s) nproc $(nproc) /scratch free $(df -BG --output=avail /scratch | tail -1) load $(cat /proc/loadavg)"
grep -c ^processor /proc/cpuinfo; lscpu | grep -E "Model name|MHz" | head -3
t0=$(date +%s); HF_RATE=120 THREADS_HF=64 $N $PYV -I $A/src_fetch.py $W $DEST; echo "== $(pt) fetch rc $? $(( $(date +%s) - t0 )) s"
t0=$(date +%s); VERIFY_THREADS=6 $N $PYV -I $A/src_verify.py $W $DEST; echo "== $(pt) verify rc $? $(( $(date +%s) - t0 )) s"
t0=$(date +%s); env FRESH_SRC=$DEST/jobs FRESH_SCAN=$W/scan FRESH_HOST=probe OMP_NUM_THREADS=1 $N $PYQ -I /data/home/amartyavarma/ppm/code-811/fresh/fresh_pose.py scan --workers 6; echo "== $(pt) scan rc $? $(( $(date +%s) - t0 )) s"
mkdir -p /data/home/amartyavarma/ppm/pose5/npl5/probe/wekacopy
t0=$(date +%s); $N cp -r $DEST/jobs /data/home/amartyavarma/ppm/pose5/npl5/probe/wekacopy/; sync; echo "== $(pt) spare->weka copy $(du -sb $DEST/jobs | cut -f1) bytes $(( $(date +%s) - t0 )) s"
rm -rf /data/home/amartyavarma/ppm/pose5/npl5/probe/wekacopy
echo "== $(pt) done; /scratch free $(df -BG --output=avail /scratch | tail -1) load $(cat /proc/loadavg)"

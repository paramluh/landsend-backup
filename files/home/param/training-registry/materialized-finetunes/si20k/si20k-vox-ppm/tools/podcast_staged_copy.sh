#!/bin/bash
# stage one donor's partial slice onto one target: donor /scratch -> /data temp -> target /scratch, sizes checked at each hop, /data temp deleted.
# usage: podcast_staged_copy.sh DONOR TARGET     (run from the login node: it only launches srun steps into running jobs; CPU only, nice)
set -euo pipefail
DN=$1; TG=$2
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
TOOL=/data/home/amartyavarma/ppm/si20k-vox-ppm/tools/podcast_staged.py
WORK=/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/staged_work
SRC=/scratch/amartyavarma/s3-rows
TMP=/data/home/amartyavarma/tmp-si20k-pod/$DN-slice
DST=/scratch/amartyavarma/si20k-pod/$DN-slice
LST=$WORK/LISTING-$DN.txt
jid() { squeue -h -u amartyavarma -w "$1" -t R -o "%i %j" | sort -k2 | awk '$2 ~ /^tt3q-/ {print $1; exit}'; }
JD=$(jid $DN); JT=$(jid $TG)
[ -n "$JD" ] && [ -n "$JT" ] || { echo "no tt3q job on $DN ($JD) or $TG ($JT)"; exit 1; }
avail=$(df -B1G --output=avail /data | tail -1 | tr -d ' ')
echo "$(date +%T) $DN -> $TG (jobs $JD / $JT); /data avail ${avail} GB"
[ "$avail" -ge 330 ] || { echo "/data below 330 GB free: refusing"; exit 1; }
RUN() { srun --jobid=$1 --overlap -N1 -n1 -c1 --mem=2G --gres=none --cpu-bind=none nice -n 10 "${@:2}"; }
# hop 1: donor -> /data temp
RUN $JD env OMP_NUM_THREADS=1 $PY $TOOL listing --donor $DN --shards-json $WORK/select.json --out $LST
mkdir -p $TMP
RUN $JD bash -c "cd $SRC && awk '{print \$2}' $LST | xargs -P 4 -I{} cp --preserve=timestamps {} $TMP/{} && \
  awk -v d=$TMP '{ cmd=\"stat -c %s \" d \"/\" \$2; cmd | getline s; close(cmd); if (s != \$1) print \$2 }' $LST > $WORK/sizecheck1-$DN.txt"
bad=$(wc -l < $WORK/sizecheck1-$DN.txt)
echo "$(date +%T) hop1 done: $(awk '{s+=$1} END {print s}' $LST) bytes listed, size mismatches on /data $bad"
[ "$bad" -eq 0 ] || { echo "hop1 size mismatch"; exit 1; }
# hop 2: /data temp -> target /scratch, sizes checked on the target, then delete the /data temp
RUN $JT bash -c "mkdir -p $DST && cd $TMP && awk '{print \$2}' $LST | xargs -P 4 -I{} cp --preserve=timestamps {} $DST/{} && \
  awk -v d=$DST '{ cmd=\"stat -c %s \" d \"/\" \$2; cmd | getline s; close(cmd); if (s != \$1) print \$2 }' $LST > $WORK/sizecheck-$TG.txt"
bad=$(wc -l < $WORK/sizecheck-$TG.txt)
echo "$(date +%T) hop2 done on $TG: size mismatches $bad"
[ "$bad" -eq 0 ] || { echo "hop2 size mismatch (kept $TMP)"; exit 1; }
rm -rf $TMP
echo "$(date +%T) deleted $TMP; /data avail $(df -B1G --output=avail /data | tail -1 | tr -d ' ') GB"

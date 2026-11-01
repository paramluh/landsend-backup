#!/bin/bash
# runs podcast_plan_ppm.py host mode for every prod host (meta + live), each ON its host via srun --overlap into one of the user's running
# jobs there (CPU only, --gres=none, nice), then the merge (on slinky-13). usage: podcast_plan_ppm_run.sh [BUILD_N]
T=/data/home/amartyavarma/ppm/si20k-vox-ppm/tools/podcast_plan_ppm.py
O=/data/home/amartyavarma/ppm/si20k-vox-ppm/plan/podcast_hosts
L=/data/home/amartyavarma/ppm/si20k-vox-ppm/logs/podcast_plan
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
B=${1:-2}
mkdir -p $O $L
HOSTS="slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-8 slinky-9 slinky-10 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19"
jid() { squeue -h -u amartyavarma -w "$1" -t R -o "%i %j" | sort -k2 | awk '$2 ~ /^tt3q-|wfilter/ {print $1; exit}'; }
for h in $HOSTS; do
  j=$(jid $h); [ -n "$j" ] || j=$(squeue -h -u amartyavarma -w $h -t R -o "%i" | head -1)
  [ -n "$j" ] || { echo "$h: no running job to overlap"; continue; }
  ( srun --jobid=$j --overlap -N1 -n1 -c2 --mem=8G --gres=none --cpu-bind=none nice -n 10 $PY $T host --host $h --src meta --out $O > $L/meta-$h.log 2>&1
    srun --jobid=$j --overlap -N1 -n1 -c2 --mem=8G --gres=none --cpu-bind=none nice -n 10 env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 \
        $PY $T host --host $h --src live --out $O --build $B > $L/live-$h.log 2>&1
    echo "$h done (job $j): $(tail -1 $L/meta-$h.log) | $(tail -1 $L/live-$h.log)" ) &
done
wait
j=$(jid slinky-13)
srun --jobid=$j --overlap -N1 -n1 -c2 --mem=16G --gres=none --cpu-bind=none nice -n 10 $PY $T merge --dir $O --out /data/home/amartyavarma/ppm/si20k-vox-ppm/plan/PODCAST_PLAN_ppm.json > $L/merge.log 2>&1
echo "merge rc $?"; tail -40 $L/merge.log

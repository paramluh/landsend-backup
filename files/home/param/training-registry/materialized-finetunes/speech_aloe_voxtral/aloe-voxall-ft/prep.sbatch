#!/bin/bash
#SBATCH -p all -N8 --ntasks-per-node=1 -c 48 --mem=200G --gres=none -t 1:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-voxall-ft/logs/prep-%j.log
# aloe-voxall-ft staging on 8 training hosts: node i (sorted -V) = store p<i> (build + A FLACs), see tools/prep_node.sh. Then merge STORES.json.
# Also removes the superseded aloe-vox-ft job 22284's node-local export (user: outputs superseded, not backed up) when DEL_OLD=1.
set -uo pipefail
V=/data/home/amartyavarma/ppm/aloe-voxall-ft
export PREP_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
echo "== prep job $SLURM_JOB_ID on $PREP_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT')"
srun --kill-on-bad-exit=0 bash -c '
H=$(hostname -s); I=$(echo "$PREP_HOSTS" | tr , "\n" | grep -nx "$H" | cut -d: -f1); K=$((I - 1))
if [ "${DEL_OLD:-0}" = 1 ] && [ -d /scratch/amartyavarma/aloe-vox-ft/exports ]; then echo "$H deleting 22284 node-local saves: $(ls /scratch/amartyavarma/aloe-vox-ft/exports | tr "\n" " ")"; rm -rf /scratch/amartyavarma/aloe-vox-ft/exports; fi
bash /data/home/amartyavarma/ppm/aloe-voxall-ft/tools/prep_node.sh $K'
rc=$?
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I $V/tools/store_meta.py merge || rc=1
echo "== prep exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc

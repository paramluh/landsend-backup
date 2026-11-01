#!/bin/bash
#SBATCH -J samstores -p all -N1 -n1 -c16 --mem=96G -t 0-02:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-ft-sam/logs/build-%j.log
# aloe-pose-ft-sam store build (10-26): SAM isolated codes swapped into the aloe-pose-ft stores. cache (32 SAM shards) -> check (STOP on any
# missing row) -> build p0..p7 at once -> report. Code: code/build_sam_stores.py.
O=/data/home/amartyavarma/ppm/aloe-pose-ft-sam; PY="nice -n 5 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $O/code/build_sam_stores.py"
NP=${SLURM_CPUS_PER_TASK:-16}
echo "== $(hostname -s) $NP cpus $(TZ=America/Los_Angeles date '+%F %T PT')"
seq 0 31 | xargs -P $NP -I{} $PY cache --shard {} || { echo "== CACHE FAILED"; exit 1; }
$PY check || { echo "== CHECK FAILED: missing / mismatched SAM rows (see CHECK.json); nothing built"; exit 2; }
mkdir -p $O/stores
for k in 0 1 2 3 4 5 6 7; do $PY build --store $k --workers $(( NP / 4 > 2 ? NP / 4 : 2 )) > $O/logs/build-p$k.log 2>&1 & done
rc=0; for p in $(jobs -p); do wait $p || rc=1; done
cat $O/logs/build-p?.log | grep -E "DONE|Error|Traceback|assert" 
[ $rc = 0 ] || { echo "== BUILD FAILED"; exit 3; }
cp -p /data/home/amartyavarma/ppm/aloe-pose-ft/stores/STORES.json /data/home/amartyavarma/ppm/aloe-pose-ft/stores/PLAN.json $O/stores/
cmp /data/home/amartyavarma/ppm/aloe-pose-ft/stores/STORES.json $O/stores/STORES.json && cmp /data/home/amartyavarma/ppm/aloe-pose-ft/stores/PLAN.json $O/stores/PLAN.json || exit 4
$PY report || exit 5
touch $O/stores/ALL_DONE
echo "== ALL DONE $(TZ=America/Los_Angeles date '+%F %T PT')"

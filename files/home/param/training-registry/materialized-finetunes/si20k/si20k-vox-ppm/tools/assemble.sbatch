#!/bin/bash
#SBATCH -p all -N1 -n1 -c 16 --mem=64G --gres=none -t 1:00:00 -J si20k-assemble -w slinky-2
#SBATCH -o /data/home/amartyavarma/qwen-1ch-data/si20k-vox/logs/assemble-%j.log
# si20k-vox FINAL store assembly (CPU, on the node holding the prep cache): build_si20k_rows.py --mode assemble (--require-final: the TTS run
# has nothing left to render / re-voice; catch-up prep of any late finishes; keep spans / keep list applied; 7,500-frame rows; stratified
# eval) -> /data stores, then finalize_plan.py (STORES.json + both runs' plan/PLAN.env). usage: sbatch [--export=ALL,SPANS=..,KEEP=..] assemble.sbatch
set -uo pipefail
B=/data/home/amartyavarma/qwen-1ch-data/si20k-vox; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; OUT=$B/stores; CACHE=/scratch/amartyavarma/si20k-vox/cache
export PPM_CD=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
ARGS=""; [ -n "${SPANS:-}" ] && ARGS="$ARGS --keep-spans $SPANS"; [ -n "${KEEP:-}" ] && ARGS="$ARGS --keep $KEEP"
echo "== assemble $SLURM_JOB_ID $(TZ=America/Los_Angeles date '+%F %T PT') args:$ARGS; /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"
cd /tmp
rm -rf $OUT/s20n0 $OUT/s20n1 $OUT/s20n2 $OUT/s20n3 $OUT/STORES.json $OUT/BUILD.json $OUT/DROPPED.tsv
nice -n 5 $PY -I $B/build_si20k_rows.py --mode assemble --cache $CACHE --out $OUT --workers 16 --require-final ${REQUIRE_FINAL:-1} $ARGS || exit 1
nice -n 5 $PY -I $B/tools/finalize_plan.py --stores $OUT || exit 1
echo "== assemble done $(TZ=America/Los_Angeles date '+%T PT'); stores $(du -sh $OUT | cut -f1); /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"

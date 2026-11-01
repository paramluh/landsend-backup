#!/bin/bash
# GenghisBench on a PPM checkpoint (v5 oracle, T 0.7, no top-k, the run_chat protocol; see gb_ppm.py):
#   bash gb_ppm.sh <ckpt> [n=20] [seed0=100]          e.g. bash gb_ppm.sh /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-859
# Submits 2-GPU worker jobs per GB_PLAN (each loads the checkpoint once and claims games one by one, so games spread over the workers),
# at most n in all, plus a CPU merge job (afterany) -> /data/home/amartyavarma/ppm/gb/<ckpt tag>/
#   games/<seed>.json, gb_s<seeds>.json, summary_s<seeds>.txt. Rerunning the same command replays only seeds without a result.
# env: GB_PLAN node:workers list (default slinky-4:4 — slinky-7/15 busy with ASR training from 10-12; slinky-18 = the user's node, add at most one worker there by hand),
#      GB_LAUNCH (join an existing launch: more workers, no new merge if GB_NO_MERGE=1), GB_TIME (minutes per worker, 480), GB_TAG
set -uo pipefail
C=/data/home/amartyavarma/ppm/code
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
CK=$(readlink -f "${1:?usage: gb_ppm.sh <ckpt> [n] [seed0]}"); N=${2:-20}; S0=${3:-100}
[ -f "$CK/COMPLETE" ] || [ -f "$CK/.metadata" ] || { echo "not a complete PPM checkpoint: $CK"; exit 1; }
if [ -n "${GB_TAG:-}" ]; then TAG=$GB_TAG
elif [ -f "$CK/meta.json" ]; then TAG=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('run_base') or 'ppm')" "$CK/meta.json")-$(basename "$CK")
else TAG=$(basename "$(dirname "$CK")")-$(basename "$CK"); fi
OUT=/data/home/amartyavarma/ppm/gb/$TAG; mkdir -p "$OUT/logs"
SEEDS=$S0-$((S0 + N - 1))
PLAN=${GB_PLAN:-slinky-4:4}
LAUNCH=${GB_LAUNCH:-$(date +%s)}   # claims are per launch; GB_LAUNCH=<id> adds workers to a running launch
IDS=(); NODES=()
for nw in ${PLAN//,/ }; do
  node=${nw%%:*}; k=${nw##*:}; NODES+=("$node")
  for i in $(seq 1 "$k"); do
    [ "${#IDS[@]}" -ge "$N" ] && break
    j=$(sbatch --parsable -p all -w "$node" -N1 -n1 -c 8 --gres=gpu:2 --mem=120G -t "${GB_TIME:-480}" -J gb-ppm \
        -o "$OUT/logs/w-%j.log" --wrap "bash $C/gb_ppm_job.sh worker --ckpt $CK --out $OUT --seeds $SEEDS --launch $LAUNCH --devs 0,1")
    IDS+=("$j")
  done
done
[ "${GB_NO_MERGE:-0}" = 1 ] && { echo "GB $TAG seeds $SEEDS: extra workers ${IDS[*]} (launch $LAUNCH)"; exit 0; }
M=$(sbatch --parsable -p all -w "${NODES[0]}" -N1 -n1 -c 1 --mem=4G -t 10 -J gb-ppm-merge --dependency=afterany:$(IFS=:; echo "${IDS[*]}") \
    -o "$OUT/logs/merge-%j.log" --wrap "$PY $C/gb_ppm.py merge --out $OUT --seeds $SEEDS --label $TAG")
echo "GB $TAG seeds $SEEDS: workers ${IDS[*]} on {${NODES[*]}}, merge $M -> $OUT/summary_s$SEEDS.txt"

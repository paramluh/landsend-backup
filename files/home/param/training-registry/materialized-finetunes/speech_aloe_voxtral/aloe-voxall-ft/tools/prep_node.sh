#!/bin/bash
# aloe-voxall-ft: stage ONE training node (run inside an allocation on that node): store K (build_voxall_rows build, node-local) and
# store K's channel-A FLACs (fetch_flacs, 64 threads, size + crc32c + md5 vs GCS at the manifest generation) IN PARALLEL, then the store's
# STORES.json entry (store_meta one). Existing FLACs of the old aloe-vox-ft root on this node are hard-linked in first (then re-verified).
#   bash prep_node.sh K
set -uo pipefail
K=$1; V=/data/home/amartyavarma/ppm/aloe-voxall-ft; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
R=/scratch/amartyavarma/aloe-voxall-ft; H=$(hostname -s); L=$V/logs/prep
mkdir -p $R/stores $R/flac $L; cd /tmp
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false S3_MIN_FRAMES=125 PPM_CD=2 PYTHONWARNINGS=ignore
export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json
echo "$H store $K start $(TZ=America/Los_Angeles date +%T) scratch free $(df -BG --output=avail /scratch | tail -1)"
if [ -d /scratch/amartyavarma/aloe-vox-ft/flac ] && [ ! -f $R/flac/.linked ]; then
  (cd /scratch/amartyavarma/aloe-vox-ft/flac && find . -name '*.flac' -print0 | cpio -0pdl --quiet $R/flac 2>/dev/null); touch $R/flac/.linked
  echo "$H hard-linked $(find $R/flac -name '*.flac' | wc -l) old FLACs"
fi
rm -rf $R/stores/p$K
( $PY -I $V/code/build_voxall_rows.py build --store $K --out $R/stores --workers ${BUILD_WORKERS:-32} > $L/build-p$K-$H.log 2>&1 && echo "$H build p$K ok" || { echo "$H build p$K FAILED"; exit 1; } ) &
BP=$!
( $PY -I $V/tools/fetch_flacs.py --store $K --workers ${FETCH_WORKERS:-64} --root $R/flac > $L/fetch-p$K-$H.log 2>&1 && echo "$H fetch p$K ok" || { echo "$H fetch p$K FAILED"; exit 1; } ) &
FP=$!
wait $BP; b=$?; wait $FP; f=$?
[ $b = 0 ] && [ $f = 0 ] || { echo "$H store $K prep FAILED (build $b fetch $f)"; exit 1; }
rm -rf /scratch/amartyavarma/aloe-voxall-build/p$K
$PY -I $V/tools/store_meta.py one --store $K --root $R/stores || exit 1
tail -1 $L/fetch-p$K-$H.log
# SPEED FIX 10-16 (SPEED_FIX.md): decode the A rows of this node once into the node-local row cache (disk-guarded; misses decode on the fly, bit-identical)
ALOE_ROWCACHE=$R/rowcache $PY -I $V/tools/stage_rowcache.py --node $K --workers ${ROWCACHE_WORKERS:-32} --verify 4 > $L/rowcache-p$K-$H.log 2>&1 && tail -1 $L/rowcache-p$K-$H.log || echo "$H store $K ROWCACHE FAILED (training still runs: on-the-fly decode)"
echo "$H store $K done $(TZ=America/Los_Angeles date +%T) scratch free $(df -BG --output=avail /scratch | tail -1)"

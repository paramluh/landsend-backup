#!/bin/bash
# aloe-vox-ft side work on ONE PPM host while PPM (job 22224) trains: build store K (node-local), then fetch store K's channel-A FLACs.
# niced 19 + ionice idle, --cpu-bind=none (never pinned onto PPM rank 0's cores), modest parallelism (build 8 workers, fetch 8 threads).
#   bash prep_host.sh K HOST [build|fetch|rowcache|both]   (rowcache: SPEED FIX 10-16, run after build + fetch)
set -uo pipefail
K=$1; H=$2; WHAT=${3:-both}; J=${PPM_JOB:-22224}
V=/data/home/amartyavarma/ppm/aloe-vox-ft; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
if [ "$WHAT" = build ] || [ "$WHAT" = both ]; then
  srun --overlap --jobid=$J -w $H -N1 -n1 -c8 --mem=64G --cpu-bind=none -J aloevox-build-p$K bash -c "cd /tmp; export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false S3_MIN_FRAMES=125 PPM_CD=2;
    df -BG /scratch | tail -1; nice -n 19 ionice -c3 $PY -I $V/code/build_voxB_rows.py build --store $K --out /scratch/amartyavarma/aloe-vox-ft/stores --workers 8 && rm -rf /scratch/amartyavarma/aloe-vox-build/p$K; df -BG /scratch | tail -1" || exit 1
fi
if [ "$WHAT" = fetch ] || [ "$WHAT" = both ]; then
  srun --overlap --jobid=$J -w $H -N1 -n1 -c4 --mem=16G --cpu-bind=none -J aloevox-fetch-p$K bash -c "cd /tmp; export GOOGLE_APPLICATION_CREDENTIALS=/data/home/amartyavarma/speechsdk-ebf4e74f9a4c.json PYTHONWARNINGS=ignore;
    nice -n 19 ionice -c3 $PY -I $V/tools/fetch_flacs.py --store $K --workers ${FETCH_WORKERS:-8} --root /scratch/amartyavarma/aloe-vox-ft/flac; df -BG /scratch | tail -1" || exit 1
fi
if [ "$WHAT" = rowcache ] || [ "$WHAT" = both ]; then             # SPEED FIX 10-16 (SPEED_FIX.md): decode the A rows of store K once (node-local row cache)
  srun --overlap --jobid=$J -w $H -N1 -n1 -c8 --mem=32G --cpu-bind=none -J aloevox-rowcache-p$K bash -c "cd /tmp; export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore S3_MIN_FRAMES=125 PPM_CD=2;
    nice -n 19 ionice -c3 $PY -I $V/tools/stage_rowcache.py --node $K --workers 8 --verify 4 --cache /scratch/amartyavarma/aloe-vox-ft/rowcache; df -BG /scratch | tail -1" || exit 1
fi

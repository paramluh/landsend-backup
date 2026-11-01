#!/bin/bash
#SBATCH -J curves-refresh -N1 -n1 -c1 --mem=4G -t 7-00:00:00 -o curves-refresh-%j.log
# Curves page refresher (CPU only, never on the login node): every 60 s rebuild stage3_<PAGE>.html / s3live_<PAGE>.json / ver_s3_<PAGE>.txt
# from s3branches_<PAGE>.json (its "doc" script reads each run's metrics.jsonl). Open pages poll s3live_<PAGE>.json every 60 s and reload
# themselves only when ver_s3_<PAGE>.txt changes. Optional: opd_prog.py every 5 s -> opd_prog.json (in-update progress block, opd_prog.js).
#   CURVES_DIR=$PWD PAGE=opd RUN_DIR=/path/to/run REF_DIR=/path/to/refs sbatch -w <cpu node> refresh.sbatch
: "${CURVES_DIR:?export CURVES_DIR=<this curves/ folder (the web root)>}"
PAGE=${PAGE:-opd}; PY=${PY:-python3}
export CURVES_DIR RUN_DIR REF_DIR
[ -n "${RUN_DIR:-}" ] && [ "${PROG:-1}" = 1 ] && { OPD_PROG_ROOT=$RUN_DIR $PY $CURVES_DIR/opd_prog.py 5 2>&1 | sed -u "s/^/prog /" & }
while true; do
  S3_PAGE=$PAGE timeout 300 $PY $CURVES_DIR/build_stage3.py $CURVES_DIR 2>&1 | sed "s/^/$(date +%a\ %H:%M) /" | tail -3
  sleep 60
done

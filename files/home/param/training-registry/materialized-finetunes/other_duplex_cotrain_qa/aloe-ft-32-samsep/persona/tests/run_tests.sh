#!/bin/bash
# PERSONA CPU tests (run on a CPU slot of slinky-4 / slinky-18 via srun; never on the login node). Writes persona/tests/out/*.json + DUMP_p0.5.txt
P=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/persona; Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $P/tests/out; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 NPROC=${NPROC:-16}
cd /tmp
case "$1" in
  ref) ( set -a; source $Q/vox/run_vox.env; set +a; $PY -I -B $P/tests/identity.py $Q/code data_aloe_v2vox $P/tests/out/ref_29345.json );;
  p0)  ( set -a; source $P/run_persona.env; set +a; export ALOE_PERSONA_P=0; $PY -I -B $P/tests/identity.py $P/code data_aloe_persona $P/tests/out/persona_p0.json );;
  p05) ( set -a; source $P/run_persona.env; set +a; export ALOE_PERSONA_P=0.5 DUMP_PATH=$P/DUMP_p0.5.txt; $PY -I -B $P/tests/identity.py $P/code data_aloe_persona $P/tests/out/persona_p05.json report );;
esac

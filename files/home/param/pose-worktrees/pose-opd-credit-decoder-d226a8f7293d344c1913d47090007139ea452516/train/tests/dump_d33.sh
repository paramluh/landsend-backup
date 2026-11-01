#!/bin/bash
# D33: dump test micros on a node holding the stores (CPU only). usage: bash tests/dump_d33.sh <out dir> <steps>
set -a; source /data/home/amartyavarma/ppm/code/prod_ppm.env; set +a
export PPM_NODE_SWAP= PPM_ALLOW_FOREIGN_DATA=1 OMP_NUM_THREADS=2
exec nice -n 19 /data/home/amartyavarma/venvs/qwen-s2/bin/python /data/home/amartyavarma/ppm/code/tests/dump_d33_micros.py "$1" "$2"

#!/bin/bash
#SBATCH -J npl-append -p all -N16 --ntasks-per-node=1 -c 48 --mem=200G -t 4:00:00
#SBATCH -o /data/home/amartyavarma/ppm/code-811/fresh/append/logs/ingest-%j.out
# 10-22 fresh pose append: the whole ingest (fresh/append/ingest_more.sh) in one allocation of the 16 training hosts (CPU only; 16 of 64 CPUs
# per host left for the HF watcher's uploads). usage:
#   sbatch -w $(cat /data/home/amartyavarma/ppm/code-811/nodes16_s7.txt) /data/home/amartyavarma/ppm/code-811/fresh/append/ingest_more.sbatch <SRC> [TAG]
exec bash /data/home/amartyavarma/ppm/code-811/fresh/append/ingest_more.sh "$@"

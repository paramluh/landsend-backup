#!/bin/bash
#SBATCH -J aloeft-initexp2 -p all -N2 -w slinky-2,slinky-5 --ntasks-per-node=1 -c4 --mem=24G -t 0:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft/logs/init_export_fix-%j.log
# Finish the aborted export 21970 of PPM dcp-11488: only slinky-2 (ranks 16-23) and slinky-5 (32-39) were killed mid-write.
C=/data/home/amartyavarma/ppm/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
rm -f $OUT/*.tmp
srun --cpu-bind=none nice -n 19 ionice -c3 $PY $C/ckpt_local_export.py --src /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-11488 --out $OUT || exit 1
srun -N1 -n1 -c1 --cpu-bind=none $PY $C/ckpt_local_export.py --out $OUT --finalize

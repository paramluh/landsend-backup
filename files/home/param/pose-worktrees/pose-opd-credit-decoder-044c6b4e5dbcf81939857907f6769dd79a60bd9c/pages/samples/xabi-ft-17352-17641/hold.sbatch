#!/bin/bash
#SBATCH -p all -N1 --gres=gpu:8 -c 48 --mem=500G -t 0-06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/samples/xabi-ft-17352-17641/logs/hold-%j-%N.log
# xabi-ft-17352-17641 samples: holder allocation; work runs as srun --overlap steps inside it.
echo "hold $(hostname -s) $(date)"; sleep 6h

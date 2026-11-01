#!/bin/bash
#SBATCH -p all -N1 --gres=gpu:8 -c 40 --mem=400G -t 0-02:00:00
#SBATCH -o /data/home/amartyavarma/ppm/samples/xabi-ft-17641-x76l/logs/hold-%j-%N.log
# X76L samples (10-23): holder allocation; work runs as srun --overlap steps inside it.
echo "hold $(hostname -s) $(date)"; sleep 2h

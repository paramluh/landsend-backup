#!/bin/bash
#SBATCH -J pose6-a3-cputest
#SBATCH -x slinky-9
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --nice=100
#SBATCH -t 00:20:00
#SBATCH -o /data/home/amartyavarma/ppm/pose6/worker/tests/logs/%x-%j.out
bash /data/home/amartyavarma/ppm/pose6/worker/tests/run_cpu_test.sh

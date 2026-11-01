#!/bin/bash
#SBATCH -J probe-17641-run -p all -N1 -w slinky-10 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=32G -t 0:10:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/tests/runtest/logs/run_test-%j.log
# dry test of run.sh's launch layout (GPU map from Slurm IDX, arbitrary-distribution hostfile, node checks) on the idle single node slinky-10:
# world 5 on one node, the trainer replaced by an env print. Nothing trained, nothing written outside tests/runtest (+ an empty scratch dir).
export FTB_NTASKS=5
exec bash /data/home/amartyavarma/ppm/aloe-v2pure-17641/tests/runtest/run_test.sh

#!/bin/bash
# run merge_A10.sbatch's body as an overlap step inside the running B2 job 29558 on slinky-18 (its node is full; main 86: use idle CPUs there)
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]+'); do unset $v; done
exec srun --overlap --jobid=29558 -w slinky-18 -N1 -n1 -c 8 --mem=96G --cpu-bind=none bash /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/tools/merge_A10.sbatch

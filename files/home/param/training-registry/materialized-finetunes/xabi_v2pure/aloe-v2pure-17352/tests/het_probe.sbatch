#!/bin/bash
#SBATCH -J probe-17352-het -p all -t 0:05:00 -o /data/home/amartyavarma/ppm/aloe-v2pure-17352/tests/het_probe-%j.log
#SBATCH -N1 -w slinky-15 --ntasks-per-node=8 --cpus-per-task=1 --gres=gpu:8 --mem=8G
#SBATCH hetjob
#SBATCH -N2 -w slinky-4,slinky-18 --ntasks-per-node=6 --cpus-per-task=1 --gres=gpu:6 --mem=8G
# aloe-v2pure-17352 launcher mechanics probe (no GPU work): het-job env, per-node task counts / SLURM_LOCALID, Slurm GPU IDX per node.
env | grep -E '^SLURM_(JOB_NODELIST|NNODES|NTASKS|HET|JOB_NUM_NODES|TASKS_PER_NODE|JOB_ID|JOBID)' | sort
scontrol show job -d $SLURM_JOB_ID | grep -E "JobId=|Nodes=.*GRES" 
echo "== one task per node"
srun --het-group=0,1 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "$(hostname -s) procid $SLURM_PROCID localid $SLURM_LOCALID ntasks $SLURM_NTASKS het $SLURM_HET_JOB_ID cvd ${CUDA_VISIBLE_DEVICES:-unset}"' | sort -V
echo "== full launch"
srun --het-group=0,1 --cpu-bind=none bash -c 'echo "$(hostname -s) procid $SLURM_PROCID localid $SLURM_LOCALID ntasks $SLURM_NTASKS cvd ${CUDA_VISIBLE_DEVICES:-unset}"' | sort -V

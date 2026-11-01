#!/bin/bash
#SBATCH -J probe-17352-het -p all -t 0:05:00 -o /data/home/amartyavarma/ppm/aloe-v2pure-17352/tests/het_probe2-%j.log
#SBATCH -N1 -w slinky-15 --ntasks-per-node=8 --cpus-per-task=1 --gres=gpu:8 --mem=8G
#SBATCH hetjob
#SBATCH -N2 -w slinky-4,slinky-18 --ntasks-per-node=6 --cpus-per-task=1 --gres=gpu:6 --mem=8G
# aloe-v2pure-17352 launcher probe 2: per-component srun syntax + code/task_ppm.sh rank / world / GPU mapping (python prints env only; no GPU use).
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST_HET_GROUP_0,$SLURM_JOB_NODELIST_HET_GROUP_1" | sort -V | paste -sd,)
export PPM_HOST_NTASKS=6,8,6 PPM_HOST_GPUS='slinky-4:0,1,2,3,4,5;slinky-15:0,1,2,3,4,5,6,7;slinky-18:2,3,4,5,6,7'
echo "PPM_HOSTS $PPM_HOSTS"
echo "== one per node"
srun --het-group=0 -N1 -n1 --cpu-bind=none : --het-group=1 -N2 -n2 --ntasks-per-node=1 --cpu-bind=none bash -c 'echo "$(hostname -s) procid $SLURM_PROCID"' | sort -V
echo "== task_ppm.sh launch"
srun --het-group=0 -N1 -n8 --ntasks-per-node=8 --cpus-per-task=1 --kill-on-bad-exit=1 : --het-group=1 -N2 -n12 --ntasks-per-node=6 --cpus-per-task=1 --kill-on-bad-exit=1 \
  bash /data/home/amartyavarma/ppm/aloe-v2pure-17352/code/task_ppm.sh -c 'import os,socket; e=os.environ; print(socket.gethostname().split(".")[0], "RANK", e["RANK"], "LOCAL", e["LOCAL_RANK"], "WORLD", e["WORLD_SIZE"], "LW", e["LOCAL_WORLD_SIZE"], "CVD", e.get("CUDA_VISIBLE_DEVICES"), "MASTER", e["MASTER_ADDR"])' | sort -k3 -n

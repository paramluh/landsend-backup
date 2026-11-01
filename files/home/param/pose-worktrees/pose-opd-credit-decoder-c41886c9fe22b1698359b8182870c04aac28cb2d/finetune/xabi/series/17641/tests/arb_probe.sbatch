#!/bin/bash
#SBATCH -J probe-17641-arb -p all -N3 -w slinky-4,slinky-15,slinky-18 --ntasks-per-node=8 --cpus-per-task=1 --mem=4G --gres=none -t 0:05:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2pure-17641/tests/arb_probe-%j.log
# launcher probe (no GPUs): uneven per-host task counts via srun --distribution=arbitrary + SLURM_HOSTFILE inside a uniform allocation
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export PPM_HOST_NTASKS=2,5,3
HF=/data/home/amartyavarma/ppm/aloe-v2pure-17641/tests/hostfile-$SLURM_JOB_ID
paste -d' ' <(echo $PPM_HOSTS | tr , '\n') <(echo $PPM_HOST_NTASKS | tr , '\n') | while read h c; do for i in $(seq 1 $c); do echo $h; done; done > $HF
cat $HF | uniq -c
SLURM_HOSTFILE=$HF srun -n10 --distribution=arbitrary --cpus-per-task=1 bash -c 'echo "$(hostname -s) procid $SLURM_PROCID localid $SLURM_LOCALID"' | sort -V

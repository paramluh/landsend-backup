#!/bin/bash
#SBATCH -p all -t 01:30:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/tests/logs/%x-%j.log
# reduced-depth correctness test of code/train_cotrain.py. usage: sbatch -J <name> -w <node> --gres=gpu:<n> --ntasks-per-node=<n> -c <c> --mem=<m> run_test.sbatch <overrides.env>
N=/data/home/amartyavarma/ppm/aloe-zoom-cotrain
set -a; source $N/aloe_zoom_cotrain.env; source $1; set +a
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR
echo "== test $SLURM_JOB_NAME job $SLURM_JOB_ID on $PPM_HOSTS ($SLURM_NTASKS_PER_NODE tasks/node) $(TZ=America/Los_Angeles date '+%F %T PT') overrides $1"; cat $1
srun -N$SLURM_NNODES --ntasks-per-node=1 bash -c 'echo "$(hostname -s): $(nvidia-smi --query-gpu=index,clocks.sm,memory.used --format=csv,noheader | tr "\n" ";") scratch free $(df -BG --output=avail /scratch | tail -1)"'
srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE --kill-on-bad-exit=1 bash $N/tests/task_test.sh $N/code/train_cotrain.py
rc=$?; echo "== exit $rc $(TZ=America/Los_Angeles date '+%T PT')"; exit $rc

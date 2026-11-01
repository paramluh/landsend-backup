#!/bin/bash
#SBATCH -J npl3-resume-wait -p all -N1 -n1 -c 1 --mem=2G --gres=none -w slinky-15 -t 3-00:00:00
#SBATCH -o /data/home/amartyavarma/ppm/code-811/fresh/append/npl3/logs/resume_wait-%j.out
# 10-23 delivery-3 resume waiter (doc + gates in npl3_resume_wait.sh; control/OPS_PLAYBOOK.md "Delivery-3 append"). CPU-only; launches the 8,1,1 resume from dcp-17641.
exec bash /data/home/amartyavarma/ppm/code-811/fresh/append/npl3_resume_wait.sh

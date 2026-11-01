#!/bin/bash
#SBATCH -J ppm811f-evallog -p all -w slinky-15 -N1 -n1 -c1 --mem=1G -t 14:00:00 -o /data/home/amartyavarma/ppm/code-811/fresh/logs/eval_log-%j.out
exec python3 /data/home/amartyavarma/ppm/code-811/fresh/eval_log16861v4.py

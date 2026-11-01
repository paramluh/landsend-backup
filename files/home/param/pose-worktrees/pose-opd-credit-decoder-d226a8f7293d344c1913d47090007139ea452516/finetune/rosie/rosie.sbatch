#!/bin/bash
#SBATCH -J pose-rosie -p all -N1 -w slinky-18 -x slinky-9 --ntasks-per-node=8 --cpus-per-task=4 --gres=gpu:8 --mem=530G -t 0-04:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-pose-rosie/logs/rosie-%j.log
# aloe-pose-rosie (10-27, user: "full finetune [the monoplex pose model] on rosie ... 2 epochs"): FULL fine-tune (no LoRA) of the monoplex pose
# FT export-357 (job 24272) on Rosie's own monoplex items (build_items.py: 88 qualifying tracks = 9.54 h of her 66 calls in 24272's stores),
# 2 epochs = 18 steps x 8 rows on slinky-18 x 8 (world 8: AdamW m / v offloaded to pinned host RAM, optoff.py). 24272's env otherwise.
exec bash /data/home/amartyavarma/ppm/aloe-pose-rosie/rosie_run.sh

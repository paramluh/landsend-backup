#!/bin/bash
scancel 30403; sleep 25
sbatch -w slinky-4 --gres=gpu:6 --mem=400G -c 24 /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/mixed.sbatch dup:/scratch/amartyavarma/pdsamples/export-498:/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/runs_dupM_b.json:/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/out/dupM mono:/data/home/amartyavarma/ppm/aloe-pose-mono-8020/exports/export-357:/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/sim/runs_mono_b.json:/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/out/mono render

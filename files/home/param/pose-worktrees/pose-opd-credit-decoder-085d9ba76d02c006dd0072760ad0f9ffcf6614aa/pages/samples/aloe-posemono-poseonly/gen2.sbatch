#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 12 --gres=gpu:4 --gres-flags=disable-binding --mem=200G -t 45 -J pmpo-gen2
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/gen2-%j.out
# pose-only generation of candidates 11-20 (rows2: R-010..R-019, 60 s windows), same settings as run_gen.sh (gen_poseonly.py: export-357,
# text + audio forced, pose generated, pose CFG 3 / top-k 30 / temp 1). slinky-4 GPUs must be 2-5 (0-1 demo, 6-7 throttled; guarded).
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/data/home/amartyavarma/ppm/aloe-pose-mono-8020/export-357; G=$W/work/gen_poseonly.py
for b in 0 1 6 7; do case ",$CUDA_VISIBLE_DEVICES," in *,$b,*) echo "ABORT: allocation holds GPU $b ($CUDA_VISIBLE_DEVICES)"; exit 3;; esac; done
source $FI/env_fast.sh
IFS=, read -ra ALL <<< "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES
echo "$(TZ=America/Los_Angeles date '+%F %T PT') start"
( export CUDA_VISIBLE_DEVICES=${ALL[0]},${ALL[1]}; exec nice -n 5 $PY -u $G --ckpt $CK --rows-dir $W/rows2 --out $W --rows R-010,R-011,R-012,R-013,R-014 --seed 20261021 ) > $W/logs/gen2-worker-2.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=${ALL[2]},${ALL[3]}; exec nice -n 5 $PY -u $G --ckpt $CK --rows-dir $W/rows2 --out $W --rows R-015,R-016,R-017,R-018,R-019 --seed 20261022 ) > $W/logs/gen2-worker-3.log 2>&1 &
wait
echo "$(TZ=America/Los_Angeles date '+%F %T PT') DONE: $(ls $W/runs-poseonly/*/DONE 2>/dev/null | wc -l)/20"

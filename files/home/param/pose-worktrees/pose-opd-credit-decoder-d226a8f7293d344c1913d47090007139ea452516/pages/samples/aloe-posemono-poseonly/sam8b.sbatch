#!/bin/bash
#SBATCH -p all -w slinky-4 -N1 -n1 -c 2 --gres=gpu:1 --gres-flags=disable-binding --mem=80G -t 30 -J pmpo-sam8b
#SBATCH -o /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/logs/sam8b-%j.out
case ",$CUDA_VISIBLE_DEVICES," in *,0,*|*,1,*|*,6,*|*,7,*) echo "ABORT: got GPU $CUDA_VISIBLE_DEVICES"; exit 3;; esac
export HF_HOME=/data/home/amartyavarma/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
echo "$(TZ=America/Los_Angeles date '+%T PT') GPU $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES
exec /data/home/amartyavarma/venvs/sam-pose/bin/python -u /data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly/work/gen_sam8b.py S-05,S-13,S-15,S-20

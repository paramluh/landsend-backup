#!/bin/bash
# 32B pose FT (22091 export-430) duo, no CFG, on the 8 NEW conversations B-008..015 (rows-duo), fast engine, ckpt staged on slinky-18 scratch.
W=/data/home/amartyavarma/ppm/samples/pose-hands-cmp; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/scratch/amartyavarma/fastinfer/ckpt/aloe-pose-ft/export-430
until grep -q "VERIFIED\|objects ok\|all ok" $W/logs/stage-posefk-s18.log 2>/dev/null || ! squeue -h -n hcmpstage | grep -q .; do sleep 10; done
echo "staging finished: $(tail -2 $W/logs/stage-posefk-s18.log)"
R=$(seq -f "B-%03g" 8 15 | paste -sd,)
srun -p all -w slinky-18 -N1 -n1 -c 8 --cpu-bind=none --gres=gpu:2 --gres-flags=disable-binding --mem=60G -t 90 -J hcmpduo \
  bash -c "source $FI/env_fast.sh; exec nice -n 5 \$PY -u $FI/samples_fast.py --ckpt $CK --rows $R --rows-dir $W/rows-duo --out $W/runs-duo --mode duo --batch 4" \
  > $W/logs/duo.log 2>&1
echo "duo exit $? $(date)"

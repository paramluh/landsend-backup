#!/bin/bash
# pose-only generation (10-19): export-357, text + audio forced, pose generated (gen_poseonly.py). One srun on slinky-4 gpu:4 (must be GPUs
# 2-5: 0-1 = the m1vox-pure demo, 6-7 thermally throttled; guarded): worker 0 = the 8 x 60 s rows on GPUs 2,3, worker 1 = the 2 x 120 s rows
# on GPUs 4,5. Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/data/home/amartyavarma/ppm/aloe-pose-mono-8020/export-357
L=$W/logs; G=$W/work/gen_poseonly.py
echo "[run_gen] $(TZ=America/Los_Angeles date '+%F %T PT') ckpt $CK"
BODY='BAD=$1; shift; IFS=, read -ra ALL <<< "$CUDA_VISIBLE_DEVICES"
for b in $BAD; do case ",$CUDA_VISIBLE_DEVICES," in *,$b,*) echo "ABORT: allocation holds GPU $b (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; exit 3;; esac; done
source '$FI'/env_fast.sh
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES
for spec in "$@"; do
  w=${spec%%|*}; rest=${spec#*|}; R=${rest%%|*}; P=${rest#*|}; i=${P%%,*}; j=${P#*,}
  ( export CUDA_VISIBLE_DEVICES=${ALL[$i]},${ALL[$j]}; echo "worker $w on $(hostname -s) GPUs $CUDA_VISIBLE_DEVICES rows $R seed $((20261019 + w))";
    exec nice -n 5 $PY -u '$G' --ckpt '$CK' --rows-dir '$W'/rows --out '$W' --rows $R --seed $((20261019 + w)) ) > '$L'/gen-worker-$w.log 2>&1 &
done
wait; echo "node $(hostname -s) gen done"'
srun -p all -w slinky-4 -N1 -n1 -c 12 --cpu-bind=none --gres=gpu:4 --gres-flags=disable-binding --mem=200G -t 45 -J pmpo-gen \
  bash -c "$BODY" _ "0 1 6 7" "0|R-000,R-001,R-002,R-003,R-004,R-005,R-006,R-007|0,1" "1|R-008,R-009|2,3" > $L/gen-node.log 2>&1
echo "[run_gen] $(TZ=America/Los_Angeles date '+%F %T PT') DONE: $(ls $W/runs-poseonly/*/DONE 2>/dev/null | wc -l)/10"

#!/bin/bash
# aloe-8b-vs-32b: the 8B page's 8 clips x {gen, force} with the 32B v2pure export-200, the runs-gen-k140 sampling (aloe-v2pure-200/work/
# gen_v2pure.py, unchanged, --pose-topk 30 --pose-topk-light 140: pose CFG 3, pose cb0 top-k 30, cb1-31 top-k 140, temp 1 everywhere, audio
# top-k 30, factorised text (gen), no rep-pen) -> runs-{gen,force}-k140/B-iii. 4 workers x 2 GPUs on slinky-15. Launch-only on login.
W=/data/home/amartyavarma/ppm/samples/aloe-8b-vs-32b; FI=/data/home/amartyavarma/ppm/fastinfer
G=/data/home/amartyavarma/ppm/samples/aloe-v2pure-200/work/gen_v2pure.py
CK=/data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200
WK=("B-000,B-001" "B-002,B-003" "B-004,B-005" "B-006,B-007")
for w in "${!WK[@]}"; do
  R=${WK[$w]}
  J=$(echo $R | tr , '\n' | sed 's/$/:gen/' | paste -sd,),$(echo $R | tr , '\n' | sed 's/$/:force/' | paste -sd,)
  srun -p all -w slinky-15 -N1 -n1 -c 8 --cpu-bind=none --gres=gpu:2 --mem=100G -t 60 -J v832gen \
    bash -c "echo worker $w on \$(hostname -s) GPUs \$CUDA_VISIBLE_DEVICES jobs $J; source $FI/env_fast.sh; \
      exec nice -n 5 \$PY -u $G --ckpt $CK --rows-dir $W/rows --out $W --jobs $J --seed $((20261018 + w)) --pose-topk 30 --pose-topk-light 140 --suffix=-k140" \
    > $W/logs/gen-worker-$w.log 2>&1 &
  sleep 3
done
wait
echo "[run_gen] DONE: gen $(ls $W/runs-gen-k140/*/DONE 2>/dev/null | wc -l)/8, force $(ls $W/runs-force-k140/*/DONE 2>/dev/null | wc -l)/8"

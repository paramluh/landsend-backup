#!/bin/bash
# aloe-v2pure LONG (10-18): 8 held-out 120 s windows (10-18 update: 2-minute clips), 2 s prefill + ~118 s all streams generated with the exact k140 column sampling
# (gen_long.py) -> runs-gen-k140. 4 groups of 2 rows (one seed per group, 20261118 + g): slinky-15 gpu:2 (6 of its GPUs are held by other
# jobs), slinky-4 gpu:4 (must be 2-5: 0-1 live demo, 6-7 throttled; guarded), slinky-7 gpu:2 (must be 6-7: 0-3 demos, 4-5 chat server;
# guarded; 40G host RAM). Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-v2pure-long; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200
L=$W/logs; G=$W/work/gen_long.py
echo "[run_long] $(TZ=America/Los_Angeles date '+%F %T PT') ckpt $CK"
BODY='BAD=$1; shift; IFS=, read -ra ALL <<< "$CUDA_VISIBLE_DEVICES"
for b in $BAD; do case ",$CUDA_VISIBLE_DEVICES," in *,$b,*) echo "ABORT: allocation holds GPU $b (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"; exit 3;; esac; done
source '$FI'/env_fast.sh
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,memory.used --format=csv,noheader -i $CUDA_VISIBLE_DEVICES
for spec in "$@"; do
  tag=${spec%%|*}; rest=${spec#*|}; GR=${rest%%|*}; P=${rest#*|}; i=${P%%,*}; j=${P#*,}
  ( export CUDA_VISIBLE_DEVICES=${ALL[$i]},${ALL[$j]}; echo "worker $tag on $(hostname -s) GPUs $CUDA_VISIBLE_DEVICES groups $GR";
    exec nice -n 5 $PY -u '$G' --ckpt '$CK' --rows-dir '$W'/rows --out '$W' --groups "$GR" --variants k140 --post-modes gen ) > '$L'/worker-$tag.log 2>&1 &
done
wait; echo "node $(hostname -s) gen done"'
srun -p all -w slinky-15 -N1 -n1 -c 8 --cpu-bind=none --gres=gpu:2 --gres-flags=disable-binding --mem=60G -t 30 -J v2plong \
  bash -c "$BODY" _ "" "g0|20261118=M-000:gen,M-001:gen|0,1" > $L/node-s15.log 2>&1 &
srun -p all -w slinky-4 -N1 -n1 -c 16 --cpu-bind=none --gres=gpu:4 --gres-flags=disable-binding --mem=120G -t 30 -J v2plong \
  bash -c "$BODY" _ "0 1 6 7" "g1|20261119=M-002:gen,M-003:gen|0,1" "g2|20261120=M-004:gen,M-005:gen|2,3" > $L/node-s4.log 2>&1 &
srun -p all -w slinky-7 -N1 -n1 -c 8 --cpu-bind=none --gres=gpu:2 --gres-flags=disable-binding --mem=40G -t 30 -J v2plong \
  bash -c "$BODY" _ "0 1 2 3 4 5" "g3|20261121=M-006:gen,M-007:gen|0,1" > $L/node-s7.log 2>&1 &
wait
echo "[run_long] $(TZ=America/Los_Angeles date '+%F %T PT') DONE: $(ls $W/runs-gen-k140/*/DONE 2>/dev/null | wc -l)/8"

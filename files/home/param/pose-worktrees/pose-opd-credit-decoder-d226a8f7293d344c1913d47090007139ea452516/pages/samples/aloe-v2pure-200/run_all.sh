#!/bin/bash
# aloe-v2pure-200 samples: 10 held-out clips x 2 modes (gen = all streams sampled, force = text teacher-forced) with the fast engine,
# 9 workers x 2 GPUs (both modes of a clip in one session): slinky-15 GPUs 0-7, slinky-4 GPUs 0-5 (6-7 throttled: guarded), slinky-7 the 4 GPUs
# the two live demos (23090 / 23091, GPUs 0-3) do not hold. Checkpoint = the consolidated /data export (the node-local exports hold only each
# node's 8 of the 24 shards). Launch-only on the login node; all compute on the compute nodes.
W=/data/home/amartyavarma/ppm/samples/aloe-v2pure-200; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200
mkdir -p $W/logs
WK=("slinky-15|B-007" "slinky-15|B-005" "slinky-15|B-006" "slinky-15|B-000" "slinky-4|B-008" "slinky-4|B-002" "slinky-4|B-004"
    "slinky-7|B-001" "slinky-7|B-003,B-009")
echo "[run_all] $(TZ=America/Los_Angeles date '+%F %T PT') ${#WK[@]} workers, ckpt $CK"
for w in "${!WK[@]}"; do
  N=${WK[$w]%%|*}; R=${WK[$w]#*|}
  J=$(echo $R | tr , '\n' | sed 's/$/:gen/' | paste -sd,),$(echo $R | tr , '\n' | sed 's/$/:force/' | paste -sd,)
  case $N in slinky-4) BAD="6 7";; slinky-7) BAD="0 1 2 3";; *) BAD="";; esac
  srun -p all -w $N -N1 -n1 -c 8 --cpu-bind=none --gres=gpu:2 --gres-flags=disable-binding --mem=100G -t 60 -J v2psmp \
    bash -c "for b in $BAD; do case \",\$CUDA_VISIBLE_DEVICES,\" in *,\$b,*) echo \"ABORT: got GPU \$b (CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES) on $N\"; exit 3;; esac; done; \
      echo \"worker $w on \$(hostname -s) GPUs \$CUDA_VISIBLE_DEVICES jobs $J\"; source $FI/env_fast.sh; \
      exec nice -n 5 \$PY -u $W/work/gen_v2pure.py --ckpt $CK --rows-dir $W/rows --out $W --jobs $J --seed $((20261018 + w))" \
    > $W/logs/worker-$w.log 2>&1 &
  sleep 4
done
wait
echo "[run_all] $(TZ=America/Los_Angeles date '+%F %T PT') DONE: gen $(ls $W/runs-gen/*/DONE 2>/dev/null | wc -l)/10, force $(ls $W/runs-force/*/DONE 2>/dev/null | wc -l)/10"

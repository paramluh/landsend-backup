#!/bin/bash
# aloe-v2pure-200 k140 variant (10-18): the SAME 10 clips, same 2 s prefill, both modes, pose CFG 3 + pose cb0 top-k 30 + pose cb1-31 top-k 140
# (everything else = run_all.sh: temp 1, audio top-k 30, no rep-pen, factorised text in mode gen) -> runs-gen-k140 / runs-force-k140.
# 5 workers x 2 GPUs: slinky-4 GPUs 0-5 (6-7 throttled: guarded), slinky-7 GPUs 4-7 (demos 23090 / 23091 hold 0-3: guarded); slinky-15 left
# to the tt-v3 test. Launch-only on the login node.
W=/data/home/amartyavarma/ppm/samples/aloe-v2pure-200; FI=/data/home/amartyavarma/ppm/fastinfer
CK=/data/home/amartyavarma/ppm/aloe-v2pure-ft/export-200
WK=("slinky-4|B-007,B-005" "slinky-4|B-006,B-000" "slinky-4|B-008,B-002" "slinky-7|B-004,B-001" "slinky-7|B-003,B-009")
echo "[run_k140] $(TZ=America/Los_Angeles date '+%F %T PT') ${#WK[@]} workers, ckpt $CK"
for w in "${!WK[@]}"; do
  N=${WK[$w]%%|*}; R=${WK[$w]#*|}
  J=$(echo $R | tr , '\n' | sed 's/$/:gen/' | paste -sd,),$(echo $R | tr , '\n' | sed 's/$/:force/' | paste -sd,)
  case $N in slinky-4) BAD="6 7";; slinky-7) BAD="0 1 2 3";; *) BAD="";; esac
  srun -p all -w $N -N1 -n1 -c 8 --cpu-bind=none --gres=gpu:2 --gres-flags=disable-binding --mem=60G -t 60 -J v2pk140 \
    bash -c "for b in $BAD; do case \",\$CUDA_VISIBLE_DEVICES,\" in *,\$b,*) echo \"ABORT: got GPU \$b (CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES) on $N\"; exit 3;; esac; done; \
      echo \"worker $w on \$(hostname -s) GPUs \$CUDA_VISIBLE_DEVICES jobs $J\"; source $FI/env_fast.sh; \
      exec nice -n 5 \$PY -u $W/work/gen_v2pure.py --ckpt $CK --rows-dir $W/rows --out $W --jobs $J --seed $((20261118 + w)) \
        --pose-topk 30 --pose-topk-light 140 --suffix=-k140" \
    > $W/logs/k140-worker-$w.log 2>&1 &
  sleep 4
done
wait
echo "[run_k140] $(TZ=America/Los_Angeles date '+%F %T PT') DONE: gen $(ls $W/runs-gen-k140/*/DONE 2>/dev/null | wc -l)/10, force $(ls $W/runs-force-k140/*/DONE 2>/dev/null | wc -l)/10"

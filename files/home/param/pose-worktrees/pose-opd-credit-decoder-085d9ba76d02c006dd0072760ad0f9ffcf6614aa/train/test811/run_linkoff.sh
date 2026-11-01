#!/bin/bash
# LINK OFF port test (2 GPUs, hold job $1): L1 = PPM mode with the link ON (as PPM), 3 steps, saved; L2 = 8,1,1 + PPM_DECP_BB_OFF=1 resumed
# from it (the switch): the link's AdamW state set aside (digest-checked), startup / per-micro / per-step asserts, start eval vs L1's eval.
J=$1; D=/data/home/amartyavarma/ppm/code-811; L=/data/home/amartyavarma/ppm/logs/ppm811test; C=/scratch/amartyavarma/ppm811-test/ckpt/ppm811-t2
run() { tag=$1; shift; echo "== $(TZ=America/Los_Angeles date '+%T PT') $tag: $*"
  env "$@" PPM_RUN=ppm811-t$tag MASTER_PORT=$((29900 + RANDOM % 90)) srun --jobid $J --overlap -n 2 -c 4 --gres=gpu:2 --cpu-bind=none \
    bash $D/test811/task_1node.sh $D/test811/t2.env $D/train_ppm.py > $L/t$tag.log 2>&1
  echo "== $(TZ=America/Los_Angeles date '+%T PT') $tag exit $?: $(grep -ac 'step [0-9]*/' $L/t$tag.log) step lines $(grep -am1 -E 'Traceback|Error' $L/t$tag.log)"; }
run L1 PPM_S811=0 PPM_DECP_BB_OFF=0 PPM_GA_SEQ=pose,speech,speech,text PPM_MAX_STEPS=3 PPM_FINAL_SAVE=1
run L2 PPM_RESUME_FROM=$C/dcp-3 PPM_MAX_STEPS=5 PPM_811_CHECK_STEPS=2 PPM_811_EVAL_REF=/data/home/amartyavarma/ppm/runs-811test/ppm811-tL1/metrics.jsonl

#!/bin/bash
# 8,1,1 2-GPU resume / equivalence chain (hold job $1 on slinky-18). Needs dcp-3 from R1 (PPM mode, 3 steps, saved).
#  R4  PPM mode (S811=0) resumed from dcp-3, 2 steps                      -> reference losses for steps 4-5
#  R5  8,1,1 mode with PPM's GA, all micros full (coefficients 1/4 = PPM) -> must equal R4 (step 4 exactly: same state, same rows)
#  R2  8,1,1 (pose*8, speech, text; 1 full + 7 frozen pose) resumed from dcp-3 = the SWITCH at 3; SAVE_NOW at 5 (dcp-5), runs to 7
#  R3  8,1,1 resumed from its own dcp-5 -> rows / step-6 losses must equal R2's
J=$1; D=/data/home/amartyavarma/ppm/code-811; L=/data/home/amartyavarma/ppm/logs/ppm811test; C=/scratch/amartyavarma/ppm811-test/ckpt/ppm811-t2
run() { tag=$1; shift; echo "== $(TZ=America/Los_Angeles date '+%T PT') $tag: $*"
  env "$@" PPM_RUN=ppm811-t$tag MASTER_PORT=$((29900 + RANDOM % 90)) srun --jobid $J --overlap -n 2 -c 4 --gres=gpu:2 --cpu-bind=none \
    bash $D/test811/task_1node.sh $D/test811/t2.env $D/train_ppm.py > $L/t$tag.log 2>&1
  echo "== $(TZ=America/Los_Angeles date '+%T PT') $tag exit $?: $(grep -ac 'step [0-9]*/' $L/t$tag.log) step lines $(grep -am1 -E 'Traceback|Error' $L/t$tag.log)"; }
REF=/data/home/amartyavarma/ppm/runs-811test/ppm811-tR1/metrics.jsonl
run R4 PPM_S811=0 PPM_GA_SEQ=pose,speech,speech,text PPM_RESUME_FROM=$C/dcp-3 PPM_MAX_STEPS=5
run R5 PPM_S811=1 PPM_GA_SEQ=pose,speech,speech,text PPM_BB_MODE=full*4 PPM_RESUME_FROM=$C/dcp-3 PPM_MAX_STEPS=5 PPM_811_EVAL_REF=$REF PPM_811_CHECK_STEPS=1
run R2 PPM_RESUME_FROM=$C/dcp-3 PPM_MAX_STEPS=7 PPM_TEST_SAVENOW_AT=5 PPM_811_EVAL_REF=$REF PPM_811_CHECK_STEPS=1 PPM_811_LOG_ROWS=4
run R3 PPM_RESUME_FROM=$C/dcp-5 PPM_MAX_STEPS=7 PPM_811_CHECK_STEPS=1 PPM_811_LOG_ROWS=2

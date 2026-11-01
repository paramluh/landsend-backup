#!/bin/bash
#SBATCH -J aloespk-ideval -p all -N1 --ntasks-per-node=8 --gres=gpu:8 -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/logs/aloespk-ideval-%j.log
# IDENTITY EVAL-ONLY (10-24, PREPARED, NOT RUN; main decides): the LoRA trainer copy (lora/code/train_spklora_vox.py) in identity mode (no speaker,
#   LoRA rank 0, nothing frozen = 29345's code path) on ONE host, env lora/run_identity_eval.env (= 29345's run_vox.env + layout / init / dirs),
#   ALOE_EVALVOX=1: build, load the init (its own vadapter + the fp32 theta), the startup checks (plan d1a3e3721328da0a, ledgers, same-as-v2,
#   cfg diffs vs 26232 / 29345), the step-0 eval of the 316 standard micros, exit. No training step, no export, no save.
# expect (init = 29345's export-249): EVAL step 0 eval_speech/word_ce_norm ~= 1.8023 (29345 step 249); with w-17127 staged as the init: ~= 1.8484.
# usage: sbatch -w slinky-18 --cpus-per-task=6 --mem=300G /data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora/run_identity_eval.sbatch
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker/lora; C=$Q/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
set -a; source ${QQA_ENV:-$Q/run_identity_eval.env}; set +a
export QQA_INIT MASTER_PORT=29895 COTRAIN_RESUME=
[ "${ALOE_EVALVOX:-0}" = 1 ] && [ "${ALOE_LORA_RANK:-0}" = 0 ] && [ -z "${ALOE_SPK_ACCOUNT:-}" ] && [ -z "${ALOE_FREEZE_GROUPS:-}" ] \
  || { echo "ABORT: identity eval = EVALVOX, LoRA rank 0, no speaker, nothing frozen"; exit 1; }
for f in $C/model_ppm.py $C/model_lora.py $C/data_aloe_v2vox.py $C/data_aloe_spk.py $C/spk_plan.py $C/safety_ppm.py $C/train_spklora_vox.py; do
  $PY -m py_compile $f || { echo "COMPILE FAIL $f"; exit 1; }; done
[ "$(hostname -s)" = "$PPM_HOSTS" ] || { echo "ABORT: host $(hostname -s) != PPM_HOSTS $PPM_HOSTS"; exit 1; }
export PPM_HOSTS PPM_HOST_NRANKS PPM_HOST_GPUS
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
mkdir -p $QQA_CONTROL_DIR $Q/logs
echo "== identity eval job $SLURM_JOB_ID on $PPM_HOSTS (world 8, logical world $ALOE_LOGICAL_WORLD) $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT"
srun -N1 -n1 --ntasks-per-node=1 bash -c '
FR=$ALOE_FLAC_ROOT
if [ -L "$FR" ]; then [ "$(readlink "$FR")" = "$ALOE_SHARED_FLAC" ] || { echo "ABORT: $FR -> $(readlink "$FR") != $ALOE_SHARED_FLAC"; exit 1; }
elif [ -e "$FR" ]; then echo "ABORT: $FR exists as a real path"; exit 1
else mkdir -p "$(dirname "$FR")" && ln -s "$ALOE_SHARED_FLAC" "$FR" || exit 1; fi
for l in 0 1 2 3 4 5 6 7; do grep -q "\"plan_digest\": \"$QQA_REF_PLAN_DIGEST\"" "$FR/_ledger/DONE_v2vox_node$l.json" || { echo "ABORT: ledger $l"; exit 1; }; done
[ -f "$QQA_INIT/COMPLETE" ] && grep -q "\"vgain_theta_fp32\"" "$QQA_INIT/meta.json" || { echo "ABORT: init not COMPLETE / no fp32 theta"; exit 1; }
echo "preflight ok: $(ls $QQA_INIT/rank*.pt | wc -l) init rank files; GPUs $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\n" " ")"' || exit 1
srun -N1 -n8 --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_samsep.sh $C/train_spklora_vox.py
rc=$?; echo "== identity eval exit $rc $(TZ=America/Los_Angeles date '+%T PT') (the EVAL step 0 line above: compare eval_speech/word_ce_norm with 1.8023)"; exit $rc

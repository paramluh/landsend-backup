#!/bin/bash
#SBATCH -p all -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/tests/logs/%x-%j.log
# [g] gradcheck run (COTRAIN_TEST_GRADCHECK=1, exits after it) then RESUME test (reduced depth): A = 6 steps uninterrupted; B = the same run stopped after step 3 (COTRAIN_TEST_STOP_AT=3 = STOP_AFTER_SAVE:
# node-local full-state save); C = resume from B's save, steps 4..6. Then tests/compare_resume.py: C's EVAL at its resume step == B's EVAL step 3
# (forward-only: exact weights), per-step data counts / LRs of C == A at steps 4..6, C's losses vs A's (nondeterministic backward: close).
# usage: sbatch -J t3 -w <node(s)> --gres=gpu:<g> --ntasks-per-node=<g> -c <c> --mem=<m> t_resume.sbatch <overrides.env>
N=/data/home/amartyavarma/ppm/aloe-zoom-cotrain
set -a; source $N/aloe_zoom_cotrain.env; source $1; set +a
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
J=$SLURM_JOB_ID
echo "== resume test job $J on $PPM_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT')"
run() {  # <tag> [env assignments...]
  local tag=$1; shift
  echo "== run $tag $(TZ=America/Los_Angeles date '+%T PT') $*"
  env QQA_RUN=cotrain-$SLURM_JOB_NAME$tag-j$J QQA_CONTROL_DIR=$N/tests/control-$SLURM_JOB_NAME$tag-$J "$@" \
    srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE --kill-on-bad-exit=1 bash $N/tests/task_test.sh $N/code/train_cotrain.py
  echo "== run $tag exit $? $(TZ=America/Los_Angeles date '+%T PT')"
}
# p: the PAD-SLOT case = zoom logical world 2 x the aloe one -> 2 slots per GPU, the aloe ranks' 2nd slot is a pad (the template's dummy-micro
#    path incl. its step-0 loss check); gradcheck + 3 steps, no saves / exports.   RUNS (env) selects / orders the runs (default g a b c p).
for r in ${RUNS:-g a b c p}; do
  case $r in
    g) run g COTRAIN_TEST_GRADCHECK=1 COTRAIN_TEST_GRADCHECK_EXIT=1;;
    a) run a;;
    b) run b COTRAIN_TEST_STOP_AT=3;;
    c) run c COTRAIN_RESUME=$COTRAIN_RESUME_ROOT/cotrain-${SLURM_JOB_NAME}b-j$J/dcp-3;;
    p) run p ZOOM_LOGICAL_WORLD=$((2 * ZOOM_LOGICAL_WORLD)) COTRAIN_TEST_GRADCHECK=1 QQA_TEST_STEPS=3 QQA_EXPORT_STEPS= MASTER_PORT=29885;;
  esac
done
srun -N1 -n1 --overlap bash -c "/data/home/amartyavarma/venvs/qwen-s2/bin/python -I $N/tests/compare_resume.py $N/tests/runs cotrain-${SLURM_JOB_NAME}a-j$J cotrain-${SLURM_JOB_NAME}b-j$J cotrain-${SLURM_JOB_NAME}c-j$J"
rc=$?
srun -N$SLURM_NNODES --ntasks-per-node=1 --overlap bash -c 'echo "$(hostname -s): $(du -sh /scratch/amartyavarma/aloe-zoom-cotrain-test/resume /scratch/amartyavarma/aloe-zoom-cotrain-test/exports 2>/dev/null | tr "\n" " ")"; ls /scratch/amartyavarma/aloe-zoom-cotrain-test/exports/export-2 2>/dev/null | tr "\n" " "; echo; rm -rf /scratch/amartyavarma/aloe-zoom-cotrain-test && echo "$(hostname -s): test files deleted"'
echo "== compare rc $rc"
exit $rc

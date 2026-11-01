#!/bin/bash
#SBATCH -p all -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/tests/logs/%x-%j.log
# RESUME test (reduced depth): A = 6 steps uninterrupted; B = the same run stopped after step 3 (COTRAIN_TEST_STOP_AT=3 = STOP_AFTER_SAVE:
# node-local full-state save); C = resume from B's save, steps 4..6. Then tests/compare_resume.py: C's EVAL at its resume step == B's EVAL step 3
# (forward-only: exact weights), per-step data counts / LRs of C == A at steps 4..6, C's losses vs A's (nondeterministic backward: close).
# usage: sbatch -J cotrain-t2 -w <node> --gres=gpu:2 --ntasks-per-node=2 -c 3 --mem=200G t2_resume.sbatch <overrides.env>
N=/data/home/amartyavarma/ppm/aloe-zoom-cotrain
set -a; source $N/aloe_zoom_cotrain.env; source $1; set +a
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
J=$SLURM_JOB_ID
echo "== resume test job $J on $PPM_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT')"
run() {  # <tag> [env assignments...]
  local tag=$1; shift
  echo "== run $tag $(TZ=America/Los_Angeles date '+%T PT') $*"
  env QQA_RUN=cotrain-t2$tag-j$J QQA_CONTROL_DIR=$N/tests/control-t2$tag-$J "$@" \
    srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE --kill-on-bad-exit=1 bash $N/tests/task_test.sh $N/code/train_cotrain.py
  echo "== run $tag exit $? $(TZ=America/Los_Angeles date '+%T PT')"
}
run a
run b COTRAIN_TEST_STOP_AT=3
run c COTRAIN_RESUME=$COTRAIN_RESUME_ROOT/cotrain-t2b-j$J/dcp-3
srun -N1 -n1 --overlap bash -c "/data/home/amartyavarma/venvs/qwen-s2/bin/python -I $N/tests/compare_resume.py $N/tests/runs cotrain-t2a-j$J cotrain-t2b-j$J cotrain-t2c-j$J"
rc=$?
du -sh $COTRAIN_RESUME_ROOT 2>/dev/null; rm -rf /dev/shm/amartyavarma-cotrain-test; echo "== test state in /dev/shm deleted; compare rc $rc"
exit $rc

#!/bin/bash
#SBATCH -J aloevox-gputest -p all -N2 -w slinky-4,slinky-7 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=5 --mem=540G -t 2:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-vox-ft/logs/test/gputest-%j.log
# GPU CHECK of the real trainer on the 2 spare nodes (world 16), NOTHING written to /data except the small metrics/log dir under tests/runs:
#   init -> vadapter init -> step-0 held-out eval over ALL 531 eval items (node 0 = stores 0,2,4,6; node 1 = 1,3,5,7) -> pack-loss check ->
#   QQA_TEST_STEPS full forward/backward steps (Voxtral -> adapter -> channel A; B text / cb0 / light / pose cb0 / pose light; both decoders' nanosteps
#   + their AdamW) with the main AdamW step SKIPPED (QQA_TEST_NO_MAIN_OPT=1: 16 GPUs cannot hold the 64-GPU run's main Adam state); no exports.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-vox-ft; C=$Q/code
export QQA_INIT=${QQA_INIT:-/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-12230}
set -a; source $Q/aloe_vox.env; set +a
export QQA_LOG_ROOT=$Q/tests/runs QQA_CONTROL_DIR=$Q/tests/control QQA_EXPORT=/scratch/amartyavarma/aloe-vox-ft/test-export-never QQA_EXPORT_STEPS= \
  ALOE_LOCAL_EXPORT_ROOT=/scratch/amartyavarma/aloe-vox-ft/test-export-never QQA_TEST_STEPS=${QQA_TEST_STEPS:-6} QQA_TEST_NO_MAIN_OPT=1 \
  ALOE_TEST_CONVS=$Q/tests/test_convs.json MASTER_PORT=29853 QQA_RUN=ppm-aloe-vox-gputest
export PPM_HOSTS=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,)
export QQA_RUN=$QQA_RUN-j$SLURM_JOB_ID
echo "== gpu test $SLURM_JOB_ID on $PPM_HOSTS $(TZ=America/Los_Angeles date '+%F %T PT') init $QQA_INIT"
srun -N$SLURM_NNODES -n$SLURM_NNODES --ntasks-per-node=1 bash -c 'find /dev/shm -maxdepth 1 -user $USER \( -name "nccl-*" -o -name "torch_*" -o -name "sem.mp-*" \) -delete 2>/dev/null; echo "$(hostname -s) shm $(df -h --output=used /dev/shm | tail -1)"'
srun --ntasks-per-node=8 --kill-on-bad-exit=1 bash $C/task_ppm.sh $C/train_aloe_vox.py
echo "== exit $? $(TZ=America/Los_Angeles date '+%T PT')"

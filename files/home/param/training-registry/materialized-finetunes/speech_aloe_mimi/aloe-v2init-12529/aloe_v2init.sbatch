#!/bin/bash
#SBATCH -J aloev2i-12529 -p all -N8 --ntasks-per-node=8 --gres=gpu:8 --cpus-per-task=5 --mem=540G -t 12:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-v2init-12529/logs/aloev2i-12529-%j.log
# aloe-v2init-12529 (10-17) = aloe-v2-ft/aloe_v2.sbatch (job 22383): store pre-flight (verify / restage the 4 Model-1 stores, digests asserted)
# + the init conversion: PPM dcp-12529 (node-local, 16 PPM hosts) -> local-12529 (ppm-local-bf16-shards-v1, = how local-11488 was made, with
# aloe-vox-ft/code/ckpt_local_export.py): THIS job converts the ranks its 8 hosts hold (nice 19); the other 8 hosts' ranks are converted by an
# overlap step in job 22562; we wait for all 128 rank files + rep.pt + meta.json, finalize (COMPLETE), then run_train.sh.
set -uo pipefail
Q=/data/home/amartyavarma/ppm/aloe-v2init-12529; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; CV=/data/home/amartyavarma/ppm/aloe-vox-ft/code/ckpt_local_export.py
SRC=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-12529; OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-12529
echo "== aloe-v2init-12529 job $SLURM_JOB_ID on $(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort -V | paste -sd,) $(TZ=America/Los_Angeles date '+%F %T PT')"
srun -N8 -n8 --ntasks-per-node=1 -c 5 --mem=32G --gres=none bash $Q/stage_check.sh || { echo "ABORT: store staging failed"; exit 1; }
srun -N8 -n8 --ntasks-per-node=1 -c 1 --mem=1G --gres=none bash -c 'echo "$(hostname -s) scratch $(df -h /scratch | tail -1)"'
if [ ! -f $OUT/COMPLETE ]; then
  echo "== ckconv (this job's 8 hosts) $(TZ=America/Los_Angeles date '+%T PT')"
  srun -N8 -n8 --ntasks-per-node=1 -c 5 --mem=64G --gres=none nice -n 19 $PY $CV --src $SRC --out $OUT || { echo "ABORT: ckconv failed"; exit 1; }
  for i in $(seq 720); do
    n=$(ls $OUT 2>/dev/null | grep -c '^rank[0-9]*\.pt$'); [ "$n" = 128 ] && [ -f $OUT/rep.pt ] && [ -f $OUT/meta.json ] && break; sleep 10
  done
  echo "== rank files $(ls $OUT | grep -c '^rank[0-9]*\.pt$') / 128 $(TZ=America/Los_Angeles date '+%T PT')"
  srun -N1 -n1 -c 1 --mem=4G --gres=none $PY $CV --out $OUT --finalize || { echo "ABORT: finalize failed"; exit 1; }
fi
exec bash $Q/run_train.sh

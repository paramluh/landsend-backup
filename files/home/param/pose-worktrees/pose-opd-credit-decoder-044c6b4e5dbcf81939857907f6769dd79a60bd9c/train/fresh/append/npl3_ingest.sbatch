#!/bin/bash
#SBATCH -J npl3-ingest -p all -N1 -n1 -c 2 --mem=4G --gres=none -w slinky-15 -t 6:00:00
#SBATCH -o /data/home/amartyavarma/ppm/code-811/fresh/append/npl3/logs/ingest-%j.out
# 10-23 delivery-3 ingest driver (control/OPS_PLAYBOOK.md "Delivery-3 append"): fresh/append/ingest_more.sh for tag npl3, every host step inside the CPU-only holder job
# (npl3_hold.sbatch, HOLD=<job id>) via srun --overlap: -c 14 / 26G per host, nice 19 + ionice best-effort 7; HF at 80 files/s/host (16 hosts =
# 1.28k/s < the hub's ~1.6k/s resolver quota). Placement balanced over plan 1 + npl2 + npl3 rows per host (PRIOR_TAGS=npl2, BALANCE=tail);
# dry run / guard against the resume checkpoint dcp-17641 (job 27204's final save); fills PPM_MAX_STEPS in prod_811_fresh_append3.env.
: "${HOLD:?HOLD=<holder job>}"
DRV=$SLURM_JOB_ID
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*'); do unset $v; done   # (this driver's own allocation must not leak into the srun --jobid $HOLD steps)
echo "driver $DRV -> holder $HOLD"
export OVERLAP_JOB=$HOLD OVL_CPT=14 OVL_MEM=26G OVL_NICE="nice -n19 ionice -c2 -n7" HF_RATE=80 THREADS_HF=64
export ENVF=/data/home/amartyavarma/ppm/code-811/prod_811_fresh_append3.env PRIOR_TAGS=npl2 RESUME_STEP=17641 FILL_MARKER="10-23 FRESH POSE APPEND 3" REF_JOB=27204
bash /data/home/amartyavarma/ppm/code-811/fresh/append/ingest_more.sh hf://buckets/InternalCan/pose-llm-data/new-pose-llm-data-delivery3 npl3
rc=$?; echo "== ingest rc $rc $(TZ=America/Los_Angeles date '+%F %T PT')"; exit $rc

#!/bin/bash
#SBATCH -J c1-test17641 -p all -w slinky-18 -N1 -n1 -c 2 --mem=24G --nice=0 -t 0-00:40:00 --gres=none
#SBATCH -o /data/home/amartyavarma/ppm/pose6/mono17850/convert/logs/test17641-%j.log
# C1 verify (1): the HF range-read converter on dcp-17641 (HF InternalCan/ppm-v1-c, pinned revision) for ranks 0, 64 (logical slinky-10 =
# HF dir slinky-7), 127 + rep.pt + meta.json -> test dir in pose6/mono17850/convert (own test output, removed after), compared with the
# existing /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641 (keys/order/dtype/shape/torch.equal per tensor + file sha256).
D=/data/home/amartyavarma/ppm/pose6/mono17850/convert
echo "== $(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT') job $SLURM_JOB_ID on $(hostname -s), cpus $SLURM_CPUS_PER_TASK"
/data/home/amartyavarma/venvs/qwen-s2/bin/python -I -u $D/hf_local_export.py --revision c9b2213751cb3f2ede1dce2c5b9ed275b5cc3ca2 \
  --tag dcp-17641 --ranks 0,64,127 --rep 1 --meta 1 --out $D/test17641 --ref /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641 \
  --stats $D/test17641_stats.json --threads 16 --chunk-mb 32
echo "== rc $? $(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')"

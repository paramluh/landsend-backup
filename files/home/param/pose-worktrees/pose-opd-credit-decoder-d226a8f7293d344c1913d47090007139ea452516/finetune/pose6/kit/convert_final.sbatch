#!/bin/bash
#SBATCH -J pose6-convert -p all -N16 --ntasks-per-node=1 --cpus-per-task=4 --mem=24G -t 0-01:00:00 -x slinky-9
#SBATCH -o /data/home/amartyavarma/ppm/pose6/kit/logs/convert-%j.log
# pose6 kit (B3) STEP "convert": run 31101's FINAL node-local save dcp-19382 -> /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-19382
# (ppm-local-bf16-shards-v1, world 128, ~88.4 GB on /data) = the FT base. SUBMITTED BY MAIN ONLY (kit/convert_final.sh prints the exact
# command and checks first; the kit never submits it). Needs: 31101 ended ('done 19382 steps'), the 16 hosts free (before the FT and before the
# MoT resume), /data free - 89 GB >= 300 GB. CPU only (one task per host, nice 19); reads the save, writes only OUT (+ its log). Deletes nothing.
# Converter = pose5/code-811-s2p/ckpt_local_export.py (= ppm/code's, md5 87c639ee) + --host-as (31101 records LOGICAL hosts: physical
# slinky-15 plays slinky-9, slinky-7 plays slinky-10 = its PPM_HOST_ALIAS), as monoft's ckpt_node_export.py (CKX_HOST_AS) did for dcp-17850.
# VERIFY: (1) the converter's own --finalize (all 128 rank files + rep.pt present, then COMPLETE); (2) the kit preflight on OUT (2,152 keys = the
# FT model + s_emb [2048, 2560] + g0 [5120, 8], all bf16, new tensors finite / non-zero, sidecar coverage). Either failing = exit 1.
set -uo pipefail
K=/data/home/amartyavarma/ppm/pose6/kit; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; C=/data/home/amartyavarma/ppm/pose6/code
STEP=${FT_CONVERT_STEP:-19382}
SRC=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$STEP; OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$STEP
ALIAS=slinky-7:slinky-10,slinky-15:slinky-9          # = prod_811_fresh_append5_s2p.env PPM_HOST_ALIAS (31101)
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
echo "== [$(pt)] convert dcp-$STEP -> $OUT on $(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)"
[ -e "$OUT/COMPLETE" ] && { echo "ABORT: $OUT already COMPLETE (never overwritten)"; exit 1; }
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ $((DF - 89)) -ge 300 ] || { echo "ABORT /data LEDGER: $DF GB free - 89 GB = $((DF - 89)) < 300 GB floor (main prunes 31101 w-N exports first)"; exit 1; }
echo "   /data ledger: $DF GB free -> ~$((DF - 89)) GB after (floor 300)"
srun -N$SLURM_NNODES --ntasks-per-node=1 --cpus-per-task=4 --mem=24G --cpu-bind=none nice -n 19 \
  $PY $K/tools/ckpt_local_export_alias.py --src $SRC --out $OUT --host-as $ALIAS || { echo "ABORT: conversion failed"; exit 1; }
srun -N1 -n1 --cpus-per-task=1 --mem=4G --cpu-bind=none $PY $K/tools/ckpt_local_export_alias.py --out $OUT --finalize || { echo "ABORT: verify 1 (--finalize) failed"; exit 1; }
echo "== [$(pt)] verify 1 PASS (--finalize: 128 rank files + rep.pt, COMPLETE written); verify 2: kit preflight"
srun -N1 -n1 --cpus-per-task=4 --mem=32G --cpu-bind=none bash -c "export FT_NODES=8; set -a; source $K/ft_s2pg0.env; set +a; . $K/tools/ft_overrides.sh; \
  export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=; $PY -u $K/tools/preflight.py --base $OUT --code $C --json $K/logs/preflight-local-$STEP.json" \
  || { echo "ABORT: verify 2 (preflight keys / shapes / sidecar) FAILED on $OUT"; exit 1; }
echo "== [$(pt)] local-$STEP READY: $OUT (verify 1 + 2 PASS); /data free now $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB. Next: launch.sh --base newest"

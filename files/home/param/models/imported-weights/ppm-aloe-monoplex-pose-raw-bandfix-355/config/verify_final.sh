#!/bin/bash
# pose6 monoband (10-29): = pose6/mono17850/verify_final.sh for this kit (export-355, 355 steps). Exit 0 = PASS, 1 = FAIL. Read only; deletes nothing.
# (mono17850 header:) verify the FINAL consolidated export.
#   bash /data/home/amartyavarma/ppm/pose6/monoband/verify_final.sh [<export path>]     (default: /data/home/amartyavarma/ppm/pose6/monoband/export-355)
# Checks: (1) the trainer's own log lines in the FT job log (logs/ft-<job>.log, job from LAUNCHED_JOB): '[export] <path>: consolidated bf16 weights',
# '[export-local] .../export-355 on every node', 'done 355 steps', '== exit 0'; (2) kit/tools/verify_export.py (CPU srun, <= 4 CPUs, read only):
# COMPLETE = 355 + .metadata + train_cfg.json; key set / shapes == export-357's (2,150 keys, the FT model's); all bf16; ALL values finite; cfg
# init = local-17850, link ON (decp_bb_off absent), s2p / g0cond absent, save_every_min 100000 + export_steps [] (option (a')).
set -uo pipefail
M=/data/home/amartyavarma/ppm/pose6/monoband; KT=/data/home/amartyavarma/ppm/pose6/kit; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
X=$(readlink -f "${1:-$M/export-355}"); F=0
echo "== verify_final $X $(TZ=America/Los_Angeles date '+%m-%d %H:%M PT')"
J=$(tail -1 $M/LAUNCHED_JOB 2>/dev/null | awk '{print $1}'); L=$M/logs/ft-$J.log
if [ -n "$J" ] && [ -f "$L" ]; then
  for pat in "\[export\] $X: consolidated bf16 weights" "\[export-local\] .*/export-355 on every node" "done 355 steps" "^== exit 0"; do
    if grep -aqE "$pat" $L; then echo "  ok: log: $(grep -aE "$pat" $L | tail -1 | cut -c1-200)"; else echo "  FAIL: log $L lacks /$pat/"; F=1; fi
  done
else echo "  FAIL: no FT job log (LAUNCHED_JOB '$J', $L)"; F=1; fi
ARGS="$X --step 355 --mods none --batch-gb 4 --expect-init /data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850 --expect-cfg decp_bb_off=<absent> s2p=<absent> g0cond=<absent> save_every_min=100000.0 export_steps=[]"
if [ -n "${SLURM_JOB_ID:-}" ]; then
  OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= $PY -u $KT/tools/verify_export.py $ARGS || F=1
else                                                     # never on the login node: a <= 4-CPU srun on a node with room (never slinky-9)
  N=""
  for n in slinky-4 slinky-10 slinky-18; do
    c=$(scontrol show node $n | tr ' ' '\n' | awk -F= '/^CPUTot=/{t=$2} /^CPUAlloc=/{a=$2} END{print t-a}')
    m=$(scontrol show node $n | tr ' ' '\n' | awk -F= '/^RealMemory=/{r=$2} /^MemSpecLimit=/{s=$2} /^AllocMem=/{u=$2} END{print r-s-u}')
    [ "${c:-0}" -ge 2 ] && [ "${m:-0}" -ge 40960 ] && { N=$n; break; }
  done
  [ -n "$N" ] || { echo "  FAIL: no node with 2 free CPUs + 40G for the read-only check"; exit 1; }
  C=$(( $(scontrol show node $N | tr ' ' '\n' | awk -F= '/^CPUTot=/{t=$2} /^CPUAlloc=/{a=$2} END{print t-a}') >= 4 ? 4 : 2 ))
  echo "  (verify_export.py on $N, -c $C --mem=40G)"
  srun -N1 -n1 -w $N -c $C --mem=40G -t 0-00:45:00 -J pose6-verify --cpu-bind=none env OMP_NUM_THREADS=$C CUDA_VISIBLE_DEVICES= \
    $PY -u $KT/tools/verify_export.py $ARGS || F=1
fi
echo "== verify_final $([ $F = 0 ] && echo PASS || echo FAIL)"; exit $F

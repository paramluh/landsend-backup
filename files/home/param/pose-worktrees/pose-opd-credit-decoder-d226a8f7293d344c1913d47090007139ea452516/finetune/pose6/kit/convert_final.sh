#!/bin/bash
# pose6 kit (B3) STEP "convert" helper, PRINT-ONLY: checks that 31101's final save dcp-19382 can be converted to the FT base local-19382 and prints
# the EXACT command for MAIN to run. It never submits, never runs the conversion, never deletes anything.
#   bash /data/home/amartyavarma/ppm/pose6/kit/convert_final.sh [STEP]       (default 19382 = 31101's PPM_MAX_STEPS)
set -uo pipefail
K=/data/home/amartyavarma/ppm/pose6/kit; STEP=${1:-19382}
H16=$(cat /data/home/amartyavarma/ppm/pose5/nodes16_s7_s15.txt)                          # 31101's 16 physical hosts
CTL=/data/home/amartyavarma/ppm/control/ppm-v1; OUT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-$STEP
F=0; fail() { F=$((F + 1)); echo "  FAIL: $1"; }; ok() { echo "  ok: $1"; }
echo "== pose6 convert check $(TZ=America/Los_Angeles date '+%m-%d %H:%M PT'): dcp-$STEP -> $OUT (print-only)"
J=$(cat $CTL/CURRENT_JOB_811 2>/dev/null || echo 31101); L=/data/home/amartyavarma/ppm/logs/ppm-$J.log
if grep -aq "done $STEP steps" $L 2>/dev/null && ! squeue -h -j $J 2>/dev/null | grep -q .; then ok "run $J ended: 'done $STEP steps' ($L)"
else fail "run $J has not finished $STEP steps yet (last: $(grep -aE '\] step [0-9]+/' $L 2>/dev/null | tail -1 | grep -oE 'step [0-9]+/[0-9]+'); queue: $(squeue -h -j $J -o %T 2>/dev/null))"; fi
grep -aq "dcp-$STEP" $L 2>/dev/null && ok "final save dcp-$STEP logged in $L" || fail "no dcp-$STEP save line in $L"
NV=$(ls /data/home/amartyavarma/setup-logs/hfb/verify/ 2>/dev/null | grep -c "__dcp-${STEP}__")
[ "$NV" -ge 16 ] && ok "dcp-$STEP HF verify records: $NV (pose5/kit/end_of_run.sh step 1)" || echo "  WARN: dcp-$STEP HF verify records $NV < 16 (the conversion only reads the save; HF verify gates its removal, not this step)"
[ -e $OUT/COMPLETE ] && fail "$OUT already exists (COMPLETE): nothing to do; use it as --base"
BUSY=""; for n in $(echo $H16 | tr , ' '); do [ "$(scontrol show node $n | tr ' ' '\n' | grep '^CPUAlloc=' | cut -d= -f2)" -ge 60 ] && BUSY="$BUSY $n"; done
[ -z "$BUSY" ] && ok "16 hosts have >= 4 free CPUs" || fail "hosts without 4 free CPUs:$BUSY (MoT / FT must not be running there)"
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ $((DF - 89)) -ge 300 ] && ok "/data ledger: $DF GB free - 89 GB (local-$STEP) = $((DF - 89)) >= 300" \
  || fail "/data LEDGER: $DF GB free - 89 GB = $((DF - 89)) < 300 GB floor: main must prune superseded 31101 w-N exports first (KEEP=0 keeps them all)"
[ $((DF - 89 - 89)) -ge 300 ] || echo "  WARN: after local-$STEP, the FT's own final /data export-357 (88.4 GB) would leave $((DF - 178)) < 300 GB: the trainer refuses it (node-local export-357 stays the copy)"
echo "-- command for MAIN (run on the login node after the checks pass; the kit never runs it):"
echo "     sbatch -w $H16 $K/convert_final.sbatch"
echo "   verify: (1) converter --finalize (128 rank files + rep.pt -> COMPLETE); (2) kit preflight on $OUT (keys / shapes / bf16 / finite + sidecar)."
echo "   both run inside that job; log $K/logs/convert-<job>.log. Then: bash $K/launch.sh --base newest   (newest = local-$STEP)"
echo "== $F FAIL"; exit $F

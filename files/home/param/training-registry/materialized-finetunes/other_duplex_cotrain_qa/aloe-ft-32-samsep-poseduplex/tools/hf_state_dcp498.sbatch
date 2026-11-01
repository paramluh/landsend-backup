#!/bin/bash
#SBATCH -J pdx-statebk -p all -N1 -n1 -c64 --mem=200G -t 06:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs/hf_state-%j.log
# 10-25 (ppmguard, main -c9 go): back up pose-duplex job 30074's END full state dcp-498 (node-local, with AdamW; ~74 GB / host) to PRIVATE HF
# InternalCan/ppm-aloe-poseduplex-v5-state/dcp-498-<host>/ with hfb_up.py (= hf_watch.sh's hf_one: repo/farm, upload child, verify; retry with
# back-off). One job per host while the 8 hosts are idle; --owner-job: the uploader exits if this job is cancelled (main hands the hosts on).
# usage: sbatch -w <host> tools/hf_state_dcp498.sbatch   (PRIVATE HF ONLY, NEVER GCS; never slinky-9)
set -uo pipefail
H=$(hostname -s); J=$SLURM_JOB_ID
[ "$H" = slinky-9 ] && { echo "ABORT: never slinky-9"; exit 1; }
D=/scratch/amartyavarma/aloe-ft-32-samsep/saves-poseduplex/ppm-aloe-poseduplex-v5-j30074/dcp-498
R=InternalCan/ppm-aloe-poseduplex-v5-state; T=dcp-498-$H
LD=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/logs; VO=$LD/hf_verify; SPD=$LD/hf_spec; mkdir -p $VO $SPD
FB=/scratch/amartyavarma/hfb; F=$FB/pdx-state-$T; mkdir -p $F $FB/xet-cache-state
PY=/data/home/amartyavarma/venvs/duplex/bin/python; UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=$FB/xet-cache-state HFB_OWNER=$J
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
[ -f $D/COMPLETE ] || { echo "[$(pt)] ABORT $H: $D/COMPLETE missing"; exit 1; }
S=$SPD/spec-$T.json
printf '{"dirs": [["%s", "%s"]], "exclude": []}\n' "$D" "$T" > $S
echo "[$(pt)] $H job $J: $D ($(ls -A $D | wc -l) files, $(du -sbL $D | cut -f1) B) -> private $R/$T/"
UP="$PY $UPY --repo $R --type model --farm $F --spec $S --manifest-name MANIFEST.$T.tsv --tag $T --require $D/COMPLETE --workers 32 --verify-out $VO --owner-job $J"
bo=60; att=0; vfail=0; rc=1
while true; do
  att=$((att+1))
  if ! $UP --verify-only --no-verify; then
    echo "[$(pt)] $H attempt $att: repo create / farm failed; retry ${bo}s"; [ $att -ge 8 ] && { rc=2; break; }
    sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  $PY $UPY --child --repo $R --type model --farm $F --spec $S --workers 32 --owner-job $J; urc=$?
  if [ $urc -ne 0 ]; then
    echo "[$(pt)] $H attempt $att: upload rc $urc; retry ${bo}s"; [ $att -ge 8 ] && { rc=1; break; }
    sleep $bo; bo=$((bo*2>600?600:bo*2)); continue; fi
  $UP --verify-only; vrc=$?
  [ $vrc -eq 0 ] && { rc=0; break; }
  vfail=$((vfail+1)); echo "[$(pt)] $H VERIFY FAIL rc $vrc ($vfail)"; [ $vfail -ge 4 ] && { rc=3; break; }; sleep $bo
done
echo "[$(pt)] $H $T HF upload rc $rc (0 = VERIFY PASS) -> private $R/$T/ job $J"
exit $rc

#!/bin/bash
# upload one node-local full save dir (dcp-249) to a PRIVATE HF repo path and verify (hfb_up.py streaming upload; no deletion). usage: bash upload_save.sh <dir> <repo> <tag>
d=$1; R=$2; T=$3; H=$(hostname -s); PY=/data/home/amartyavarma/venvs/duplex/bin/python; UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py
VO=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify; SP=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_spec; F=/scratch/amartyavarma/hfb/save-$T-$H
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=/scratch/amartyavarma/hfb/xet-cache
mkdir -p $F; S=$SP/spec-save-$T.json; printf '{"dirs": [["%s", "%s"]], "exclude": []}\n' "$d" "$T" > $S
UP="$PY $UPY --repo $R --type model --farm $F --spec $S --manifest-name MANIFEST.$T.tsv --tag $T --require $d --workers 4 --verify-out $VO"
for att in 1 2 3; do
  nice -n 19 ionice -c3 $UP --verify-only --no-verify || { sleep 60; continue; }
  nice -n 19 ionice -c3 $PY $UPY --child --repo $R --type model --farm $F --spec $S --workers 4 || { sleep 60; continue; }
  nice -n 19 ionice -c3 $UP --verify-only && { echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] $d -> private $R/$T/ VERIFY PASS" | tee -a /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/BACKUP_LOG.txt; rm -rf $F; exit 0; }
done
echo "$d upload/verify FAILED"; exit 1

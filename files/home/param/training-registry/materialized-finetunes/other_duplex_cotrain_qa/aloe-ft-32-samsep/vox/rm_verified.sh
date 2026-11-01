#!/bin/bash
# delete dirs whose private-HF backup (free_room.sh, repo InternalCan/scratch-backup-1024-aloevox) has a clean VERIFY PASS json:
# PASS, private, n_missing 0, no size/sha mismatch, local_files == files in the dir + 1 (the _backup manifest). usage: bash rm_verified.sh <dir>...
H=$(hostname -s); S=/scratch/amartyavarma; LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log; VO=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify
source /data/home/amartyavarma/ppm/scratch_guard.sh
for d in "$@"; do
  [ -d "$d" ] || { echo "$H: $d absent"; continue; }
  T=$(echo "${d#$S/}" | tr / _)__$H; VJ=$VO/InternalCan__scratch-backup-1024-aloevox__${T}__${H}.json; nf=$(find "$d" -type f | wc -l)
  ok=$(python3 -I -c 'import json,sys; j=json.load(open(sys.argv[1])); n=int(sys.argv[2])
print(1 if (j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch") and int(j.get("local_files",-1)) == n + 1) else 0)' $VJ $nf 2>/dev/null || echo 0)
  if [ "$ok" = 1 ]; then sz=$(du -sh "$d" | cut -f1)
    guard_rm "$d" && echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] ALOEFT32 vox room (main go 10-24 07:35): deleted $H:$d ($sz) - backed up first: private HF InternalCan/scratch-backup-1024-aloevox/$T/ VERIFY PASS ($nf files + manifest, size + sha256; $VJ)" | tee -a $LOG
  else echo "$H: $d verify json not clean ($VJ): KEPT"; fi
done
echo "$H /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')"

#!/bin/bash
# VOX FT end-save room (main go 10-24 ~07:35 PT), ONE host: (1) delete the staged init aloevox-init/w-17127 (loaded at 07:23; on HF) and the node-local
# PPMVOX w-17127 export copy (only if its HF-verify done marker exists); (2) for each listed dir WITHOUT a backup: upload it to PRIVATE HF
# (hfb_up.py: farm + upload_large_folder + verify size + LFS sha256 vs upload-time sha256), and guard_rm it ONLY after a clean VERIFY PASS. Logged.
set -u
H=$(hostname -s); S=/scratch/amartyavarma; LOG=/data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log
PY=/data/home/amartyavarma/venvs/duplex/bin/python; UPY=/data/home/amartyavarma/setup-logs/hfb/hfb_up.py; REPO=InternalCan/scratch-backup-1024-aloevox
VO=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_verify; SP=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/hf_spec; mkdir -p $VO $SP
export PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_XET_CACHE=/dev/shm/amartyavarma-xet-$H
source /data/home/amartyavarma/ppm/scratch_guard.sh
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
fr() { df -BG --output=avail /scratch | tail -1 | tr -dc 0-9; }
rmlog() { local d=$1 why=$2 sz; [ -e "$d" ] || return 0; sz=$(du -sh "$d" | cut -f1)
  if guard_rm "$d"; then echo "[$(pt)] ALOEFT32 vox room (main go 10-24 07:35): deleted $H:$d ($sz) - $why" | tee -a $LOG; else echo "$H: guard_rm REFUSED $d"; fi; }
echo "[$(pt)] $H start, /scratch free $(fr) GB"
[ "${KEEP_INIT:-0}" = 1 ] || rmlog $S/aloevox-init/w-17127 "staged init subset of PPMVOX v3 w-17127 (loaded by job 29345 at 07:23; source private HF InternalCan/ppmvox-v3-16861/w-17127, verified)"
if [ -f /data/home/amartyavarma/ppm/ppmvox/companions/done-v3/w-17127-$H ]; then
  rmlog $S/ppmvox/exports/ppmvox-v3/w-17127 "node-local PPMVOX v3 stop export, HF VERIFY PASS (marker companions/done-v3/w-17127-$H; InternalCan/ppmvox-v3-16861/w-17127/$H/)"
else echo "$H: no done-v3 marker for w-17127: node-local PPMVOX export KEPT"; fi
echo "[$(pt)] $H after step 1: /scratch free $(fr) GB"
for d in "$@"; do
  [ -d "$d" ] || { echo "$H: $d absent"; continue; }
  case "$d" in *q32-s4-val-bs4x-j9453/dcp-6207) OJ=/data/home/amartyavarma/setup-logs/hfb/verify/InternalCan__qwen-s4-val-bs4x-j9453-dcp-6207__dcp-6207__$H.json
    if $PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if (j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch")) else 1)' $OJ 2>/dev/null; then
      rmlog "$d" "existing backup record: private HF VERIFY PASS $OJ"; echo "[$(pt)] $H /scratch free $(fr) GB"; continue; fi;;
  esac
  T=$(echo "${d#$S/}" | tr / _)__$H; SPEC=$SP/spec-bk-$T.json; F=$S/hfb/bk-$T
  printf '{"dirs": [["%s", "%s"]], "exclude": []}\n' "$d" "$T" > $SPEC; mkdir -p $F
  nf=$(find "$d" -type f | wc -l); t0=$(date +%s); echo "[$(pt)] $H backup $d ($nf files, $(du -sh $d | cut -f1)) -> private $REPO/$T/"
  UP="$PY $UPY --repo $REPO --type model --farm $F --spec $SPEC --manifest-name MANIFEST.$T.tsv --tag $T --require $d --workers 16 --verify-out $VO"
  ok=0
  for att in 1 2 3; do
    $UP --verify-only --no-verify > /dev/null 2>&1 || { sleep 30; continue; }
    $PY $UPY --child --repo $REPO --type model --farm $F --spec $SPEC --workers 16 > /dev/null 2>&1 || { sleep 30; continue; }
    $UP --verify-only > /dev/null 2>&1 && { ok=1; break; }
  done
  VJ=$VO/${REPO//\//__}__${T}__${H}.json
  pass=$($PY -I -c 'import json,sys,os; j=json.load(open(sys.argv[1])); n=int(sys.argv[2]); t0=int(sys.argv[3])
ok = os.path.getmtime(sys.argv[1]) >= t0 and j.get("PASS") is True and j.get("private") is True and j.get("n_missing") == 0 and not j.get("size_mismatch") and not j.get("sha_mismatch") and int(j.get("local_files", -1)) == n + 1
print(1 if ok else 0)' $VJ $nf $t0 2>/dev/null || echo 0)
  if [ "$ok" = 1 ] && [ "$pass" = 1 ]; then rmlog "$d" "backed up first: private HF $REPO/$T/ VERIFY PASS ($nf files, size + sha256; $VJ)"
  else echo "[$(pt)] $H: $d backup NOT verified (ok=$ok pass=$pass): KEPT"; fi
  rm -rf $F; echo "[$(pt)] $H /scratch free $(fr) GB"
done
rm -rf /dev/shm/amartyavarma-xet-$H
echo "[$(pt)] $H done: /scratch free $(fr) GB"

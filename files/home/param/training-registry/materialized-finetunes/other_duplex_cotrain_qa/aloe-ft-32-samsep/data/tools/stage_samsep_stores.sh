#!/bin/bash
# aloe-ft-32-samsep: stage the 4 sam_sep stores node-local on THIS host, in the reference's layout (stage_raw.py --stores-only:
# /scratch/amartyavarma/aloe-ft/stores/m1nK) under a NEW root so the reference's staged originals stay untouched:
#   /data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores/m1nK -> /scratch/amartyavarma/aloe-ft-samsep/stores/m1nK
# copy via m1nK.tmp + rename; verified: s3rows.load_node digest == the /data store's == the expected digest, every file's size equal,
# every file byte-identical (cmp). Already staged + verified -> skipped. Deletes only its own .tmp / mismatching m1nK (scratch_guard checked).
# usage (one task on the host): bash stage_samsep_stores.sh
set -euo pipefail
source /data/home/amartyavarma/ppm/scratch_guard.sh
SRC=/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores; DST=/scratch/amartyavarma/aloe-ft-samsep/stores
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; H=$(hostname -s)
DG=(7318ffb3ea75cbcc3ff48b63b6cf8b88 31ecb2273542310096936c0baf49efa8 1759200f7818c6017dfb6a46e37fa5df 611b2e7edb1503cc600a6489e49dcf40)
mkdir -p $DST
dig() { $PY -I -c "import sys; sys.path.insert(0, '/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts'); import s3rows as F; n = F.load_node(sys.argv[1]); print(n['digest'], len(n['length']))" "$1"; }
same() { local a=$1 b=$2 f; for f in $(ls $a); do [ -f "$b/$f" ] && [ "$(stat -c %s $a/$f)" = "$(stat -c %s $b/$f)" ] && cmp -s $a/$f $b/$f || return 1; done; [ "$(ls $a | wc -l)" = "$(ls $b | wc -l)" ]; }
for k in 0 1 2 3; do
  s=$SRC/m1n$k; d=$DST/m1n$k
  read sd sr < <(dig $s); [ "$sd" = "${DG[$k]}" ] || { echo "ABORT $H: /data store m1n$k digest $sd != ${DG[$k]}"; exit 1; }
  if [ -f $d/NODE.json ] && [ "$(dig $d | cut -d' ' -f1)" = "${DG[$k]}" ] && same $s $d; then echo "[stage] $H m1n$k already staged + verified digest $sd rows $sr"; continue; fi
  for x in $d.tmp $d; do [ -e $x ] && { guard_check $x || exit 1; rm -rf --one-file-system $x; }; done
  cp -r $s $d.tmp
  same $s $d.tmp || { echo "ABORT $H: m1n$k copy differs from /data"; exit 1; }
  mv $d.tmp $d
  read dd dr < <(dig $d); [ "$dd" = "${DG[$k]}" ] && [ "$dr" = "$sr" ] || { echo "ABORT $H: staged m1n$k digest $dd rows $dr"; exit 1; }
  echo "[stage] $H m1n$k copied + verified (sizes + cmp all files) digest $dd rows $dr"
done
echo "[stage] $H DONE $DST: $(du -sh $DST | cut -f1); /scratch free $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')"

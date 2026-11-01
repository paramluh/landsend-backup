#!/bin/bash
# speaker FT init staging (10-24, PREP ONLY; never touches the source export / save). Two phases, one task per host:
#   phase xfer  (slinky-1 only, the rank-0 host of 29345): copy rank0.pt / rep.pt / COMPLETE of export-249 into the /data transfer dir
#               (speaker/init-xfer/export-249) + the NEW meta.json (export meta + the exact fp32 vgain theta from dcp-249, tools/init_meta.py)
#               + SHA256SUMS of the slinky-1 sources.
#   phase stage (every host of PPM_HOSTS order slinky-1,2,5,11,3,0,8,12 = 29345's; host i holds ranks 8i..8i+7):
#               /scratch/amartyavarma/aloespk-init/export-249 = HARDLINKS of this host's own rank<8i..8i+7>.pt of the node-local export-249
#               (same inode, no space) + rank0.pt / rep.pt (slinky-1: hardlinks; others: copies from the /data transfer dir) + meta.json (new)
#               + COMPLETE (last). Every staged file sha256'd: hardlink -> same inode as the source; copy -> sha256 == slinky-1's source sha256.
#               Built in export-249.part, renamed when verified; refuses if /scratch would keep < 30 GB.
# usage: srun ... bash stage_init.sh xfer|stage
set -uo pipefail
PH=$1; H=$(hostname -s)
HOSTS=slinky-1,slinky-2,slinky-5,slinky-11,slinky-3,slinky-0,slinky-8,slinky-12
X=/scratch/amartyavarma/aloe-ft-32-samsep/exports-vox17127/export-249
S=/scratch/amartyavarma/aloe-ft-32-samsep/saves/ppm-aloe-samsep-vox17127-j29345/dcp-249
SPK=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/speaker
XF=$SPK/init-xfer/export-249
D=/scratch/amartyavarma/aloespk-init/export-249
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
FLOOR=30
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
cd /tmp
if [ "$PH" = xfer ]; then
  [ "$H" = slinky-1 ] || { echo "ABORT: xfer runs on slinky-1"; exit 1; }
  [ -f $X/COMPLETE ] && [ "$(cat $X/COMPLETE)" = 249 ] || { echo "ABORT: $X not COMPLETE 249"; exit 1; }
  [ -e $XF/SHA256SUMS ] && { echo "$H: xfer already done ($XF)"; exit 0; }
  mkdir -p $XF.part
  $PY -I $SPK/tools/init_meta.py $X $S /data/home/amartyavarma/ppm/aloe-ft-32-samsep/runs/ppm-aloe-samsep-vox17127-j29345/metrics.jsonl $XF.part/meta.json || exit 1
  for f in rank0.pt rep.pt COMPLETE; do cp --preserve=timestamps $X/$f $XF.part/$f || exit 1; done
  ( for f in rank0.pt rep.pt COMPLETE meta.json; do echo "$(sha256sum $X/$f | cut -d' ' -f1)  $f"; done ) > $XF.part/SRC_SHA256_slinky1.txt
  ( for f in rank0.pt rep.pt COMPLETE meta.json; do echo "$(sha256sum $XF.part/$f | cut -d' ' -f1)  $f"; done ) > $XF.part/SHA256SUMS
  for f in rank0.pt rep.pt COMPLETE; do
    a=$(grep "  $f\$" $XF.part/SRC_SHA256_slinky1.txt | cut -d' ' -f1); b=$(grep "  $f\$" $XF.part/SHA256SUMS | cut -d' ' -f1)
    [ "$a" = "$b" ] || { echo "ABORT: $f copy sha256 $b != source $a"; exit 1; }; done
  mv $XF.part $XF
  echo "[$(pt)] $H xfer done: $(ls $XF | tr '\n' ' ')"; cat $XF/SHA256SUMS
  exit 0
fi
[ "$PH" = stage ] || { echo "usage: stage_init.sh xfer|stage"; exit 1; }
I=$(echo $HOSTS | tr , '\n' | grep -nx "$H" | cut -d: -f1); [ -n "$I" ] || { echo "ABORT: $H not a staging host"; exit 1; }
i=$((I - 1))
[ -f $XF/SHA256SUMS ] || { echo "ABORT $H: transfer dir not ready"; exit 1; }
[ -f $D/COMPLETE ] && { echo "$H: $D already COMPLETE"; exit 0; }
[ -f $X/NODE_COMPLETE ] || { echo "ABORT $H: $X has no NODE_COMPLETE"; exit 1; }
free0=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9)
need=0; [ "$H" != slinky-1 ] && need=$(( ( $(stat -c %s $XF/rank0.pt) + $(stat -c %s $XF/rep.pt) ) / 1000000000 + 1 ))
[ $((free0 - need)) -ge $FLOOR ] || { echo "ABORT $H: /scratch free ${free0} GB - ${need} GB < floor $FLOOR"; exit 1; }
mkdir -p $D.part
[ "$(stat -c %d $X)" = "$(stat -c %d $(dirname $D))" ] || { echo "ABORT $H: export and staging dir on different filesystems (no hardlinks)"; exit 1; }
own=""
for r in $(seq $((8 * i)) $((8 * i + 7))); do own="$own rank$r.pt"; done
links="$own"; copies="meta.json"
if [ "$H" = slinky-1 ]; then links="$links rep.pt"; else copies="$copies rank0.pt rep.pt"; fi
for f in $links; do [ -e $D.part/$f ] || ln $X/$f $D.part/$f || { echo "ABORT $H: ln $f"; exit 1; }; done
for f in $copies; do cp --preserve=timestamps $XF/$f $D.part/$f || { echo "ABORT $H: cp $f"; exit 1; }; done
: > $D.part/STAGE_SHA256.txt; bad=0
for f in $links; do
  [ "$(stat -c %i $X/$f)" = "$(stat -c %i $D.part/$f)" ] || { echo "BAD $H $f: not the source inode"; bad=1; }
  echo "$(sha256sum $D.part/$f | cut -d' ' -f1)  $f  hardlink-of $X/$f" >> $D.part/STAGE_SHA256.txt
done
for f in $copies; do
  a=$(sha256sum $D.part/$f | cut -d' ' -f1); b=$(grep "  $f\$" $XF/SHA256SUMS | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "BAD $H $f: sha256 $a != transfer $b"; bad=1; }
  echo "$a  $f  copy-of $XF/$f (slinky-1 source sha256 $(grep "  $f\$" $XF/SRC_SHA256_slinky1.txt | cut -d' ' -f1))" >> $D.part/STAGE_SHA256.txt
done
[ $bad = 0 ] || { echo "ABORT $H: verification failed, $D.part left for inspection"; exit 1; }
n=$(ls $D.part/rank*.pt | wc -l)
cp $XF/COMPLETE $D.part/COMPLETE
mv $D.part $D
echo "[$(pt)] $H (host $i, ranks $((8 * i))-$((8 * i + 7))): staged $D: $n rank files (+rank0 $( [ $i = 0 ] && echo own || echo copy)) rep.pt meta.json COMPLETE, sha256 verified; /scratch free ${free0} -> $(df -BG --output=avail /scratch | tail -1 | tr -d ' ')"

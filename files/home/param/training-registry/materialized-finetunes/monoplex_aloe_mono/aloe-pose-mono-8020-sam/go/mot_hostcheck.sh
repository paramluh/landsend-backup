#!/bin/bash
# monosam go/ (10-26): READ-ONLY per-host probe of the MoT node-local saves + /scratch free. Runs as ONE tiny srun task per host
# (mot_stop.sh / mot_resume.sh: srun --overlap into the running ppm-mot job, or a -J mot-*check allocation with --gres=none -c1 --mem=256M
# once the job is gone). Writes nothing. Usage: mot_hostcheck.sh [STEP]
#   prints: MOTHC <host> free_GiB=<n> newest=<newest COMPLETE dcp step or -1> want=<STEP>:<OK|BAD|none> tmp=<n .tmp dirs> saves=[dcp-N:C<COMPLETE content>:<files>f ...] hfhold=[...]
#   OK = dcp-STEP/COMPLETE holds STEP and the save has its 20 files (8 rank*.pt + 8 digest-R + rep.pt + digest-rep + train_state.json + COMPLETE)
c=/scratch/amartyavarma/ppm-mot/ckpt/ppm-mot
h=$(hostname -s)
f=$(df -BG --output=avail /scratch 2>/dev/null | tail -1 | tr -dc 0-9)
s=""; newest=-1; ntmp=0
for d in "$c"/dcp-*; do
  [ -e "$d" ] || continue
  b=${d##*/}; n=$(ls -A "$d" 2>/dev/null | wc -l)
  case "$b" in *.tmp) ntmp=$((ntmp + 1)); s="$s $b:TMP:${n}f"; continue;; esac
  if [ -f "$d/COMPLETE" ]; then
    v=$(head -c 32 "$d/COMPLETE" | tr -dc 0-9); s="$s $b:C$v:${n}f"; k=${b#dcp-}
    [[ $k =~ ^[0-9]+$ ]] && [ "$k" -gt "$newest" ] && newest=$k
  else s="$s $b:NOCOMPLETE:${n}f"; fi
done
want=${1:-}; st=none
if [ -n "$want" ]; then
  d=$c/dcp-$want; st=BAD
  if [ -f "$d/COMPLETE" ] && [ "$(head -c 32 "$d/COMPLETE" | tr -dc 0-9)" = "$want" ] && [ "$(ls "$d"/rank*.pt 2>/dev/null | wc -l)" -eq 8 ] \
     && [ "$(ls "$d"/digest-*.txt 2>/dev/null | wc -l)" -eq 9 ] && [ -f "$d/rep.pt" ] && [ -f "$d/train_state.json" ] && [ "$(ls -A "$d" | wc -l)" -eq 20 ]; then st=OK; fi
fi
hh=$(ls /scratch/amartyavarma/ppm-mot/hfhold 2>/dev/null | tr '\n' ' ')
echo "MOTHC $h free_GiB=${f:-NA} newest=$newest want=${want:-none}:$st tmp=$ntmp saves=[$s ] hfhold=[$hh]"

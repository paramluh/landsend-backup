#!/bin/bash
# pose5 D: submit every post-merge job (merged code-811-s2p). CPU only, nice, --mem always; never slinky-9 / the 16 production hosts.
#   combined suite (stage_combo.sbatch): cfg | tA..tE | resume ; then cmp (afterok)
#   real width (stage3v.sh, A's shared weight cache): g0 | cb0 | full | g0+cb0 | g0+full, zero-init, vs A's real_orig_off.json
#   A's s2p suite re-run on the merged code (copy of A's harness in tests/s2p_postmerge, own out dir; micros step skipped: A's pickles reused)
# REAL_NODES: comma list of nodes for the 5 real jobs (default slinky-4); X: extra sbatch args.
set -uo pipefail
TD=/data/home/amartyavarma/ppm/pose5/code-811-g0/tests/g0
S=/data/home/amartyavarma/ppm/pose5/code-811-s2p
export G0CODE=$S G0OUT=$S/tests/g0/out
mkdir -p $G0OUT $TD/logs
EXC="-x slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-7,slinky-8,slinky-9,slinky-11,slinky-12,slinky-13,slinky-14,slinky-15,slinky-16,slinky-17,slinky-19"
J=()
for st in cfg:4G tA:4G tB:4G tC:4G tD:4G tE:4G resume:8G; do
  s=${st%%:*}; m=${st##*:}
  j=$(STEPS=$s sbatch --parsable -J pose5-g0c-$s $EXC --mem=$m -t 0-01:30:00 ${CNODE:+-w $CNODE} $TD/stage_combo.sbatch) && J+=($j) && echo "combo $s -> $j"
done
STEPS=cmp sbatch --parsable -J pose5-g0c-cmp $EXC --mem=2G -t 0:20:00 -d afterany:$(IFS=:; echo "${J[*]}") $TD/stage_combo.sbatch | sed 's/^/combo cmp -> /'
# real width: one job per combination (~45 min each, ~50 GB peak)
IFS=, read -ra RN <<< "${REAL_NODES:-slinky-4}"
i=0; R=()
for v in "COPY 1 none real_g0" "COPY 0 cb0 real_cb0" "COPY 0 full real_full" "COPY 1 cb0 real_g0cb0" "COPY 1 full real_g0full"; do
  set -- $v; node=${RN[$((i % ${#RN[@]}))]}; i=$((i + 1))
  j=$(sbatch --parsable -J pose5-g0c-$4 -p all -w $node -N1 -n1 -c 2 --mem=52G -t 0-02:00:00 -o $TD/logs/combo-$4-%j.log \
      --wrap "export G0CODE=$S G0OUT=$G0OUT; bash $TD/stage3v.sh $1 $2 $3 $4 $((29860 + i)) && /data/home/amartyavarma/venvs/qwen-s2/bin/python $TD/compare2.py var $S/tests/s2p/out/real_orig_off.json $G0OUT/$4.json z:$( [ $2 = 1 ] && echo -n g0 )$( [ $2 = 1 ] && [ $3 != none ] && echo -n + )$( [ $3 != none ] && echo -n $3 ) $G0OUT/CMP_$4.json") \
    && R+=($j) && echo "real $4 on $node -> $j"
done

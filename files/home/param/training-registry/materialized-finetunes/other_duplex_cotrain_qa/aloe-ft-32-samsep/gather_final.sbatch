#!/bin/bash
#SBATCH -J aloesamsep-gather -p all --ntasks-per-node=1 -c 4 --mem=16G --gres=none -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs/gather-%j.log
# aloe-ft-32-samsep: the FINAL node-local export (ppm-local-bf16-shards-v1, world 14: rank0-7.pt + rep.pt + meta.json + COMPLETE on slinky-18,
# rank8-13.pt on slinky-4, ~78 GB) gathered on slinky-18 into ONE dir in the format the live demo loads (= aloe-zoom-cotrain/
# gather_export_slinky18.sbatch's): <exports>/export-<last>-full. slinky-18's own files are HARD LINKS (no extra space); slinky-4's 6 files are
# pulled over the LAN (python http.server on slinky-4, nice'd, read only) into export-<last>-full.part, each sha256-checked against the source,
# then COMPLETE + rename. Sources are never modified or deleted; nothing goes to /data except logs.
#   sbatch --dependency=afterok:<train job> -N2 -w slinky-4,slinky-18 gather_final.sbatch
set -u
Q=/data/home/amartyavarma/ppm/aloe-ft-32-samsep
set -a; source ${QQA_ENV:-$Q/run.env}; set +a
LAST=$QQA_STEPS; SRC=$ALOE_LOCAL_EXPORT_ROOT/export-$LAST; DST=$ALOE_LOCAL_EXPORT_ROOT/export-$LAST-full
DSTH=slinky-18; SRCH=slinky-4; PORT=18761; MAN=$Q/logs/gather-$SLURM_JOB_ID; VJ=$Q/logs/GATHER_VERIFY.json
mkdir -p $MAN
export SRC DST PORT MAN
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
echo "== gather $SRC ($SRCH ranks) -> $DSTH:$DST job $SLURM_JOB_ID $(pt)"
srun -N1 -n1 -w $SRCH -c 2 --mem=4G --kill-on-bad-exit=0 bash -c '
  H=$(hostname -s)
  [ -f $SRC/NODE_COMPLETE ] || { echo "$H: NO $SRC/NODE_COMPLETE"; touch $MAN/$H.missing; exit 1; }
  nice -n 10 python3 -m http.server $PORT --bind 0.0.0.0 --directory $SRC > $MAN/http-$H.log 2>&1 &
  F=$(ls $SRC | grep -E "^rank[0-9]+\.pt$")
  (for f in $F; do echo $f; done) | nice -n 10 xargs -P 4 -I{} sha256sum $SRC/{} | sed "s#$SRC/##" > $MAN/sha-$H.tmp && mv $MAN/sha-$H.tmp $MAN/sha-$H.txt
  touch $MAN/$H.done; echo "$H: serving + sha256 of $(echo $F | wc -w) files"
  wait' &
SP=$!
srun -N1 -n1 -w $DSTH -c 4 --mem=16G --kill-on-bad-exit=0 bash -c '
  set -u
  [ -e $DST ] && { echo "ABORT: $DST already exists"; exit 1; }
  [ -f $SRC/COMPLETE ] && [ -f $SRC/meta.json ] && [ -f $SRC/rep.pt ] || { echo "ABORT: $SRC not COMPLETE on $(hostname -s)"; exit 1; }
  PART=$DST.part; rm -rf $PART; mkdir -p $PART
  for f in $(ls $SRC | grep -E "^(rank[0-9]+\.pt|rep\.pt|meta\.json)$"); do ln $SRC/$f $PART/$f; done
  WORLD=$(python3 -I -c "import json,sys; m=json.load(open(sys.argv[1])); assert m[\"format\"]==\"ppm-local-bf16-shards-v1\"; print(m[\"world\"])" $SRC/meta.json)
  for i in $(seq 900); do [ -f $MAN/slinky-4.done ] && break; [ -f $MAN/slinky-4.missing ] && { echo "ABORT: source missing"; exit 1; }; sleep 2; done
  for f in $(cut -d" " -f3 $MAN/sha-slinky-4.txt); do curl -sf --retry 3 -o $PART/$f.tmp http://slinky-4:'"$PORT"'/$f && mv $PART/$f.tmp $PART/$f || echo "PULLFAIL $f"; done
  (cd $PART && sha256sum -c --quiet $MAN/sha-slinky-4.txt) > $MAN/verify.txt 2>&1 || { echo "VERIFY FAIL: $(head -3 $MAN/verify.txt)"; exit 1; }
  N=$(ls $PART | grep -cE "^rank[0-9]+\.pt$")
  [ "$N" = "$WORLD" ] || { echo "VERIFY FAIL: $N rank files != world $WORLD"; exit 1; }
  cp $SRC/COMPLETE $PART/COMPLETE; mv $PART $DST
  echo "VERIFY PASS: $N rank files (slinky-4 ones sha256 == source) + rep.pt + meta.json; COMPLETE $(cat $DST/COMPLETE); $(du -sh $DST | cut -f1) at $DST $(date +%T)"; df -h /scratch | tail -1'
rc=$?
python3 -I -c "import json,sys; json.dump({'job': sys.argv[1], 'dest': sys.argv[2], 'host': 'slinky-18', 'PASS': sys.argv[3] == '0', 'time': sys.argv[4]}, open(sys.argv[5], 'w'), indent=1)" \
  $SLURM_JOB_ID $DST $rc "$(pt)" $VJ
echo "== gather rc $rc $(pt)"
kill $SP 2>/dev/null; sleep 1; kill $SP 2>/dev/null
exit $rc

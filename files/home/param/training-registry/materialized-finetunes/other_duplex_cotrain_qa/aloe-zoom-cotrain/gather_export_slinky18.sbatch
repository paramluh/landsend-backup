#!/bin/bash
#SBATCH -J cotrain-gather -p all --ntasks-per-node=1 -c 4 --mem=16G -t 01:00:00
#SBATCH -o /data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/gather-%j.log
# 10-21 PT: gather the co-train's FINAL node-local export (export-253, ppm-local-bf16-shards-v1, world 128: 8 rank<R>.pt per host on the 16 training
# hosts + rep.pt / meta.json / COMPLETE on rank 0's host, ~78 GB) onto slinky-18's /scratch (same path) for the live demo. LAN only (no download,
# nothing written to /data except logs + manifests): every source host serves its export dir read-only over HTTP (python3 http.server, nice'd)
# and computes sha256 of its files; slinky-18 pulls every file in parallel into export-253.part, checks each file's sha256 against the source's,
# then (PASS only) writes COMPLETE and renames export-253.part -> export-253 (the demo keys on COMPLETE). Sources are never modified or deleted.
#   sbatch --dependency=afterok:26467 -N17 -w <nodes16>,slinky-18 gather_export_slinky18.sbatch
set -u
SRC=/scratch/amartyavarma/aloe-zoom-cotrain/exports/export-253
DSTH=slinky-18
DST=/scratch/amartyavarma/aloe-zoom-cotrain/exports/export-253
PORT=18753
MAN=/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/gather-$SLURM_JOB_ID
VJ=/data/home/amartyavarma/ppm/aloe-zoom-cotrain/logs/GATHER_VERIFY.json
SRCH=$(cat /data/home/amartyavarma/ppm/code/nodes16.txt)
mkdir -p $MAN
export SRC DST PORT MAN
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
echo "== gather $SRC (16 hosts) -> $DSTH:$DST  job $SLURM_JOB_ID $(pt)"

# A. sources: serve + sha256 (manifest written atomically, then a .done sentinel)
srun -N16 -n16 -w $SRCH --ntasks-per-node=1 -c 2 --mem=4G --kill-on-bad-exit=0 bash -c '
  H=$(hostname -s)
  [ -d $SRC ] || { echo "$H: NO $SRC"; touch $MAN/$H.missing; exit 1; }
  nice -n 10 python3 -m http.server $PORT --bind 0.0.0.0 --directory $SRC > $MAN/http-$H.log 2>&1 &
  F=$(ls $SRC | grep -E "^(rank[0-9]+\.pt|rep\.pt|meta\.json|COMPLETE)$")
  echo "$H: serving $(echo $F | wc -w) files $(du -sb $SRC | cut -f1) B"
  (for f in $F; do echo $f; done) | nice -n 10 xargs -P 4 -I{} sha256sum $SRC/{} | sed "s#$SRC/##" > $MAN/sha-$H.tmp && mv $MAN/sha-$H.tmp $MAN/sha-$H.txt
  touch $MAN/$H.done
  wait' &
SP=$!

# B. receiver on slinky-18: pull, verify, publish
srun -N1 -n1 -w $DSTH -c 4 --mem=16G --kill-on-bad-exit=0 bash -c '
  set -u
  PART=$DST.part; mkdir -p $PART
  [ -e $DST ] && { echo "ABORT: $DST already exists on $(hostname -s)"; exit 1; }
  df -h /scratch | tail -1
  HOSTS=$(cat /data/home/amartyavarma/ppm/code/nodes16.txt | tr , " ")
  for h in $HOSTS; do for i in $(seq 90); do curl -sf -m 3 -o /dev/null http://$h:$PORT/ && break; sleep 1; done; done
  R0=""; for h in $HOSTS; do curl -sf -m 10 -o $PART/meta.json http://$h:$PORT/meta.json && { R0=$h; break; }; done
  [ -n "$R0" ] || { echo "ABORT: no host serves meta.json"; exit 1; }
  /usr/bin/python3 -I -c "import json,sys; m=json.load(open(sys.argv[1])); assert m[\"format\"]==\"ppm-local-bf16-shards-v1\"; print(\"meta: step\", m[\"step\"], \"world\", m[\"world\"], \"rank0\", m[\"hosts\"][0])
for r,h in enumerate(m[\"hosts\"]): open(sys.argv[2],\"a\").write(f\"{h} rank{r}.pt\n\")" $PART/meta.json $PART/.list || exit 1
  echo "$R0 rep.pt" >> $PART/.list
  echo "rank-0 host $R0; $(wc -l < $PART/.list) files to pull $(date +%T)"
  t0=$(date +%s)
  xargs -P 32 -L 1 bash -c "curl -sf --retry 3 -o $PART/\$1.tmp http://\$0:$PORT/\$1 && mv $PART/\$1.tmp $PART/\$1 || echo PULLFAIL \$0 \$1" < $PART/.list
  echo "pulled $(du -sb $PART | cut -f1) B in $(( $(date +%s) - t0 )) s"
  for i in $(seq 900); do n=$(ls $MAN/*.done 2>/dev/null | wc -l); [ $n -ge 16 ] && break; sleep 2; done
  echo "source manifests: $(ls $MAN/sha-*.txt | wc -l) / 16"
  cat $MAN/sha-*.txt | grep -v " COMPLETE$" | (cd $PART && nice xargs -P 16 -L 1 bash -c "echo \"\$0  \$1\" | sha256sum -c --quiet - || echo SHAFAIL \$1") > $PART/.verify 2>&1
  NEXP=$(cat $MAN/sha-*.txt | grep -cv " COMPLETE$"); NHAVE=$(ls $PART | grep -cE "^(rank[0-9]+\.pt|rep\.pt|meta\.json)$")
  if [ -s $PART/.verify ] || [ $NEXP -ne $NHAVE ]; then echo "VERIFY FAIL: expected $NEXP have $NHAVE; $(head -5 $PART/.verify)"; exit 1; fi
  curl -sf -m 10 -o $PART/COMPLETE http://$R0:$PORT/COMPLETE || { echo "ABORT: no COMPLETE at source"; exit 1; }
  rm -f $PART/.list $PART/.verify; mv $PART $DST
  echo "VERIFY PASS: $NHAVE files sha256 == source; COMPLETE $(cat $DST/COMPLETE); $(du -sh $DST | cut -f1) at $DST $(date +%T)"; df -h /scratch | tail -1'
rc=$?
python3 -I -c "import json,sys; json.dump({'job': sys.argv[1], 'dest': sys.argv[2], 'host': 'slinky-18', 'PASS': sys.argv[3] == '0', 'time': sys.argv[4]}, open(sys.argv[5], 'w'), indent=1)" \
  $SLURM_JOB_ID $DST $rc "$(pt)" $VJ
echo "== receiver rc $rc $(pt)"
kill $SP 2>/dev/null; sleep 1; kill $SP 2>/dev/null
exit $rc

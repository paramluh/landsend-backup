#!/bin/bash
# DRY end-to-end test of ../hf_local_exports.sh + ../hf_backup_watch.py on 2 hosts. Uploads nothing (tests/hfb_up_stub.py: no network,
# refuses real repos); never touches the training job, real exports, /scratch/amartyavarma/ppm or any PPM file. Run ON slinky-15 as an
# overlap step of a klweb job there (all other steps: 1-CPU overlap steps of J15 on slinky-15 / J18 on slinky-18, nice / ionice idle):
#   srun --jobid=26424 --overlap -n1 -c1 --mem=0 --cpu-bind=none nice -n 19 bash tests/hf_dry.sh      (J15=26424 J18=26231 defaults)
# Roles: slinky-15 = rank 0's node (rank0-7 + rep.pt / meta.json / COMPLETE), slinky-18 = node 1 (rank8-15); world 16; plan 10,20,30.
# Fake exports: /scratch/amartyavarma/hflx-dry-ttconvo/exports on both hosts (4 KB files); shared logs / state on /data under
# tests/dry-<t>. Both removed at the end (key outputs copied into this log first).
#  a  RUNNING_JOB.txt -> J18 (wrong job name)                    -> REFUSED, nothing launched
#  b  RUNNING_JOB.txt -> J15; export-10 COMPLETE, 10 step lines -> launched; uploads PAUSED (control 0: no step-time BASE yet)
#  c  step lines appended (20.0 s)                              -> BASE, N 0 -> 1 (-> 2); export-10 verified on both hosts; a test guard
#                                                                  list protects export-10 -> KEPT (scratch_guard refused); summary PASS
#  d  export-20 with a .tmp on node 1 and no COMPLETE           -> not launched; completed -> launched; a 45 s step -> N -> 0 + cool-off;
#                                                                  upload fails once per host -> retried; verified; DELETED on both hosts
#  e  job gone (dead id); export-30 (final) + its [export-local] log line -> fallback launch (dry-alloc: node-0 COMPLETE filter, fixed
#                                                                  8 workers, --child path); verified; KEPT on both; watcher exits rc 0
#  f  bad dry roots / modes refused; `sbatch --test-only` of the real 16-node fallback job line (nothing submitted)
set -uo pipefail
B=/data/home/amartyavarma/ppm/ttconvo-ft; T=$B/tests; PY=/data/home/amartyavarma/venvs/duplex/bin/python
J15=${J15:-26424}; J18=${J18:-26231}; H0=slinky-15; H1=slinky-18
ROOT=/scratch/amartyavarma/hflx-dry-ttconvo; SH=$T/dry-$(date +%s); nfail=0
HOSTS16=slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19
chk() { if eval "$2"; then echo "  ok   $1"; else echo "  FAIL $1"; nfail=$((nfail+1)); fi; }
waitfor() {   # waitfor <what> <timeout s> <condition>
  local t=0; while ! eval "$3"; do sleep 1; t=$((t+1)); if [ $t -ge $2 ]; then echo "  FAIL $1 (timeout ${2}s)"; nfail=$((nfail+1)); return 1; fi; done
  echo "  ok   $1 (${t}s)"; }
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*'); do unset $v; done    # nested sruns into other jobs
[ "$(hostname -s)" = $H0 ] || { echo "ABORT: run me on $H0"; exit 1; }
on() { local h=$1 j=$J15; shift; [ $h = $H1 ] && j=$J18
  srun --jobid=$j --overlap -w $h -N1 -n1 -c1 --mem=0 --gres=none --cpu-bind=none -J ttconvo-hfb-drytest nice -n 19 ionice -c3 "$@"; }
echo "== $(TZ=America/Los_Angeles date '+%F %T PT') DRY test on $H0 (job $J15 $(squeue -h -j $J15 -o %j)) + $H1 (job $J18 $(squeue -h -j $J18 -o %j))"
for f in $B/hf_local_exports.sh $B/hf_backup_watch.sbatch $T/hf_dry.sh; do chk "bash -n $(basename $f)" "bash -n $f"; done
chk "python syntax (watcher, stub)" "$PY -I -c 'import ast,sys; [ast.parse(open(f).read()) for f in sys.argv[1:]]' $B/hf_backup_watch.py $T/hfb_up_stub.py"
for h in $H0 $H1; do on $h test ! -e $ROOT || { echo "ABORT: $ROOT exists on $h"; exit 1; }; echo "  $h /scratch: $(on $h df -BG --output=avail /scratch | tail -1) free"; done
mkdir -p $SH/plan $SH/logs $SH/hf_backup
echo "10,20,30" > $SH/plan/EXPORT_STEPS.txt
echo $J18 > $SH/RUNNING_JOB.txt
for i in $(seq 1 10); do echo "[10-21 11:00:$(printf %02d $i) PT] step $i/999 20.0s (pose 1.0 speech 1.0 text 1.0 wait 0.1) fake" >> $SH/train.log; done
{ cat /data/home/amartyavarma/ppm/PROTECTED_SCRATCH.txt; echo; echo "$ROOT/exports/export-10"; } > $SH/guard_list.txt
cat > $SH/mk.sh <<'EOF'
# mk.sh <root> <step> <host index 1|2> <COMPLETE 0|1> [tmp] : fake ppm-local-bf16-shards-v1 export part of one host (world 16)
R=$1; N=$2; I=$3; C=$4; d=$R/exports/export-$N; mkdir -p $d
for r in $(seq $((8*(I-1))) $((8*I-1))); do head -c 4096 /dev/urandom > $d/rank$r.pt; done
if [ $I = 1 ]; then head -c 2048 /dev/urandom > $d/rep.pt; printf '{"format": "ppm-local-bf16-shards-v1", "step": %s, "world": 16}\n' $N > $d/meta.json
  [ $C = 1 ] && echo $N > $d/COMPLETE; fi
[ "${5:-}" = tmp ] && head -c 100 /dev/urandom > $d/rank$((8*(I-1))).pt.tmp
exit 0
EOF
on $H0 bash $SH/mk.sh $ROOT 10 1 1; on $H1 bash $SH/mk.sh $ROOT 10 2 1
on $H0 bash $SH/mk.sh $ROOT 20 1 0; on $H1 bash $SH/mk.sh $ROOT 20 2 0 tmp
WL=$SH/logs/hfb_watch.log; EV=$SH/events.txt; touch $EV
JN=$(squeue -h -j $J15 -o %j)
env HFLX_DRY=1 HFLX_ROOT=$ROOT HFLX_SHARED=$SH HFLX_UP=$T/hfb_up_stub.py HFLX_HOSTS=$H0,$H1 HFLX_JOBMAP=$H0:$J15,$H1:$J18 HFLX_JOBNAME="$JN" \
    HFLX_TRAIN_LOG=$SH/train.log HFLX_GUARD_LIST=$SH/guard_list.txt HFLX_STUB_EVENTS=$EV HFLX_STUB_UPLOAD_S=2 HFLX_STUB_CHILD_FAIL_ONCE=export-20 \
    TTC_POLL_S=2 TTC_CTL_S=1 TTC_RAMP_S=4 TTC_COOL_S=6 TTC_PROBE_IDLE_S=4 TTC_FALLBACK_WAIT_S=3 TTC_GONE_EXIT_S=900 TTC_MAX_LAUNCH=3 \
    timeout 900 $PY -u $B/hf_backup_watch.py > $SH/watch.out 2>&1 &
WPID=$!

echo "== a: wrong job name"
waitfor "a: job $J18 (name != $JN) refused" 40 "grep -q REFUSED $WL 2>/dev/null"
sleep 3
chk "a: nothing launched" "! grep -q 'launch export' $WL"
echo $J15 > $SH/RUNNING_JOB.txt

echo "== b: export-10 COMPLETE, no step-time baseline yet"
waitfor "b: export-10 launched" 40 "grep -q 'launch export-10-' $WL"
waitfor "b: both hosts paused (control 0)" 40 "[ \$(grep -c 'export-10 PAUSED' $EV) = 2 ]"
sleep 3
chk "b: no upload before the baseline" "! grep -q 'export-10 UPLOAD-START' $EV"

echo "== c: step lines -> BASE -> uploads"
( i=10; while [ -d $SH ] && [ ! -f $SH/STOP_APPENDER ] && [ $i -lt 1500 ]; do i=$((i+1)); t=20.0; [ -f $SH/SLOW ] && { t=45.0; rm -f $SH/SLOW; }
    echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] step $i/999 ${t}s (pose 1.0 speech 1.0 text 1.0 wait 0.1) fake" >> $SH/train.log
    sleep 1; done ) &
APID=$!
waitfor "c: export-10 verified on both hosts (summary)" 150 "[ -f $SH/hf_backup/HF_BACKUP-export-10.json ]"
BF=$SH/hf_backup/state/base-$J15
chk "c: BASE 20.0 s written" "grep -qx '20.0' $BF"
chk "c: throttle N 0 -> 1 at BASE" "grep -q 'BASE .* -> N 0 -> 1' $WL"
chk "c: first upload started after BASE" "awk -v b=\$(stat -c %Y $BF) '/UPLOAD-START/ {print (\$1 >= b) ? \"y\" : \"n\"; exit}' $EV | grep -qx y"
chk "c: uploads used the 2 throttle workers" "[ \$(grep 'export-10 UPLOAD-START' $EV | grep -c 'workers 2') = 2 ]"
for h in $H0 $H1; do
  chk "c: export-10 KEPT on $h (guard list)" "on $h test -d $ROOT/exports/export-10"
  chk "c: $h export-10 .local = scratch_guard refused" "grep -q 'scratch_guard refused' $SH/hf_backup/state/export-10/$h.local"
done
chk "c: summary export-10 PASS, intermediate, 2/2 hosts" "$PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j[\"PASS\"] and not j[\"final\"] and j[\"hosts_verified\"] == 2 else 1)' $SH/hf_backup/HF_BACKUP-export-10.json"

echo "== d: export-20 incomplete, then complete; slow step; one upload failure per host"
sleep 5
chk "d: incomplete export-20 not launched" "! grep -q 'launch export-20-' $WL"
on $H1 rm -f $ROOT/exports/export-20/rank8.pt.tmp
on $H0 bash -c "echo 20 > $ROOT/exports/export-20/COMPLETE"
waitfor "d: export-20 launched once COMPLETE" 40 "grep -q 'launch export-20-' $WL"
touch $SH/SLOW
waitfor "d: 45 s step -> N -> 0 + cool-off" 30 "grep -q 'N [0-9]* -> 0, cool-off' $WL"
waitfor "d: export-20 verified on both hosts (summary)" 180 "[ -f $SH/hf_backup/HF_BACKUP-export-20.json ]"
chk "d: throttle ramped N 1 -> 2 (c or d)" "grep -q -- '-> N 1 -> 2' $WL"
for h in $H0 $H1; do
  chk "d: export-20 DELETED on $h" "on $h test ! -e $ROOT/exports/export-20"
  chk "d: $h upload failed once and was retried" "grep -q 'export-20 attempt 1: upload rc 1' $SH/logs/hf_up-export-20-$h.log"
done

echo "== e: job gone, final export-30 -> fallback"
echo 2147483000 > $SH/RUNNING_JOB.txt
waitfor "e: job gone seen" 30 "grep -q 'not in squeue' $WL"
touch $SH/STOP_APPENDER; wait $APID
on $H0 bash $SH/mk.sh $ROOT 30 1 1; on $H1 bash $SH/mk.sh $ROOT 30 2 1
echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] [export-local] $ROOT/exports/export-30 on every node: 0.0 GB bf16 shards (world 16), 1s" >> $SH/train.log
if waitfor "e: watcher exited by itself" 180 "grep -q 'every export handled' $WL"; then wait $WPID; wrc=$?
else touch $SH/hf_backup/state/STOP_WATCH; wait $WPID; wrc=$?; fi
chk "e: watcher rc 0" "[ $wrc = 0 ]"
chk "e: fallback launch (dry-alloc) for export-30" "grep -q 'launch export-30-.*dry-alloc' $WL"
chk "e: fallback used fixed workers (--child, 8)" "grep 'export-30 CHILD-UPLOAD' $EV | grep -q 'workers 8'"
for h in $H0 $H1; do
  chk "e: final export-30 KEPT on $h" "on $h test -d $ROOT/exports/export-30"
  chk "e: $h export-30 .local = final" "grep -q 'final export' $SH/hf_backup/state/export-30/$h.local"
done
chk "e: summary export-30 PASS, final" "$PY -I -c 'import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j[\"PASS\"] and j[\"final\"] and j[\"hosts_verified\"] == 2 else 1)' $SH/hf_backup/HF_BACKUP-export-30.json"
chk "BACKUP_LOG: 6 upload lines, all rc 0" "[ \$(grep -c 'HF upload rc' $SH/logs/BACKUP_LOG.txt) = 6 ] && [ \$(grep -c 'HF upload rc 0 ' $SH/logs/BACKUP_LOG.txt) = 6 ]"
chk "BACKUP_LOG: 3 export summary lines" "[ \$(grep -c 'HF backup VERIFIED on 2/2 hosts' $SH/logs/BACKUP_LOG.txt) = 3 ]"

echo "== f: refusals + real fallback job line"
env HFLX_DRY=1 HFLX_ROOT=/scratch/amartyavarma/ttconvo-ft HFLX_SHARED=$SH HFLX_UP=$T/hfb_up_stub.py HFLX_HOSTS=$H0 HFLX_JOBMAP=$H0:$J15 \
    TTC_LAUNCH=dry TTC_EXPORTS=1 bash $B/hf_local_exports.sh > /dev/null 2>&1; r=$?
chk "f: dry run on the real export root refused" "[ $r = 1 ]"
env HFLX_DRY=1 HFLX_ROOT=$ROOT HFLX_SHARED=$B/logs HFLX_UP=$T/hfb_up_stub.py HFLX_HOSTS=$H0 HFLX_JOBMAP=$H0:$J15 \
    TTC_LAUNCH=dry TTC_EXPORTS=1 bash $B/hf_local_exports.sh > /dev/null 2>&1; r=$?
chk "f: dry run with the real logs dir refused" "[ $r = 1 ]"
env HFLX_DRY=1 HFLX_ROOT=$ROOT HFLX_SHARED=$SH HFLX_UP=$T/hfb_up_stub.py HFLX_HOSTS=$H0 HFLX_JOBMAP=$H0:$J15 \
    TTC_LAUNCH=overlap TTC_EXPORTS=1 bash $B/hf_local_exports.sh > /dev/null 2>&1; r=$?
chk "f: dry run in a real launch mode refused" "[ $r = 1 ]"
env -u HFLX_DRY TTC_LAUNCH=dry TTC_EXPORTS=1 bash $B/hf_local_exports.sh > /dev/null 2>&1; r=$?
chk "f: real run in a dry launch mode refused" "[ $r = 1 ]"
env HFLX_DRY=1 HFLX_SHARED=$B/logs HFLX_ROOT=$ROOT $PY -I $B/hf_backup_watch.py > /dev/null 2>&1; r=$?
chk "f: dry watcher with the real logs dir refused" "[ $r != 0 ]"
out=$(sbatch --test-only -p all --gres=none -N16 -w $HOSTS16 --ntasks-per-node=1 -c 4 --mem=40G -t 0-06:00:00 -J ttconvo-hfb-final -o /dev/null \
      --export=NONE,TTC_LAUNCH=alloc,TTC_EXPORTS=0 $B/hf_local_exports.sh 2>&1); r=$?
echo "  sbatch --test-only: $out"
chk "f: real fallback job line accepted by sbatch --test-only (nothing submitted)" "[ $r = 0 ]"

echo "-- BACKUP_LOG (dry)"; cat $SH/logs/BACKUP_LOG.txt
echo "-- hfb_watch.log (dry)"; cat $WL
echo "-- watcher stdout"; cat $SH/watch.out
echo "-- HF_BACKUP-export-30.json (dry, head)"; head -c 2500 $SH/hf_backup/HF_BACKUP-export-30.json; echo
echo "-- host log export-20 $H1"; cat $SH/logs/hf_up-export-20-$H1.log
echo "-- stub events"; cat $EV
echo "== cleanup"
for h in $H0 $H1; do
  on $h bash -c "unset SCRATCH_GUARD_LIST; source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_rm $ROOT && echo \"  removed $ROOT on \$(hostname -s)\""
done
chk "cleanup: $ROOT gone on both hosts" "on $H0 test ! -e $ROOT && on $H1 test ! -e $ROOT"
case "$SH" in $T/dry-[0-9]*) rm -rf --one-file-system "$SH";; esac
chk "cleanup: $SH removed" "[ ! -e $SH ]"
echo "== DRY TEST $([ $nfail = 0 ] && echo PASS || echo "FAIL ($nfail)") on $H0 + $H1 $(TZ=America/Los_Angeles date '+%F %T PT')"
[ $nfail = 0 ]

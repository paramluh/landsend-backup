#!/bin/bash
# MONOFT init distribution, ONE task per host on all 16 PPM hosts (init_convert.sbatch, or an overlap step for the dry run). Replaces
# ftb_convert.sbatch's /data copy (ckpt-infer/ppm-v1/local-<S>, 83 GB: /data is at its 300 GB floor) with a node-local layout:
#  1. every host converts ITS ranks of the main run's save dcp-<S> (bf16, = ckpt_local_export_slot.py's maths; slinky-7 reads slinky-10's
#     logical slot, the hosts-list check of ftb_convert.sbatch) into a node-local stage dir + a manifest (size / sha256 / rows) on /data;
#  2. serves its stage dir over HTTP on the pod network (no ssh between compute nodes);
#  3. each FT host computes the source files its 8 FT ranks need (init_need.py need: ~16-24 of 128 + rank0 + rep.pt + meta.json), fetches
#     them (own files hard-linked) with sha256 checks, runs the trainer's REAL load_init on CPU for its 8 ranks (init_need.py check, + key set
#     and global shapes == local-17641's, which the 17641 FT loaded), writes COMPLETE, and checks the FT's /scratch need is still there;
#  4. once every FT host is FETCHED (or one FAILED), servers stop and every stage dir is deleted (scratch_guard). dcp-<S> is only READ.
# env: MF_S MF_FT_HOSTS MF_ALL_HOSTS MF_CTL MF_INIT MF_STAGE MF_PORT [MF_REF MF_MBPS MF_DRY MF_FINAL_SAVE MF_TRAINER MF_WORKERS MF_TOOLS]
# exit 0 only if every FT host placed + checked its init (the FT job is chained afterok on the job running this).
set -uo pipefail
: "${MF_S:?}" "${MF_FT_HOSTS:?}" "${MF_ALL_HOSTS:?}" "${MF_CTL:?}" "${MF_INIT:?}" "${MF_STAGE:?}" "${MF_PORT:?}"
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python; T=${MF_TOOLS:-/data/home/amartyavarma/ppm/monoft/tools}   # launch.sh: the run dir's frozen copy
REF=${MF_REF:-/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17641}; MBPS=${MF_MBPS:-0}; DRY=${MF_DRY:-0}; FS=${MF_FINAL_SAVE:-0}
TRN=${MF_TRAINER:-/data/home/amartyavarma/ppm/aloe-v2pure-17641/code/train_aloe_pose.py}
H=$(hostname -s); CTL=$MF_CTL; mkdir -p $CTL/logs
# 10-22 fix: torch otherwise starts one thread per core of the node (64) inside a -c 16 cpuset: 27756 ran convert / load_init 7-10x slower
# than the dry run (13 min instead of ~4)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
pt() { TZ=America/Los_Angeles date '+%m-%d %T PT'; }
say() { echo "[$(pt)] $H $*"; }
free_g() { df -BG --output=avail /scratch | tail -1 | tr -dc 0-9; }
NALL=$(echo $MF_ALL_HOSTS | tr , '\n' | wc -l); NFT=$(echo $MF_FT_HOSTS | tr , '\n' | wc -l)
R0=$(echo $MF_FT_HOSTS | tr , '\n' | sort -V | head -1)
IS_FT=0; echo ",$MF_FT_HOSTS," | grep -q ",$H," && IS_FT=1
case "$MF_INIT$MF_STAGE" in *..*) say "ABORT: bad paths"; exit 1;; esac
case "$MF_INIT" in /scratch/amartyavarma/?*) ;; *) say "ABORT: init $MF_INIT not node-local"; exit 1;; esac
case "$MF_STAGE" in /scratch/amartyavarma/?*) ;; *) say "ABORT: stage $MF_STAGE not node-local"; exit 1;; esac
say "SCRATCHFREE0 $(free_g) (FT host $IS_FT, R0 $R0, dry $DRY, S $MF_S)"
fail() { say "ABORT: $*"; echo "$(pt) $*" > $CTL/ABORT-$H; [ $IS_FT = 1 ] && echo "$(pt) $*" > $CTL/FAILED-$H; }
SPID=""
cleanup() {   # always: stop the server, delete this host's stage dir (our own derived files; dcp-<S> is never touched); dry run: the init too
  [ -n "$SPID" ] && { kill $SPID 2>/dev/null; wait $SPID 2>/dev/null; }
  source /data/home/amartyavarma/ppm/scratch_guard.sh
  if [ -d $MF_STAGE ]; then guard_rm $MF_STAGE && say "stage $MF_STAGE deleted" || say "stage $MF_STAGE KEPT (guard refused)"; fi
  if [ "$DRY" = 1 ] && [ -d $MF_INIT ]; then guard_rm $MF_INIT && say "dry run: init $MF_INIT deleted"; fi
  say "SCRATCHFREE $(free_g)"
}
trap cleanup EXIT
# ---- 1. convert this host's ranks
SRC=/scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-$MF_S
if [ ! -f $SRC/COMPLETE ]; then
  for k in /scratch/amartyavarma/ppm/ckpt-keep*/dcp-$MF_S; do [ -f $k/COMPLETE ] && { SRC=$k; break; }; done
fi
[ -f $SRC/COMPLETE ] || { fail "dcp-$MF_S not COMPLETE here (nor a keep copy)"; exit 1; }
AS=""; $PY -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open(sys.argv[2])).get(\"hosts\", []) else 1)" $H $SRC/train_state.json || { [ $H = slinky-7 ] && AS=slinky-10; }
if [ -f $CTL/manifest-$H.json ] && $PY -I -c 'import json,os,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if all(os.path.getsize(f"{sys.argv[2]}/{f}") == e["size"] for f, e in m["files"].items()) else 1)' $CTL/manifest-$H.json $MF_STAGE 2>/dev/null; then
  say "stage + manifest already present (re-run): conversion skipped"
else
  rm -f $CTL/manifest-$H.json
  CKX_HOST_AS=$AS nice -n 19 ionice -c3 $PY $T/ckpt_node_export.py --src $SRC --out $MF_STAGE --manifest $CTL/manifest-$H.json || { fail "conversion failed"; exit 1; }
fi
# ---- 2. serve the stage dir
nice -n 10 $PY -m http.server $MF_PORT --bind 0.0.0.0 --directory $MF_STAGE > $CTL/logs/serve-$H.log 2>&1 &
SPID=$!
sleep 2; kill -0 $SPID 2>/dev/null || { fail "http server on port $MF_PORT did not start ($(tail -2 $CTL/logs/serve-$H.log))"; exit 1; }
# ---- 3. wait for all 16 manifests (or an abort anywhere)
for i in $(seq 1 360); do
  [ "$(ls $CTL/manifest-*.json 2>/dev/null | wc -l)" -ge $NALL ] && break
  ls $CTL/ABORT-* > /dev/null 2>&1 && break
  sleep 5
done
if [ "$(ls $CTL/manifest-*.json 2>/dev/null | wc -l)" -lt $NALL ]; then
  say "ABORT seen or manifests missing after 30 min: $(ls $CTL/ABORT-* 2>/dev/null | xargs -rn1 basename | paste -sd' ')"
  [ $IS_FT = 1 ] && echo "$(pt) manifests incomplete" > $CTL/FAILED-$H
else
  say "all $NALL manifests present"
fi
rc=0
if [ $IS_FT = 1 ] && [ ! -f $CTL/FAILED-$H ]; then
  ok=1
  $PY $T/init_need.py need --ctl $CTL --ft-hosts $MF_FT_HOSTS --host $H > $CTL/need-$H.json || ok=0
  [ $ok = 1 ] && say "needs $($PY -c 'import json,sys; n=json.load(open(sys.argv[1])); print(len(n), "files", round(sum(e["size"] for e in n)/1e9, 1), "GB:", " ".join(e["file"] for e in n))' $CTL/need-$H.json)"
  [ $ok = 1 ] && { nice -n 10 $PY $T/init_fetch.py --need $CTL/need-$H.json --init $MF_INIT --stage $MF_STAGE --port $MF_PORT --mbps $MBPS --workers ${MF_WORKERS:-4} || ok=0; }
  [ $ok = 1 ] && { nice -n 10 $PY $T/init_need.py check --init $MF_INIT --ctl $CTL --ft-hosts $MF_FT_HOSTS --host $H --step $MF_S $([ -f $REF/COMPLETE ] && echo "--ref $REF") --trainer $TRN --complete || ok=0; }
  if [ $ok = 1 ]; then
    NEED=$((65 + 79 * FS)); [ $H = $R0 ] && NEED=$((NEED + 7)); f=$(free_g)
    # the stage files hard-linked into the init stay; the others (link count 1) are freed at cleanup
    fr=$(find $MF_STAGE -type f -links 1 -printf '%s\n' | awk '{s += $1} END {printf "%d", s / 2^30}')
    say "init $MF_INIT COMPLETE $(cat $MF_INIT/COMPLETE) ($(du -sBG $MF_INIT | cut -f1)); /scratch free ${f} GB + ${fr} GB stage freed at cleanup; the FT needs $NEED GB"
    [ "$DRY" = 1 ] || [ $((f + fr)) -ge $NEED ] || { ok=0; say "/scratch short for the FT: ${f} + ${fr} GB < $NEED GB"; }
  fi
  if [ $ok = 1 ]; then echo "$(pt)" > $CTL/FETCHED-$H; else echo "$(pt) fetch / check failed" > $CTL/FAILED-$H; rc=1; fi
fi
# ---- 4. keep serving until every FT host is FETCHED or FAILED (<= 40 min)
for i in $(seq 1 480); do
  [ $(( $(ls $CTL/FETCHED-* 2>/dev/null | wc -l) + $(ls $CTL/FAILED-* 2>/dev/null | wc -l) )) -ge $NFT ] && break
  sleep 5
done
nf=$(ls $CTL/FETCHED-* 2>/dev/null | wc -l)
say "FT hosts fetched: $nf of $NFT$(ls $CTL/FAILED-* > /dev/null 2>&1 && echo "; FAILED: $(ls $CTL/FAILED-* | xargs -n1 basename | paste -sd' ')")"
[ "$nf" = "$NFT" ] || rc=1
exit $rc

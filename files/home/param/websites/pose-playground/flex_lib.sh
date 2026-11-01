# 10-26 (user: "for vansh's model lets do 20 steps not 50 and also use many many gpus"): shared FLEX logic for playground GPU pairs.
# Sourced by render_vansh_unit.sh (renderer pairs) and unit.sbatch (mono / dup worker pairs). Every renderer is one render_flex.sh in its own
# process group (killable); a role switch only happens BETWEEN jobs (the lane log's last job start has its DONE / ABORTED / FAIL).
#   renderer pair: lanes serve vansh20 > vansh50 > param-pg, the role whose queue has waited longest (Param after >= 10 s unclaimed)
#   worker pair:   the worker runs by default; when a Vansh queue has waited >= 10 s, no run of the worker's model is queued and the worker is
#                  idle, it is stopped and the lanes render Vansh; a queued run of its model (or 60 s with no Vansh work) brings it back
#                  once every lane has finished its current render.
# VANSH_PER=1 (default): one render_service per GPU = two samples in parallel per pair; VANSH_PER=2: one per pair (sampler | resolver ...).
: "${CANOPY_ROOT:?export CANOPY_ROOT=<cluster home that holds ppm/ venvs/ models/; see README>}"
GW=${GENWEB:-$CANOPY_ROOT/ppm/aloe-posedup-live/genweb}; GR=$GW/render; FLOGS=$CANOPY_ROOT/video/dmd-rt/logs
VANSH_PER=${VANSH_PER:-1}
qdir(){ case $1 in vansh20) echo jobs_vansh;; vansh50) echo jobs_vansh50;; param-pg) echo jobs;; esac; }
# age in s of the oldest unclaimed job of a queue dir (0 = none waiting)
qwait(){ local f b now=$(date +%s) old=0 t; for f in $GR/$1/*.feed.jsonl; do [ -e "$f" ] || continue; b=$(basename "$f" .feed.jsonl)
  [ -f $GR/out/$b.mp4 ] || [ -f $GR/out/$b.claim ] && continue; t=$((now - $(stat -c %Y "$f"))); [ $t -gt $old ] && old=$t; done; echo $old; }
lanelog(){ echo $FLOGS/render-flex-pg$SLURM_JOB_ID-g${1//,/_}.log; }
rbusy(){ # a lane is mid-render when its GPU(s) compute: any of 3 samples over ~1 s at >= 10 % (the seq renderer logs no job start; a loaded
  # idle render_service sits at 0 %). SIGTERM mid-job would only hand the job back (render_service releases its claim), but we never do that.
  local g i u; for g in ${1//,/ }; do for i in 1 2 3; do u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" -ge 10 ] && return 0; sleep 0.4; done; done; return 1; }
# the role to serve next: the longest-waiting queue among the allowed roles ("" = nothing waiting); ties -> listed order
pick(){ local best="" bw=0 r w min; for r in "$@"; do w=$(qwait $(qdir $r)); min=${PMIN:-1}; [ $r = param-pg ] && min=10
  [ $w -ge $min ] && [ $w -gt $bw ] && { best=$r; bw=$w; }; done; echo $best; }
lstart(){ # lane gpu(s), role -> pid in LPID[lane]; renderer env isolated from the worker's (env -i), = how videort's scripts are run
  setsid env -i HOME=$HOME USER=$USER LOGNAME=$USER PATH=/usr/local/bin:/usr/bin:/bin LANG=${LANG:-C.UTF-8} SLURM_JOB_ID=$SLURM_JOB_ID \
    bash $GW/render_flex.sh $1 $2 pg$SLURM_JOB_ID & LPID[$1]=$!; LROLE[$1]=$2; echo "$(date -u +%T) lane g$1 -> $2 (pid ${LPID[$1]})"; }
lstop(){ local p=${LPID[$1]}; [ -n "$p" ] && { kill -- -$p 2>/dev/null; kill $p 2>/dev/null; wait $p 2>/dev/null; }; LPID[$1]=""; LROLE[$1]=""; }
lalive(){ [ -n "${LPID[$1]}" ] && kill -0 ${LPID[$1]} 2>/dev/null; }
lanes(){ local GP; IFS=, read -ra GP <<< "$CUDA_VISIBLE_DEVICES"; if [ "$VANSH_PER" = 2 ] && [ ${#GP[@]} -ge 2 ]; then echo "${GP[0]},${GP[1]}"; else echo "${GP[@]}"; fi; }
declare -A LPID LROLE

# renderer pair (10-26 main: Param is the common case): with >= 2 lanes, the FIRST lane stays on param-pg for good (always warm); the other
# lanes idle on param-pg too and go to Vansh only when a Vansh job has waited >= 5 s with no Param job waiting, or has waited >= 60 s (no
# starvation); a lane with nothing to do for 60 s returns to param-pg. Every switch happens between jobs only.
# 10-26 (incident: no Param lane existed for 10 min): WARM PARAM LANES are registered (genweb/flex/warm/<job>-g<gpu>, touched every 5 s while
# a param-pg lane is alive with its engine loaded). A renderer pair converts to generation ONLY while another unit's warm Param lane exists;
# the last warm Param lane never leaves; when Param render jobs wait and no warm lane exists anywhere, bringing one up comes first (a pair
# serving generation drains back, Vansh lanes switch back between jobs).
WARM=$GW/flex/warm; mkdir -p $WARM
lane_ready(){ local L=$(lanelog $1) a b; a=$(grep -n "render_flex g${1//,/_} role" $L 2>/dev/null | tail -1 | cut -d: -f1); b=$(grep -n "engine ready" $L 2>/dev/null | tail -1 | cut -d: -f1)
  [ -n "$b" ] && [ "$b" -gt "${a:-0}" ]; }
warm_mark(){ local l; for l in "$@"; do if [ "${LROLE[$l]}" = param-pg ] && lalive $l && lane_ready $l; then touch $WARM/$SLURM_JOB_ID-g$l; else rm -f $WARM/$SLURM_JOB_ID-g$l; fi; done; }
warm_count(){ if [ "$1" = others ]; then find $WARM -type f -newermt "-20 seconds" ! -name "$SLURM_JOB_ID-*" 2>/dev/null | wc -l
  else find $WARM -type f -newermt "-20 seconds" 2>/dev/null | wc -l; fi; }
param_starved(){ [ "$(warm_count all)" = 0 ] && [ "$(qwait jobs)" -gt 0 ]; }
# the decision of one tick for a non-warm lane (role, busy, own-queue waiting, idle s, param wait, vansh20 wait, vansh50 wait) -> new role or ""
lane_decide(){ local role=$1 idle=$2 pw=$3 v20=$4 v50=$5 starved=$6
  if [ "$starved" = 1 ] && [ "$role" != param-pg ]; then echo param-pg; return; fi
  if [ $pw -gt 0 ] && [ "$role" != param-pg ] && [ $v20 -lt 60 ] && [ $v50 -lt 60 ]; then echo param-pg
  elif { [ $pw = 0 ] && { [ $v20 -ge 5 ] || [ $v50 -ge 5 ]; }; } || [ $v20 -ge 60 ] || [ $v50 -ge 60 ]; then
    [ $v50 -gt $v20 ] && echo vansh50 || echo vansh20
  elif [ $idle -ge 60 ] && [ "$role" != param-pg ]; then echo param-pg; fi; }
render_pair(){ local l r i=0 pw v20 v50 warm="" rmode=render pend=0 midle=0 t0 st; declare -A LIDLE
  local L=($(lanes)); [ ${#L[@]} -ge 2 ] && warm=${L[0]}
  trap 'slot_off; rm -f $WARM/$SLURM_JOB_ID-*' EXIT
  for l in "${L[@]}"; do lstart $l param-pg; LIDLE[$l]=0; done
  while true; do sleep 5
    if [ $rmode = mono ]; then                                                  # serving as an extra monoplex worker
      kill -0 $WPID 2>/dev/null || { echo "$(date -u +%T) mono worker exited; restart"; wstart "${MONO_CMD[@]}"; continue; }
      if param_starved; then echo "$(date -u +%T) FLEX Param jobs wait and no warm Param lane exists: back to render lanes (drain first)"; midle=60
      elif [ "$(runs_queued mono)" = 0 ] && mono_idle; then midle=$((midle + 5)); else midle=0; fi
      [ $midle -ge 60 ] || continue
      echo "$(date -u +%T) FLEX mono worker -> render lanes"; wstop $MWID; slot_off; rmode=render
      for l in "${L[@]}"; do lstart $l param-pg; LIDLE[$l]=0; done; continue
    fi
    warm_mark "${L[@]}"
    if [ $pend = 0 ] && want_mono; then
      if [ "$(warm_count others)" -ge 1 ]; then pend=1; t0=$(date +%s); slot_on; echo "$(date -u +%T) FLEX render lanes -> mono worker (queued $(runs_queued mono); other warm Param lanes $(warm_count others))"
      else [ $((i % 12)) = 0 ] && echo "$(date -u +%T) FLEX mono wanted but this pair holds the last warm Param lane(s): stays"; i=$((i + 1)); fi
    fi
    if [ $pend = 1 ]; then
      if [ "$(runs_queued mono)" = 0 ] || [ "$(warm_count others)" = 0 ]; then pend=0; slot_off; echo "$(date -u +%T) FLEX mono switch cancelled (queue drained or no other warm Param lane)"; continue; fi
      local all=1; for l in "${L[@]}"; do if [ -n "${LPID[$l]}" ] && rbusy $l; then all=0; else lstop $l; rm -f $WARM/$SLURM_JOB_ID-g$l; fi; done   # finish current renders
      [ $all = 1 ] || continue
      touch $MSLOTS/$SLURM_JOB_ID; wstart "${MONO_CMD[@]}"; rmode=mono; pend=0; midle=0
      echo "$(date -u +%T) FLEX mono worker starting ($(( $(date +%s) - t0 )) s after the decision)"; continue
    fi
    pw=$(qwait jobs); v20=$(qwait jobs_vansh); v50=$(qwait jobs_vansh50); param_starved && st=1 || st=0
    for l in "${L[@]}"; do
      lalive $l || { echo "$(date -u +%T) lane g$l ${LROLE[$l]} exited; restart"; r=${LROLE[$l]:-param-pg}; lstart $l $r; LIDLE[$l]=0; continue; }
      [ "$l" = "$warm" ] && continue                                            # the warm Param lane never leaves param-pg
      rbusy $l && { LIDLE[$l]=0; continue; }
      [ "$(qwait $(qdir ${LROLE[$l]}))" -gt 0 ] && [ $st = 0 ] && { LIDLE[$l]=0; continue; }   # its own queue has work: it takes it next
      LIDLE[$l]=$((LIDLE[$l] + 5))
      r=$(lane_decide "${LROLE[$l]}" ${LIDLE[$l]} $pw $v20 $v50 $st)
      [ -n "$r" ] && [ "$r" != "${LROLE[$l]}" ] && { echo "$(date -u +%T) lane g$l ${LROLE[$l]} -> $r (param wait $pw s, vansh20 $v20 s, vansh50 $v50 s, starved $st)"; lstop $l; lstart $l $r; LIDLE[$l]=0; }
    done; done; }

# 10-26 user (fixed layout): a VANSH-ONLY pair (the throttled slinky-4 GPUs 6-7): lanes serve vansh20 / vansh50 only (longest wait first,
# switching between jobs), never Param renders or generation.
vansh_pair(){ local l r; for l in $(lanes); do lstart $l vansh20; done
  while true; do sleep 5
    for l in $(lanes); do
      lalive $l || { echo "$(date -u +%T) lane g$l ${LROLE[$l]} exited; restart"; lstart $l ${LROLE[$l]:-vansh20}; continue; }
      rbusy $l && continue
      [ "$(qwait $(qdir ${LROLE[$l]}))" -gt 0 ] && continue
      r=$(pick vansh20 vansh50); [ -n "$r" ] && [ "$r" != "${LROLE[$l]}" ] && { lstop $l; lstart $l $r; }
    done; done; }

# 10-26 user (fixed layout, fallback): a dedicated PARAM render pair: every lane param-pg for good (restarted if it exits), nothing else.
param_pair(){ local l; for l in $(lanes); do lstart $l param-pg; done
  while true; do sleep 5; for l in $(lanes); do lalive $l || { echo "$(date -u +%T) lane g$l exited; restart"; lstart $l param-pg; }; done; done; }

# worker pair: $1 = model (mono | dup), $2... = the worker command (run in its own process group)
WPID=""
wstart(){ setsid "$@" & WPID=$!; echo "$(date -u +%T) worker up (pid $WPID)"; }
wkill(){ [ -n "$WPID" ] && { kill -- -$WPID 2>/dev/null; kill $WPID 2>/dev/null; wait $WPID 2>/dev/null; }; WPID=""; echo "$(date -u +%T) worker stopped"; }
# 10-26 (a reshuffle killed a worker mid-run): a worker is NEVER stopped with a run in flight. wstop <worker id> DRAINS first: the drain flag
# (genweb/flex/drain/<wid>) makes the worker claim nothing new; then wait until no run claimed by it is queued / running; only then kill.
DRAIN=$GW/flex/drain; mkdir -p $DRAIN
wbusy(){ local c s; for c in $GW/runs/*/CLAIMED.$1; do [ -e "$c" ] || continue
  s=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state'))" "$(dirname "$c")/status.json" 2>/dev/null)
  [ "$s" = running ] || [ "$s" = queued ] && return 0; done; return 1; }
wstop(){ local w=${1:-$WID} t=0; touch $DRAIN/$w; sleep 2
  while wbusy $w; do [ $((t % 60)) = 0 ] && echo "$(date -u +%T) draining $w: waiting for its run to finish"; sleep 3; t=$((t + 3)); done
  wkill; rm -f $DRAIN/$w; }
runs_queued(){ local m=$1 n=0 r q; for r in $GW/runs/*/READY; do [ -e "$r" ] || continue; q=$(dirname "$r")/request.json
  if grep -q '"model": "duplex"' "$q" 2>/dev/null; then [ $m = dup ] && n=$((n + 1)); else [ $m = mono ] && n=$((n + 1)); fi; done; echo $n; }
widle(){ local c s; for c in $GW/runs/*/CLAIMED.$WID; do [ -e "$c" ] || continue
  s=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('state'))" "$(dirname "$c")/status.json" 2>/dev/null)
  [ "$s" = running ] || [ "$s" = queued ] && return 1; done; return 0; }

# 10-26 user ("i hope monoplex doesn't generate sequentially"): GENERATION FIRST. Any flex unit turns into an extra monoplex worker (exactly
# unit.sbatch's mono worker: same env, worker.py, one run per worker process, a whole GPU pair: a hot worker holds ~52 + 41 GB) as soon as >= 2
# monoplex runs are queued and the extra workers already loading do not cover them; it goes back to its own role after 60 s with no
# monoplex run queued and its worker idle. Slots: genweb/flex/mono/<job> (= the worker id) while a unit serves as a mono worker.
MSLOTS=$GW/flex/mono; mkdir -p $MSLOTS
MWID=flexmono$SLURM_JOB_ID
MONO_CMD=(bash -c "unset TRITON_CACHE_AUTOTUNING; source $CANOPY_ROOT/ppm/aloe-posedup-live/env_live.sh; export TRITON_CACHE_DIR=/scratch/$USER/genweb/triton-cache MIMI_DIR=$CANOPY_ROOT/models/mimi FA3_DIR=$CANOPY_ROOT/models/qwen-s2/vllm-flash-attn3; mkdir -p \$TRITON_CACHE_DIR; exec \$PY -u $GW/worker.py $MWID >> $GW/logs/worker-flexmono-$SLURM_JOB_ID.log 2>&1")
mono_loading(){ local n=0 f w; for f in $MSLOTS/*; do [ -e "$f" ] || continue; w=$(cat "$f")
  find $GW/runs -maxdepth 2 -name "CLAIMED.$w" -newer "$f" 2>/dev/null | grep -q . || n=$((n + 1)); done; echo $n; }
want_mono(){ local q=$(runs_queued mono); [ $q -ge 2 ] && [ $((q - $(mono_loading))) -ge 1 ]; }
slot_on(){ echo $MWID > $MSLOTS/$SLURM_JOB_ID; }
slot_off(){ rm -f $MSLOTS/$SLURM_JOB_ID; }
mono_idle(){ WID=$MWID widle; }
# 10-26 (main: the duplex pair became a live call backend): the MONO flex unit also serves offline duplex generation (worker_dup.py, exactly
# unit.sbatch's dup env: shared autotune cache) when a duplex run waits, nothing monoplex is queued, and no duplex worker exists elsewhere.
DWID=dup$SLURM_JOB_ID
DUP_CMD=(bash -c "export TRITON_CACHE_AUTOTUNING=1 TRITON_CACHE_DIR=$GW/triton-shared; exec \$PY -u $GW/worker_dup.py $DWID >> $GW/logs/worker-dup-$SLURM_JOB_ID.log 2>&1")
dup_worker_exists(){ squeue -h -u $USER -t R -n genweb-dup -o %i | grep -q . && return 0
  local j; for j in $(squeue -h -u $USER -t R -n genweb-pool-w -o %i); do head -1 $GW/logs/genweb-pool-w-$j.out 2>/dev/null | grep -q "role dup" && return 0; done; return 1; }
worker_pair(){ local m=$1; shift; local mode=worker idle_s=0 back=0 l r
  trap 'slot_off' EXIT
  wstart "$@"
  while true; do sleep 5
    if [ $mode = dupw ]; then                                                   # a mono pair serving offline duplex generation
      kill -0 $WPID 2>/dev/null || { echo "$(date -u +%T) duplex worker exited; restart"; wstart "${DUP_CMD[@]}"; continue; }
      if [ "$(runs_queued dup)" = 0 ] && ! wbusy $DWID; then idle_s=$((idle_s + 5)); else idle_s=0; fi
      if { [ "$(runs_queued mono)" -gt 0 ] || [ $idle_s -ge 60 ]; } && ! wbusy $DWID; then
        echo "$(date -u +%T) FLEX duplex worker -> mono worker"; wstop $DWID; mode=worker; idle_s=0; wstart "$@"; fi
      continue
    fi
    if [ $mode = mono ]; then                                                   # a duplex pair serving as an extra monoplex worker
      kill -0 $WPID 2>/dev/null || { echo "$(date -u +%T) mono worker exited; restart"; wstart "${MONO_CMD[@]}"; continue; }
      if [ "$(runs_queued mono)" = 0 ] && mono_idle; then idle_s=$((idle_s + 5)); else idle_s=0; fi
      if { [ "$(runs_queued $m)" -gt 0 ] || { [ "${DUP_PREFER_MONO:-0}" != 1 ] && [ $idle_s -ge 60 ]; }; } && mono_idle; then
        echo "$(date -u +%T) FLEX mono worker -> $m worker"; wstop $MWID; slot_off; mode=worker; wstart "$@"; fi
      continue
    fi
    if [ $mode = worker ]; then
      kill -0 $WPID 2>/dev/null || { echo "$(date -u +%T) worker exited; restart"; sleep 5; wstart "$@"; continue; }
      if [ $m = mono ] && [ "$(runs_queued dup)" -ge 1 ] && [ "$(runs_queued mono)" = 0 ] && widle && ! dup_worker_exists \
         && [ -d "${GENWEB_DUP_CK:-/scratch/$USER/aloe-posedup-live/export-498-dcp}" ]; then
        echo "$(date -u +%T) FLEX mono worker -> duplex worker (duplex queued $(runs_queued dup))"; wstop $WID; wstart "${DUP_CMD[@]}"; mode=dupw; idle_s=0; continue
      fi
      if [ "${DUP_PREFER_MONO:-0}" = 1 ] && [ $m = dup ] && [ "$(runs_queued dup)" = 0 ] && widle; then   # 10-26 main: mono before Vansh, kept hot
        slot_on; echo "$(date -u +%T) FLEX dup worker -> hot mono worker (no duplex run queued)"; wstop $WID
        touch $MSLOTS/$SLURM_JOB_ID; wstart "${MONO_CMD[@]}"; mode=mono; idle_s=0; continue
      fi
      if [ "${FLEX_NO_MONO:-0}" != 1 ] && [ $m = dup ] && [ "$(runs_queued dup)" = 0 ] && widle && want_mono; then   # 10-26 user: generation first (no duplex run waits)
        local t0=$(date +%s); slot_on; echo "$(date -u +%T) FLEX dup worker -> mono worker (queued $(runs_queued mono))"; wstop $WID
        touch $MSLOTS/$SLURM_JOB_ID; wstart "${MONO_CMD[@]}"; mode=mono; idle_s=0; echo "$(date -u +%T) FLEX mono worker starting ($(( $(date +%s) - t0 )) s)"; continue
      fi
      # 10-26 main: PARAM RENDERS before Vansh: a Param render job unclaimed >= 10 s and nothing of this worker's model queued -> two param-pg lanes
      if [ "$(qwait jobs)" -ge 10 ] && [ "$(runs_queued $m)" = 0 ] && widle; then
        echo "$(date -u +%T) FLEX $m -> Param render lanes (Param waiting $(qwait jobs) s)"; wstop $WID; mode=render; idle_s=0; back=0
        for l in $(lanes); do lstart $l param-pg; done; continue
      fi
      r=$(PMIN=10 pick vansh20 vansh50); [ -z "$r" ] && continue          # >= 10 s unclaimed: the dedicated renderers did not take it
      [ "$(runs_queued $m)" = 0 ] && widle || continue
      echo "$(date -u +%T) FLEX $m -> Vansh renderers ($r waiting)"; wstop $WID; mode=render; idle_s=0; back=0
      for l in $(lanes); do lstart $l $r; done
    else
      [ "$(runs_queued $m)" -gt 0 ] && back=1
      [ "${FLEX_NO_MONO:-0}" != 1 ] && [ $m = dup ] && want_mono && back=1     # generation first: back, then on to a mono worker
      [ $m = mono ] && [ "$(runs_queued dup)" -ge 1 ] && ! dup_worker_exists && back=1   # offline duplex generation (no duplex worker elsewhere)
      if [ "$(qwait jobs_vansh)" = 0 ] && [ "$(qwait jobs_vansh50)" = 0 ] && [ "$(qwait jobs)" = 0 ]; then idle_s=$((idle_s + 5)); else idle_s=0; fi
      [ $idle_s -ge 60 ] && back=1
      local alldone=1
      for l in $(lanes); do
        if [ $back = 1 ]; then { [ -n "${LPID[$l]}" ] && rbusy $l; } && { alldone=0; continue; }; lstop $l; continue; fi
        lalive $l || { lstart $l ${LROLE[$l]:-vansh20}; continue; }
        rbusy $l && continue
        [ "$(qwait $(qdir ${LROLE[$l]}))" -gt 0 ] && continue
        if [ "$(qwait jobs)" -gt 0 ]; then r=param-pg; else r=$(pick vansh20 vansh50); fi   # Param first, then Vansh
        [ -n "$r" ] && [ "$r" != "${LROLE[$l]}" ] && { lstop $l; lstart $l $r; }
      done
      [ $back = 1 ] && [ $alldone = 1 ] && { echo "$(date -u +%T) FLEX Vansh renderers -> $m worker"; mode=worker; wstart "$@"; }
    fi; done; }

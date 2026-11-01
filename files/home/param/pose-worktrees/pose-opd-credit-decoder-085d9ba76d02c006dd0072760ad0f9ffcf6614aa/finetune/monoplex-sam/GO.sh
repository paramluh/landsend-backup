#!/bin/bash
# monosam GO.sh (10-26): PAUSE MoT (clean save) -> monoplex SAM FT on 12 of MoT's hosts (world 96, identical maths to 24272's world 64)
# -> RESUME MoT from that save on its 16 hosts. STAGED: main runs it ONLY on the user's relayed go ("do not pause without my explicit
# permission"). Run it in the background (it takes ~1.5 h) and read go/GO_PHASES.log; phase lines also go to the cluster ALERTS.log.
#
#   GO.sh --dry-run        every check of every phase that does not stop / launch / submit anything (incl. mot_stop.sh / mot_resume.sh
#                          --dry-run); prints what it WOULD do. Writes only go/GO_DRYRUN.log.
#   GO.sh                  the real thing. Options: --nodes=12|8 (default 12, the user's ask; 8 = 24272's exact layout + code path, ~3 min
#                          slower, 4 hosts stay free)  --hosts=h1,... (default go/HOSTS<nodes>.txt)  --no-idcheck (12 nodes: skip the 1-step
#                          world-96 identity check vs 24272, ~3.5 min, default ON)  --no-wait-mono (stop after phase B)
#
# Phases: 0 pre-flight | A MoT stop (go/mot_stop.sh) + FT job held on the 12 hosts (submitted afterany:<MoT job>, gated on READY) ->
# READY once the stop save is COMPLETE on all 16 | B FT launch checks (world 96, rows digest, cfg diff = paths only, init, step 0, step 1)
# + the HF backup controller (setup-logs/hfb/aloe-bk/posemono-sam) | C FT end: "done 357 steps", exit 0, consolidated export-357 COMPLETE
# on /data, the node-local export-357 copied to Weka (controller) -> MoT resume (go/mot_resume.sh) + its verification | D summary.
# On an FT failure: the MoT resume still runs (the FT can resume later from its own save on the same hosts) unless go/HOLD_MOT exists.
# Never: scancel of a running job, slinky-9, a node without --mem, deleting any checkpoint.
set -uo pipefail
. /data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/go/mot_common.sh       # pt / say / phase / ok / warn / fail / hostscan (+ MoT paths)
Q=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam; ORIG=/data/home/amartyavarma/ppm/aloe-pose-mono-8020
ST=/data/home/amartyavarma/ppm/aloe-pose-ft-sam; BK=/data/home/amartyavarma/setup-logs/hfb/aloe-bk/posemono-sam
ALERTS=/data/home/amartyavarma/setup-logs/watchdog/ALERTS.log; PH=$GO/GO_PHASES.log
INIT=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488
RUN_GIB=${RUN_GIB:-}               # this FT's node-local footprint per host: world 96 2 full saves ~53 GiB + 6 exports ~6.9 GiB + slack = 150;
                                   # world 64 2 x ~73 + 6 x ~10.3 + slack = 215
MOT_MIN_GIB=${MOT_MIN_GIB:-64}     # MoT's resume pre-flight (mot.sbatch ILP_MIN_SCRATCH_GB=64): must still be free on every host at FT end
DRY=0; WAITMONO=1; HOSTS=""; NODES=12; IDC=1
for a in "$@"; do case "$a" in
  --dry-run) DRY=1;; --no-wait-mono) WAITMONO=0;; --hosts=*) HOSTS=${a#*=};; --nodes=*) NODES=${a#*=};; --no-idcheck) IDC=0;;
  *) echo "usage: GO.sh [--dry-run] [--nodes=12|8] [--hosts=h1,...] [--no-idcheck] [--no-wait-mono]"; exit 2;; esac; done
case $NODES in 12) RUN_GIB=${RUN_GIB:-150};; 8) RUN_GIB=${RUN_GIB:-215}; IDC=0;; *) echo "--nodes must be 12 or 8"; exit 2;; esac
[ -n "$HOSTS" ] || HOSTS=$(tr -d ' \n' < $GO/HOSTS$NODES.txt 2>/dev/null)
if [ "$DRY" = 1 ]; then exec > >(tee $GO/GO_DRYRUN.log) 2>&1; else exec 9>"$GO/.go.lock"; flock -n 9 || { echo "REFUSED: GO.sh already running"; exit 1; }; fi
report() {   # one phase line: stdout + GO_PHASES.log + the cluster ALERTS.log (main's waiter); dry-run: stdout only
  echo "[$(pt)] [monosam GO] $*"
  [ "$DRY" = 1 ] && return 0
  echo "[$(pt)] $*" >> $PH; echo "[$(TZ=America/Los_Angeles date '+%m-%d %H:%M') PT] [monosam GO] $*" >> $ALERTS
}
die() { report "STOPPED IN PHASE $1: $2"; exit 1; }
J_MOT=$(tr -dc 0-9 < $M/JOB)
UNSETV=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|ILP|MOT|QQA|ALOE)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')

# =============================================================================== 0 pre-flight (read-only; dry-run stops after it)
phase "0 pre-flight (dry-run=$DRY)"
# --- data: SAM stores built + independently verified, nothing missing
[ -f $ST/stores/ALL_DONE ] && ok "SAM stores ALL_DONE" || fail "SAM stores not built ($ST/stores/ALL_DONE)"
python3 - "$ST" <<'EOF' && ok "verify/p0..p7 PASS, CHECK.json 0 missing / 0 T mismatch" || fail "store verification / CHECK.json not clean"
import json, sys
d = sys.argv[1]
c = json.load(open(f"{d}/CHECK.json")); assert not c["missing"] and not c["T_mismatch"], "check"
for k in range(8):
    v = json.load(open(f"{d}/verify/p{k}.json")); assert not v["fail"] and v["spot_rows"] == 64, k
EOF
# --- plan / schedule / cfg: the pre-launch records (tests/plan_sam_check.py)
python3 - "$Q/tests/plan" <<'EOF' && ok "plan records: SAM == original at world 64 (rows 3681cc26ea97cad7, plan a5d995bcec45dc6f); world 96 ranks 0..63 == world 64, 64..95 dummies; LR / decoder schedules equal" || fail "plan records missing or different"
import json, sys
p = sys.argv[1]; o, s, s96 = (json.load(open(f"{p}/{n}.json")) for n in ("orig_w64", "sam_w64", "sam_w96"))
for k in ("rows_digest", "plan_digest", "items_digest", "rows_list_digest", "eval_micros_digest", "step_micros_digest", "lr_mult_digest", "dec_lr_mult_digest", "steps", "warmup", "export_steps"):
    assert o[k] == s[k], k
assert s["rows_digest"] == "3681cc26ea97cad7" and s["plan_digest"] == "a5d995bcec45dc6f" and s["steps"] == 357
for k in ("rows_digest", "items_digest", "rows_list_digest", "eval_micros_digest", "step_micros_ranks_lt64_digest", "lr_mult_digest", "dec_lr_mult_digest"):
    assert s96[k] == s[k], k
assert s96["step_micros_ranks_ge64_all_dummy"] and s96["micro_layout"] == "row_per_rank"
EOF
# --- env / run script / sbatch: only the agreed differences vs 24272's
ED=$(diff <(grep -vE '^\s*#|^\s*$' $ORIG/aloe_mono8020.env) <(grep -vE '^\s*#|^\s*$' $Q/aloe_mono8020.env) | grep -E '^[<>]' | sed -E 's/^([<>]) ([A-Z0-9_]+)=.*/\1\2/' | sort | uniq | paste -sd' ')
EXP="<ALOE_LOCAL_EXPORT_ROOT <ALOE_SAVE_ROOT <ALOE_STORE_DIR <QQA_CONTROL_DIR <QQA_EXPORT <QQA_LOG_ROOT <QQA_PLAN_DIR <QQA_REF_CFG <QQA_REF_CFG2 <QQA_RUN >ALOE_LOCAL_EXPORT_ROOT >ALOE_SAVE_ROOT >ALOE_STORE_DIR >QQA_CONTROL_DIR >QQA_EXPORT >QQA_LOG_ROOT >QQA_PLAN_DIR >QQA_REF_CFG >QQA_REF_CFG2 >QQA_RUN"
[ "$ED" = "$EXP" ] && ok "env diff vs 24272's = paths / names / ref cfg only (10 keys)" || fail "env diff vs 24272's unexpected: $ED"
grep -q '@' $Q/aloe_mono8020.env && fail "env has @placeholders@" || ok "env has no placeholders"
bash -n $Q/aloe_mono8020_run.sh && bash -n $Q/aloe_mono8020.sbatch && bash -n $BK/controller_posemonosam.sh && ok "run script / sbatch / backup controller: bash syntax OK" || fail "bash syntax"
CDS=$(diff -r -x __pycache__ $ORIG/code $Q/code | sha256sum | cut -c1-16)
[ "$CDS" = "$(cat $GO/CODE_DIFF.sha)" ] && ok "code/ = 24272's + exactly the reviewed env-gated world-96 changes (go/CODE_DIFF.txt, $CDS; defaults = 24272's path)" || fail "code/ diff vs 24272's changed since review ($CDS != $(cat $GO/CODE_DIFF.sha))"
[ -f $INIT/meta.json ] && [ "$(ls $INIT/rank*.pt 2>/dev/null | wc -l)" = 128 ] && ok "init $INIT (meta.json + 128 rank shards)" || fail "init $INIT incomplete"
# --- fresh run dir, no job of this run queued
for f in RUN_TAG.txt READY ABORT RUNNING_JOB.txt IDCHECK IDCHECK_RESULT idcheck; do [ -e $Q/$f ] && fail "$Q/$f exists (fresh run expected)"; done
[ -e $Q/runs ] && fail "$Q/runs exists (fresh run expected)"; ls $Q/control/* >/dev/null 2>&1 && fail "$Q/control not empty"
[ -z "$(squeue -h -u amartyavarma -n posemonosam,posemonosambk-ctl -o %i)" ] && ok "no posemonosam / its backup controller in the queue" || fail "a posemonosam job is already queued"
# --- hosts: 12 distinct MoT hosts, none forbidden
NH=$(tr , '\n' <<<"$HOSTS" | grep -c .); MOTN=$(scontrol show hostnames "$(squeue -h -j $J_MOT -o %N)" 2>/dev/null | paste -sd,)
[ "$NH" = $NODES ] && [ "$(tr , '\n' <<<"$HOSTS" | sort -u | grep -c .)" = $NODES ] && ok "$NODES distinct hosts: $HOSTS (idcheck $IDC)" || fail "need $NODES distinct hosts (got '$HOSTS')"
for h in ${HOSTS//,/ }; do
  case $h in slinky-4|slinky-9|slinky-10|slinky-18) fail "forbidden host $h";; esac
  [[ ",$MOTN," == *",$h,"* ]] || fail "$h is not one of MoT job $J_MOT's hosts ($MOTN)"
done
# --- /scratch: FT footprint + MoT's resume minimum on each of the 12 (MoT's own save dcp-S stays: already counted in today's usage)
SC=$(hostscan "$J_MOT" "$HOSTS")
if [ "$(wc -l <<<"$SC")" = $NODES ]; then
  while read -r _ h f _; do f=${f#free_GiB=}; [ "${f:-0}" -ge $((RUN_GIB + MOT_MIN_GIB)) ] && ok "$h /scratch ${f} GiB free >= ${RUN_GIB} + ${MOT_MIN_GIB}" \
    || fail "$h /scratch ${f} GiB free < ${RUN_GIB} (FT) + ${MOT_MIN_GIB} (MoT resume) GiB: free space first (go/SCRATCH_CANDIDATES.md, main's OK)"; done <<<"$SC"
else fail "host scan answered $(wc -l <<<"$SC")/$NODES"; fi
# --- /data: consolidated export-357 (88.5 GB, trainer floor 300 GB after) + one Weka shard copy in flight (88.4 GB, controller floor 300 GB)
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); [ "$DF" -ge 480 ] && ok "/data ${DF} GB free (>= 300 + 2 x 89)" || fail "/data ${DF} GB free < 480"
# --- HF upload host: a spare node with >= 8 free CPUs and >= 40 GB allocatable memory
UPHOST=""
for h in slinky-10 slinky-18 slinky-4; do
  read -r ca ct am rm_ < <(scontrol show node $h | tr ' ' '\n' | grep -E '^(CPUAlloc|CPUTot|AllocMem|RealMemory)=' | cut -d= -f2 | paste -sd' ' | awk '{print $1, $2, $4, $3}')
  [ $((ct - ca)) -ge 9 ] && [ $(( (rm_ - am) / 1024 )) -ge 220 ] && { UPHOST=$h; break; }   # RealMemory over-states schedulable memory by ~170 GB
done
[ -n "$UPHOST" ] && ok "HF upload host $UPHOST (1 upload at a time, 8 CPUs / 40G)" || warn "no spare node with 9 CPUs + 40G now; the controller's uploads will queue (backup only, the FT does not wait)"
# --- MoT stop / resume scripts (their own dry-runs)
bash $GO/mot_stop.sh --dry-run > $GO/GO_mot_stop_dry.log 2>&1 && ok "mot_stop.sh --dry-run rc 0 (go/GO_mot_stop_dry.log)" || fail "mot_stop.sh --dry-run failed (go/GO_mot_stop_dry.log)"
bash $GO/mot_resume.sh --dry-run > $GO/GO_mot_resume_dry.log 2>&1 && ok "mot_resume.sh --dry-run rc 0 (go/GO_mot_resume_dry.log)" || fail "mot_resume.sh --dry-run failed (go/GO_mot_resume_dry.log)"
SBM="env $UNSETV sbatch --parsable -N $NODES -w $HOSTS --dependency=afterany:$J_MOT $Q/aloe_mono8020.sbatch"
CTL="env $UNSETV JOB=<FT job> HOSTS=$HOSTS UPHOST=${UPHOST:-slinky-10} MAXUP=1 sbatch --parsable -p all -w ${UPHOST:-slinky-10} -N1 -n1 -c1 --mem=2G --gres=none -t 3-00:00:00 -J posemonosambk-ctl -o $BK/logs/controller-%j.out --wrap 'bash $BK/controller_posemonosam.sh'"
if [ ${#FAIL[@]} -gt 0 ]; then
  echo; echo "PRE-FLIGHT FAILED (${#FAIL[@]}):"; printf '   - %s\n' "${FAIL[@]}"
  [ "$DRY" = 1 ] || die 0 "pre-flight failed: ${FAIL[0]}"
fi
if [ "$DRY" = 1 ]; then
  phase "dry-run: what the real run WOULD do"
  would "A1 submit the FT, held on the $NODES hosts behind MoT and gated on READY: $SBM"
  would "A2 bash $GO/mot_stop.sh  (STOP_AFTER_SAVE -> dcp-S COMPLETE on 16/16 -> job gone -> S in go/MOT_STOP_STEP); on rc != 0: touch $Q/ABORT (the held FT exits), report, end"
  would "A3 $([ $IDC = 1 ] && echo "touch $Q/IDCHECK + ")touch $Q/READY -> the FT starts on the $NODES hosts (its own node check: /scratch >= $((1500 / NODES + 50)) GB, GPU clocks, stores)$([ $IDC = 1 ] && echo "; first the 1-step world-96 identity check vs 24272 (~3.5 min; FAIL -> no real run, MoT resumes)")"
  would "B1 start the HF backup controller: $CTL"
  would "B2 check the FT log: world $((8 * NODES)) (lw 8), [data] rows digest 3681cc26ea97cad7, [cfgdiff] vs ref_cfg_24272 = 8 path keys, [init] resharded 128 -> 96, step-0 [pack] LOSS CHECK pass, step 1 line; report step 1 + ETA"
  would "C1 wait for the FT to end: 'done 357 steps', '== exit 0', $Q/export-357/COMPLETE; then the controller's node-local export-357 -> Weka copy (<= 10 min)"
  would "C2 bash $GO/mot_resume.sh  (refuses if any of the 16 hosts has < 64 GiB /scratch free; verifies resume step = S, digests, plan, cfg, eval / loss continuity)"
  would "D  report the FT's final evals + the MoT resume line"
  echo; echo "DRY-RUN: ${#FAIL[@]} FAIL, ${#WARNS[@]} WARN"; [ ${#WARNS[@]} -gt 0 ] && printf '   warn: %s\n' "${WARNS[@]}"
  exit $([ ${#FAIL[@]} -eq 0 ] && echo 0 || echo 1)
fi
report "pre-flight PASS ($NODES hosts $HOSTS, idcheck $IDC; MoT job $J_MOT; HF upload host ${UPHOST:-none yet})"

# =============================================================================== A MoT stop + the FT held on the 12 hosts
phase "A MoT stop"
JF=$($SBM) || die A "FT sbatch failed"; echo $JF > $GO/FT_JOB
report "A1 FT job $JF submitted, held behind MoT $J_MOT on $HOSTS, gated on READY"
bash $GO/mot_stop.sh 2>&1 | tee $GO/GO_mot_stop.log; rc=${PIPESTATUS[0]}
if [ $rc != 0 ]; then
  touch $Q/ABORT
  die A "mot_stop.sh rc $rc (go/GO_mot_stop.log): ABORT written, the held FT job $JF exits at its gate; MoT state per mot_stop.sh exit code (3 = still training; 4/5/6 = motguard)"
fi
S=$(tr -dc 0-9 < $GO/MOT_STOP_STEP)
report "A2 MoT stopped cleanly at step $S (dcp-$S COMPLETE on 16/16 hosts)"
[ $IDC = 1 ] && touch $Q/IDCHECK
touch $Q/READY
report "A3 READY written$([ $IDC = 1 ] && echo " + IDCHECK"): FT job $JF starting on the $NODES hosts"

# =============================================================================== B FT launch checks + backup controller
phase "B FT launch"
L=$Q/logs/posemonosam-$JF.log
ERRP='Traceback|^ABORT|ABORT:|COMPILE FAIL|srun: error|CUDA error|NCCL error|OutOfMemory|AssertionError|loss=nan'
t0=$(date +%s)
until grep -qE '\] step 1/357 ' $L 2>/dev/null; do
  if grep -qE "$ERRP" $L 2>/dev/null || [ -z "$(squeue -h -j $JF -o %i)" ]; then
    report "B FT job $JF failed before step 1: $(grep -m1 -E "$ERRP" $L | cut -c1-200)"; FTFAIL=1; break; fi
  [ $(( $(date +%s) - t0 )) -gt 1800 ] && { report "B no step 1 after 30 min (job $JF $(squeue -h -j $JF -o %T)); see $L"; FTFAIL=1; break; }
  sleep 20
done
if [ -z "${FTFAIL:-}" ]; then
  jc=$(env $UNSETV JOB=$JF HOSTS=$HOSTS UPHOST=${UPHOST:-slinky-10} MAXUP=1 sbatch --parsable -p all -w ${UPHOST:-slinky-10} -N1 -n1 -c1 --mem=2G --gres=none -t 3-00:00:00 \
       -J posemonosambk-ctl -o $BK/logs/controller-%j.out --wrap "bash $BK/controller_posemonosam.sh")
  report "B1 HF backup controller job ${jc:-SUBMIT FAILED} (private InternalCan/ppm-aloe-pose-mono-8020-sam)"
  W=$(grep -m1 -oE 'world [0-9]+ \(lw 8\)' $L); RD=$(grep -m1 -oE 'rows digest [0-9a-f]+' $L)
  CDN=$(grep -m1 -oE "resolved config vs $Q/tools/ref_cfg_24272.json \([0-9]+ keys differ" $L | grep -oE '\([0-9]+' | tr -d '(')
  CDK=$(awk "/resolved config vs .*ref_cfg_24272.json/{f=1;next} /resolved config vs/{f=0} f && /\[cfgdiff\]   /" $L | sed -E 's/.*\[cfgdiff\]   ([^:]+):.*/\1/' | sort | paste -sd' ')
  [ "$CDK" = "control_dir export local_export_root log_root plan_dir run_tag save_root stores" ] && CDOK=ok || CDOK="UNEXPECTED: $CDK"
  LC=$(grep -m1 -E '\[pack\] LOSS CHECK' $L | cut -c1-160); S1=$(grep -m1 -E '\] step 1/357 ' $L | cut -c1-230)
  report "B2 FT $W; $RD (expect 3681cc26ea97cad7); cfgdiff vs 24272 ${CDN:-?} keys: $CDOK; $(grep -m1 -E '\[sam\] ' $L | cut -c1-120); idcheck: $(cat $Q/IDCHECK_RESULT 2>/dev/null || echo off); $LC; $S1"
fi

# =============================================================================== C FT end -> MoT resume
phase "C FT end -> MoT resume"
if [ -z "${FTFAIL:-}" ] && [ "$WAITMONO" = 1 ]; then
  last=0
  while [ -n "$(squeue -h -j $JF -o %i)" ]; do
    sl=$(grep -E '\] step [0-9]+/357 ' $L | tail -1); n=$(sed -nE 's/.*\] step ([0-9]+)\/357 .*/\1/p' <<<"$sl")
    [ -n "$n" ] && [ $((n / 60)) -gt $((last / 60)) ] && { report "FT step $n/357 $(grep -oE 'ETA [0-9.]+ min' <<<"$sl")"; last=$n; }
    ls_=$(stat -c %Y $L 2>/dev/null || echo 0); [ $(( $(date +%s) - ls_ )) -gt 900 ] && [ -z "${STALLREP:-}" ] && { report "FT log silent > 15 min at step ${n:-?} (job $JF): stacks before anything else (memory slinky10-rail7-broken)"; STALLREP=1; }
    sleep 60
  done
  if grep -q "done 357 steps" $L && grep -q "== exit 0" $L && [ -f $Q/export-357/COMPLETE ]; then
    report "C1 FT done: $(grep -E 'EVAL step 357:' $L | cut -c1-200); export-357 COMPLETE on /data ($(du -sb $Q/export-357 | awk '{printf "%.1f GB", $1/1e9}'))"
    t1=$(date +%s); until [ -e $BK/state/export-357.copied ] || [ -e $BK/state/export-357.copyfail ] || [ $(( $(date +%s) - t1 )) -gt 600 ]; do sleep 15; done
    report "C2 node-local export-357 -> Weka: $([ -e $BK/state/export-357.copied ] && echo COPIED || echo 'NOT copied within 10 min (node-local copy kept; HF has export-357-dcp from /data)')"
  else
    report "C1 FT job $JF ended WITHOUT 'done 357 steps' / exit 0 / export-357 COMPLETE: $(tail -2 $L | cut -c1-200); it can resume later from its own save on the same $NODES hosts"; FTFAIL=1
  fi
elif [ "$WAITMONO" = 0 ]; then report "C --no-wait-mono: not waiting for the FT; resume MoT later with go/mot_resume.sh"; exit 0; fi
if [ -n "${FTFAIL:-}" ]; then
  [ -n "$(squeue -h -j $JF -o %i)" ] && { report "C FT job $JF still in the queue after a failure: save-and-exit it (touch $Q/control/STOP_AFTER_SAVE), never scancel; MoT resume waits"; \
    touch $Q/control/STOP_AFTER_SAVE; t3=$(date +%s); while [ -n "$(squeue -h -j $JF -o %i)" ]; do sleep 30; \
    [ $(( $(date +%s) - t3 )) -gt 900 ] && die C "FT job $JF still queued 15 min after STOP_AFTER_SAVE (hung before its step loop?): stacks first; main decides; MoT NOT resumed yet"; done; }
  report "C FT failed: waiting 10 min for main to touch go/HOLD_MOT (keeps the $NODES hosts for an FT fix), else MoT resumes"
  t2=$(date +%s); while [ $(( $(date +%s) - t2 )) -lt 600 ]; do [ -e $GO/HOLD_MOT ] && die C "HOLD_MOT present: MoT NOT resumed (run go/mot_resume.sh when ready)"; sleep 15; done
fi
bash $GO/mot_resume.sh 2>&1 | tee $GO/GO_mot_resume.log; rc=${PIPESTATUS[0]}
[ $rc = 0 ] && report "C3 MoT RESUMED from dcp-$S and verified (go/GO_mot_resume.log)" || die C "mot_resume.sh rc $rc (go/GO_mot_resume.log): MoT NOT verified running; motguard / main to act"

# =============================================================================== D summary
report "D DONE: FT ${FTFAIL:+FAILED }job $JF ($(grep -E 'done 357 steps' $L | cut -c1-120)); MoT resumed at $S"

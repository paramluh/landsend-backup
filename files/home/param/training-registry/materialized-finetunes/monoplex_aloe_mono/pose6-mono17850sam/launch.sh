#!/bin/bash
# pose6 B3 (10-27 main 14:29 PT) ONE-COMMAND launcher for mono17850 = the aloe-pose-mono-8020 recipe (job 24272 -> export-357) EXACTLY with
# init = 8,1,1 dcp-17850 (C1's local-17850; old arch, PPM_S2P / PPM_G0COND OFF). Same machinery as pose6/kit/launch.sh (flags documented there);
# here: --base defaults to local-17850, --saves defaults to hourly (= the recipe), preflight --mods none, no sidecar, fallback host slinky-10. DEFAULT = DRY RUN: prints everything, runs the checks it can, submits NOTHING,
# deletes NOTHING (ever).
#
#   bash /data/home/amartyavarma/ppm/pose6/kit/launch.sh --base <newest | w-N dir | local-N dir> [--hosts h1,..,h8] [--go]
#
#   --base      'newest' = the newest COMPLETE export of run 31101 (s2p cb0 + g0cond recorded) under ckpt-weka/ppm-v1/w-* or
#               ckpt-infer/ppm-v1/local-* (a converted final save); or any such dir. REQUIRED.
#   --hosts     the 8 hosts (default slinky-15,14,5,7,1,2,17,3 by projected /scratch free at 31101's hand-back; spare slinky-8; README "Hosts").
#               Never slinky-9. (--nodelist = alias)
#   --nodes     8 (DEFAULT) = world 64 = mono8020's recipe VERBATIM (row_per_rank); 1 = the FALLBACK one-node kit (world 8: lpt layout,
#               AdamW host offload, no FULL saves; NOT bit-identical; README "Fallback"), default host slinky-15.
#   --go        submit (only if every check PASSES). Without it nothing is submitted.
#   --resume    allow an existing RUN_TAG.txt (resume of a stopped 64-rank run; same base + hosts).
#   --link-on   train the D33 decp_bb link ON (mono8020's knob value) instead of the default OFF (= base 31101's state). USER CALL.
#   --saves final|hourly|none   64-rank FULL resumable saves: final = DEFAULT (main 10-27): no hourly saves, the final FULL save (masters +
#               AdamW); hourly = the recipe (keep 2); none = no FULL save (ALOE_SKIP_FINAL_FULL). Save cadence only, maths unchanged.
#   --disk-rule floor|recipe   /scratch need per host: floor = DEFAULT (main 10-27): measured write + 30 GiB (saves final: 165, rank 0 178);
#               recipe = mono8020's 1500|1000|500 / 8 + 50 (final: 175).
#   --rank0     the host that runs rank 0 (default slinky-15: most /scratch; rank 0 also writes rep.pt / meta.json). Placement only.
#   --final-export-only   no intermediate exports 60..298 (only the final export-357, node-local + consolidated /data). USER CALL: export
#               schedule only, maths unchanged (cfg export_steps -> []). Allowed for both kits (64 ranks: placement (a') = --saves none --final-export-only).
#   --stage     stage a DCP base node-local into mono8020's local-shards format first (tools/stage_base.py, bit-exact) instead of the
#               default direct DCP read (pose6/code load_init_dcp); +~15 GiB/node at 64 ranks.
# Checks: base COMPLETE + format; base keys / shapes (tower_p.s_emb.weight [2048, 2560] cb0, inp.g0.weight [5120, 8], key set = the FT model's,
# bf16, finite) + sidecar coverage (27,517 + 531 items) via tools/preflight.py on a compute node; every host fully idle; /scratch free per host
# (live df in a tiny srun; fails loud with the number needed) + no stale saves; /data headroom for the final export (warning only).
set -uo pipefail
K=/data/home/amartyavarma/ppm/pose6/mono17850sam; KT=/data/home/amartyavarma/ppm/pose6/kit; C=/data/home/amartyavarma/ppm/pose6/code; PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
REF_ENV=/data/home/amartyavarma/ppm/aloe-pose-mono-8020/aloe_mono8020.env
HB=/data/home/amartyavarma/ppm/pose5/kit/state/handback/handback-records-1027-114558.txt   # pose5 hand-back projection (read only)
DEF8=slinky-15,slinky-14,slinky-5,slinky-7,slinky-1,slinky-2,slinky-17,slinky-3                # top 8 projected free at hand-back (README)
DEF1=slinky-10
BASE=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-17850; NODES=8; HOSTS=""; GO=0; RESUME=0; STAGE=0; LINK=off; EXPS=all; SAVES=hourly; RULE=floor; RANK0=slinky-15
DISK_FLOOR_GIB=30          # main 10-27: need = the FT's measured write + this floor (= pose5's floor for 31101); --disk-rule recipe = mono8020's
while [ $# -gt 0 ]; do case "$1" in
  --base) BASE=$2; shift 2;; --nodes) NODES=$2; shift 2;; --hosts|--nodelist) HOSTS=$2; shift 2;; --go) GO=1; shift;; --resume) RESUME=1; shift;;
  --stage) STAGE=1; shift;; --link-on) LINK=on; shift;; --final-export-only) EXPS=final; shift;; --saves) SAVES=$2; shift 2;; --disk-rule) RULE=$2; shift 2;; --rank0) RANK0=$2; shift 2;;
  *) echo "unknown arg $1"; exit 2;; esac; done
[ -n "$BASE" ] || { echo "usage: bash $K/launch.sh [--base <dir>] [--hosts h1,..,h8] [--go]"; exit 2; }
[ "$NODES" = 1 ] || [ "$NODES" = 8 ] || { echo "--nodes must be 8 (default, the recipe) or 1 (fallback)"; exit 2; }
case $SAVES in hourly|final|none) ;; *) echo "--saves hourly|final|none"; exit 2;; esac
case $RULE in floor|recipe) ;; *) echo "--disk-rule floor|recipe"; exit 2;; esac
[ $NODES = 1 ] && SAVES=none            # the fallback has no FULL saves at all (host RAM; Rosie)
[ -z "$HOSTS" ] && HOSTS=$([ $NODES = 8 ] && echo $DEF8 || echo $DEF1)
FAIL=(); PEND=(); WARN=()
fail() { FAIL+=("$1"); echo "  FAIL: $1"; }
pend() { PEND+=("$1"); echo "  NOT RUN: $1"; }
warn() { WARN+=("$1"); echo "  WARN: $1"; }
ok() { echo "  ok: $1"; }
now() { TZ=America/Los_Angeles date '+%m-%d %H:%M PT'; }
WORLD=$((8 * NODES))
echo "== pose6 mono17850 kit (mono8020 recipe, init 8,1,1 dcp-17850, both mods OFF) $(now)  mode: $([ $GO = 1 ] && echo GO || echo 'DRY RUN (nothing submitted)')"
echo "   kit: $([ $NODES = 8 ] && echo '64-rank = the recipe VERBATIM (default)' || echo 'FALLBACK one 8-GPU node (world 8; not bit-identical, no FULL saves)')"
echo "   D33 decp_bb link: $([ $LINK = off ] && echo 'OFF (default = base 31101: frozen, never computed; QQA_DECP_BB_OFF=1)' || echo 'ON (--link-on = mono8020 knob value; stale frozen link revived at step 0)')  << USER CALL (pose6 recommends OFF)"
[ $NODES = 8 ] && echo "   FULL saves: $SAVES ($([ $SAVES = final ] && echo 'DEFAULT, main 10-27: final FULL save only (masters + AdamW), no hourly' || echo '--saves'); recipe = hourly; maths unchanged); /scratch rule: $RULE$([ $RULE = floor ] && echo " (measured write + $DISK_FLOOR_GIB GiB)"); rank 0 on $RANK0"
[ $EXPS = final ] && echo "   EXPORTS: final only (--final-export-only)  << user call: export schedule only, maths unchanged"

# ---------------------------------------------------------------- 1. base
echo "-- 1. base"
newest() {   # newest COMPLETE 31101 export: w-N (DCP) or local-N (converted save) with step > 17850 and s2p cb0 + g0cond recorded
  python3 - <<'EOF'
import glob, json, os
c = []
for d in glob.glob("/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-*") + glob.glob("/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-*"):
    s = os.path.basename(d).split("-", 1)[1]
    if not s.isdigit() or int(s) <= 17850 or not os.path.isfile(f"{d}/COMPLETE"):
        continue
    try:
        j = json.load(open(f"{d}/train_cfg.json")) if os.path.isfile(f"{d}/train_cfg.json") else json.load(open(f"{d}/meta.json")).get("cfg", {})
    except Exception:
        continue
    if j.get("s2p") == "cb0" and int(j.get("g0cond", 0) or 0) == 1:
        c.append((int(s), d))
print(max(c)[1] if c else "")
EOF
}
if [ "$BASE" = newest ]; then   # (s2pg0 kit only)
  BASE=$(newest)
  [ -n "$BASE" ] && ok "--base newest -> $BASE" || fail "--base newest: no COMPLETE 31101 export yet (w-N > 17850 with s2p cb0 + g0cond in train_cfg.json); first is due ~17:40 PT 10-27"
fi
BASE=$(readlink -f "$BASE" 2>/dev/null || echo "$BASE")
if [ -f "$BASE/.metadata" ] && [ -f "$BASE/COMPLETE" ]; then
  FMT=dcp; ok "$BASE = consolidated bf16 DCP, step $(cat $BASE/COMPLETE), side $(cat $BASE/train_cfg.json 2>/dev/null); $([ $STAGE = 1 ] && echo 'STAGED node-local (--stage)' || echo 'read directly by pose6/code load_init_dcp')"
elif [ -f "$BASE/meta.json" ] && [ -f "$BASE/COMPLETE" ]; then
  FMT=local; ok "$BASE = ppm-local-bf16-shards-v1, step $(cat $BASE/COMPLETE) (read in place, as mono8020 read local-11488)"
else
  FMT=none; fail "base '$BASE': not a COMPLETE consolidated DCP (.metadata + COMPLETE) nor a local-shards dir (meta.json + COMPLETE)"
fi
case "$BASE" in /data/home/amartyavarma/ppm/ckpt-weka/ppm-v1/w-*|/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-*) ;; *) warn "base is not under ckpt-weka/ppm-v1 or ckpt-infer/ppm-v1: check it is the intended 31101 base";; esac

# ---------------------------------------------------------------- 2. hosts
echo "-- 2. hosts (world $WORLD): $HOSTS"
nfield() { scontrol show node "$1" 2>/dev/null | tr ' ' '\n' | grep "^$2=" | head -1 | cut -d= -f2-; }
node_idle() {   # fully idle: no CPU / GPU allocated, state IDLE
  local st ca gp; st=$(nfield $1 State); ca=$(nfield $1 CPUAlloc); gp=$(nfield $1 AllocTRES | tr , '\n' | grep -c 'gres/gpu=')
  [ "$ca" = 0 ] && [ "$gp" = 0 ] && echo "$st" | grep -q '^IDLE' && ! echo "$st" | grep -qE 'DRAIN|DOWN|RESERVED|MAINT'
}
free_cpu() { echo $(( $(nfield $1 CPUTot) - $(nfield $1 CPUAlloc) )); }
node_ok_hfb() {  # pose6 10-27: during a 31101 pause pose5's hfb-watch uploads the pause save in small on-host allocations (--comment=hfb-onhost);
  # the FT (8 x 4 CPUs, 500G, 8 GPUs, not exclusive) may share the node iff: no GPU allocated, >= 32 CPUs and >= 512000 MB free, no
  # DRAIN/DOWN, and EVERY job on the node is an hfb-onhost upload.
  local st gp other; st=$(nfield $1 State); gp=$(nfield $1 AllocTRES | tr , '\n' | grep -c 'gres/gpu=')
  other=$(squeue -h -w $1 -o '%k' | grep -vc '^hfb-onhost$')
  [ "$gp" = 0 ] && [ "$other" = 0 ] && [ "$(free_cpu $1)" -ge 32 ] && [ "$(free_mem $1)" -ge 512000 ] && ! echo "$st" | grep -qE 'DRAIN|DOWN|RESERVED|MAINT'
}
free_mem() { local r m a; r=$(nfield $1 RealMemory); m=$(nfield $1 MemSpecLimit); a=$(nfield $1 AllocMem); echo $(( ${r:-0} - ${m:-0} - ${a:-0} )); }
NH=$(echo "$HOSTS" | tr , '\n' | grep -c .)
[ $NODES = 1 ] && RANK0=$HOSTS
echo ",$HOSTS," | grep -q ",$RANK0," || fail "--rank0 $RANK0 is not one of --hosts"
[ "$NH" = "$NODES" ] || fail "--hosts has $NH hosts, the kit needs $NODES"
[ "$(echo "$HOSTS" | tr , '\n' | sort -u | grep -c .)" = "$NH" ] || fail "--hosts has duplicates"
for n in $(echo $HOSTS | tr , ' '); do
  [ $n = slinky-9 ] && { fail "slinky-9 is never used"; continue; }
  [ $n = slinky-4 ] && warn "slinky-4: GPUs 6-7 thermal-throttle (the run script aborts on an active thermal slowdown)"
  if node_idle $n; then ok "$n fully idle"
  elif node_ok_hfb $n; then ok "$n free for the FT (only pose5 hfb-onhost uploads there; free CPU $(free_cpu $n), free mem $(free_mem $n) MB, 0 GPUs allocated)"
  else fail "$n not free: $(nfield $n State) CPUAlloc $(nfield $n CPUAlloc) jobs: $(squeue -h -w $n -o '%i:%j' | head -6 | paste -sd' ')"; fi
done

# ---------------------------------------------------------------- 3. env (= mono8020's + the marked lines)
FTENV() { export FT_NODES=$NODES FT_LINK=$LINK FT_SAVES=$SAVES FT_EXPORTS=$EXPS; }
echo "-- 3. env $K/mono17850.env vs $REF_ENV (resolved, non-comment)"
envof() { ( unset $(env | grep -E '^(QQA_|ALOE_|PPM_|S3_MIN|GEMM|FT_)' | cut -d= -f1); FTENV; set -a; source $1; set +a
            [ "$1" = $K/mono17850.env ] && . $KT/tools/ft_overrides.sh; env ) | grep -E '^(QQA_|ALOE_|PPM_|S3_MIN|GEMM)' | sort; }
diff <(envof $REF_ENV) <(envof $K/mono17850.env) | grep '^[<>]' | sed 's/^/     /'
grep -q '@' $K/mono17850.env && fail "env has @placeholders@"
grep -q 'QQA_DECP_BB_OFF' $C/train_aloe_pose.py && ok "pose6/code reads QQA_DECP_BB_OFF; s2p / G0COND unset = mono8020's trainer (B1-certified)" || fail "pose6/code lacks the link knob"
[ -f $K/CFG_DIFF.txt ] && echo "     resolved-cfg diff vs job 24272: $K/CFG_DIFF.txt"
if [ -f $K/RUN_TAG.txt ] && [ $RESUME = 0 ]; then fail "RUN_TAG.txt exists ($(cat $K/RUN_TAG.txt)): a run was already launched from this kit; --resume to continue it, or move RUN_TAG.txt + that run's saves aside (main's call)"; fi

# ---------------------------------------------------------------- 4. preflight on a compute node (keys / shapes / sidecar)
echo "-- 4. base keys / shapes (tools/preflight.py --mods none: the pre-mods key set exactly; 4 CPUs / 32G, ~1 min)"
PFN=""
for n in $(echo $HOSTS | tr , ' ') slinky-4 slinky-10 slinky-18; do
  [ $n = slinky-9 ] && continue
  [ "$(free_cpu $n 2>/dev/null || echo 0)" -ge 4 ] && [ "$(free_mem $n 2>/dev/null || echo 0)" -ge 32768 ] && { PFN=$n; break; }
done
if [ $FMT = none ]; then pend "preflight (no valid base)"
elif [ -z "$PFN" ]; then pend "preflight: no node with 4 free CPUs + 32G (MemSpecLimit applied) among the hosts / slinky-4 / slinky-10 / slinky-18"
else
  PFL=$K/logs/preflight-launch-$(date +%m%d-%H%M%S).log; echo "     on $PFN (log $PFL)"
  srun -N1 -n1 -w $PFN -c4 --mem=32G -t 0-00:20:00 -J pose6-preflight --cpu-bind=none bash -c \
    "$(declare -f FTENV); NODES=$NODES LINK=$LINK SAVES=$SAVES EXPS=$EXPS; FTENV; set -a; source $K/mono17850.env; set +a; . $KT/tools/ft_overrides.sh; export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=; $PY -u $KT/tools/preflight.py --mods none --skip-sidecar --base $BASE --code $C --json ${PFL%.log}.json" > $PFL 2>&1
  r=$?
  grep -E 'RESULT' $PFL | sed 's/^/     /'
  if [ "$r" = 0 ]; then ok "preflight PASS (${PFL%.log}.json)"; else tail -12 $PFL | sed 's/^/     | /'; fail "preflight FAILED (rc $r; $PFL)"; fi
fi

# ---------------------------------------------------------------- 5. /scratch per host (live) + stale saves; /data headroom
echo "-- 5. disk"
STG=0; [ $FMT = dcp ] && [ $STAGE = 1 ] && STG=$((90 / NODES + 4))
NEXP=6; [ $EXPS = final ] && NEXP=1
# write per host at 64 ranks (README "Disk rule"): FULL save 73 GiB, export slice 10.0 (+ rep.pt 2.2 on rank 0) -> final 135 / rank 0 148,
# hourly 208 / 221, none 62 / 75
if [ $NODES = 8 ]; then
  case $SAVES in hourly) NSAVE=2;; final) NSAVE=1;; none) NSAVE=0;; esac
  WR=$(( (NSAVE * 730 + NEXP * 103 + 9) / 10 )); R0X=$(( (NEXP * 22 + 9) / 10 )); SV=$(( NSAVE * 500 + NEXP * 83 ))   # GiB: 73 / save, 10.3 / export (+2.2 rank 0)
  if [ $RULE = recipe ]; then NEED=$((SV / 8 + 50 + STG)); NEED0=$NEED; else NEED=$((WR + DISK_FLOOR_GIB + STG)); NEED0=$((NEED + R0X)); fi
else NEED=$((STG + NEXP * 83 + 30)); NEED0=$NEED; fi   # df -BG units (GiB)
echo "     NEED per host: $NEED GiB, rank-0 host $RANK0: $NEED0 GiB (df -BG units; $([ $NODES = 8 ] && { [ $RULE = recipe ] && echo "rule recipe = mono8020's $SV/8+50" || echo "rule floor = measured write $WR (+$R0X on rank 0) + $DISK_FLOOR_GIB"; } || echo "fallback: $NEXP export(s) x 83 GiB + 30, no FULL saves"); saves $SAVES; staged base $STG)"
for n in $(echo $HOSTS | tr , ' '); do
  hc=/data/home/amartyavarma/ppm/pose5/kit/state/hold_chain/free-$n.tsv   # pose5 hold chain: /scratch free GB per poll (31101 running, 1 save + held)
  proj=$([ -f $hc ] && tail -1 $hc | awk '{printf "%.1f GiB now (%.1f while the previous save is held)", $2 / 1.073741824, ($2 - 46.09) / 1.073741824}'); NN_=$([ $n = $RANK0 ] && echo $NEED0 || echo $NEED)
  if [ $n != slinky-9 ] && [ "$(free_cpu $n 2>/dev/null || echo 0)" -ge 1 ] && [ "$(free_mem $n 2>/dev/null || echo 0)" -ge 1024 ]; then   # read-only df in a tiny srun
    out=$(srun -N1 -n1 -w $n -c1 --mem=1G -t 0-00:05:00 -J pose6-df --cpu-bind=none bash -c 'fr=$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9); u=$(du -sBG /scratch/amartyavarma/pose6-mono17850sam 2>/dev/null | cut -f1 | tr -dc 0-9); s=$(ls -d /scratch/amartyavarma/pose6-mono17850sam/saves/step-* 2>/dev/null | wc -l); echo "$fr ${u:-0} $s"' 2>/dev/null)
    read fr us ns <<< "$out"
    if [ -z "$fr" ]; then pend "$n /scratch check (srun failed; NEED $NEED GiB)"
    elif [ $((fr + us)) -lt $NN_ ]; then fail "$n /scratch $fr GiB free (+ $us in pose6-ft) < NEEDED $NN_ GiB"
    else ok "$n /scratch $fr GiB free (+ $us in pose6-ft) >= $NN_ GiB"; fi
    [ -n "$ns" ] && [ "$ns" -gt 0 ] && [ $RESUME = 0 ] && fail "$n has $ns stale save(s) in /scratch/amartyavarma/pose6-mono17850sam/saves: ALOE_RESUME=auto would resume from them (main's call)"
  else pend "$n /scratch: node busy, df not run (NEED $NN_ GiB; pose5 hold-chain record: ${proj:-?})"; fi
done
DF=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
[ "$DF" -ge 389 ] && ok "/data $DF GB free >= 88.4 + 300 floor for the final consolidated export-357" \
  || warn "/data $DF GB free < 389: the trainer will REFUSE the final /data export-357 (floor 300 GB); the node-local export-357 stays the copy"

# ---------------------------------------------------------------- 6. plan / ETA / exports
echo "-- 6. plan"
if [ $NODES = 8 ]; then MEM=500G; TL=2-00:00:00; ETA="~1 h 30 min (mono8020 job 24272 measured: 357 steps x 11.6 s = 69 min, 80 min wall incl. evals / exports / saves)"
else MEM=530G; TL=1-06:00:00; ETA="~14 h (12-16 h: 357 x (9.17 lpt rounds x ~13.7 s + ~3 s offloaded AdamW) = 12.8 h + 16 evals x ~3.3 min + exports)"; fi
echo "     world $WORLD on $HOSTS; 357 steps x 64 rows (one epoch, rows digest 3681cc26ea97cad7); warmup 12; LR -> 0 at 357; eval every 25"
echo "     exports: node-local /scratch/amartyavarma/pose6-mono17850sam/exports/export-$([ $EXPS = final ] && echo 357 || echo "{60,119,178,238,298,357}") + consolidated $K/export-357 on /data"
echo "     ETA: $ETA"
SB="sbatch --parsable -N$NODES -w $HOSTS --mem=$MEM -t $TL --export=ALL,FT_BASE=$BASE,FT_STAGE=$STAGE,FT_LINK=$LINK,FT_EXPORTS=$EXPS,FT_SAVES=$SAVES,FT_DISK_RULE=$RULE,FT_RANK0=$([ $NODES = 8 ] && echo $RANK0) $K/mono17850.sbatch"
echo "     command: $SB"

# ---------------------------------------------------------------- 7. before 31101 resumes (print only; the kit never deletes anything)
echo "-- 7. after the FT (placement (a): on 31101's hosts during its STOP_AFTER_SAVE pause), BEFORE 31101 resumes"
echo "     the FT's node-local files on each of $HOSTS (the kit NEVER deletes them), all under /scratch/amartyavarma/pose6-mono17850sam/:"
echo "       exports/export-{60,119,178,238,298,357} (~60 GiB per host; +13 on the rank-0 host)  saves/step-* ($SAVES)  cache/{triton,inductor}"
echo "     31101's next save (46.09 GB per host, + the held pause save until it is HF-verified) needs that space: (1) the final export-357 is also"
echo "     on /data ($K/export-357, if >= 300 GB stays free); (2) HF backup of the node-local exports to the PRIVATE repo InternalCan/ppm-pose6-mono17850sam"
echo "     (setup-logs/hfb/aloe-bk/posemono convention; NEVER GCS); (3) main OKs removal; (4) main resumes 31101 (PPM_RESUME_FROM=auto, same 16 hosts)."

# ---------------------------------------------------------------- verdict
echo "== $(now): ${#FAIL[@]} FAIL, ${#PEND[@]} NOT RUN, ${#WARN[@]} WARN"
if [ $GO = 0 ]; then echo "== DRY RUN: nothing submitted. Add --go to submit (requires 0 FAIL and 0 NOT RUN)."; [ ${#FAIL[@]} = 0 ]; exit $?; fi
[ ${#FAIL[@]} = 0 ] && [ ${#PEND[@]} = 0 ] || { echo "== GO REFUSED: every check must PASS (FAIL ${#FAIL[@]}, NOT RUN ${#PEND[@]})"; exit 1; }
J=$($SB) || { echo "sbatch failed"; exit 1; }
echo "$J $(now) hosts $HOSTS rank0 $RANK0 world $WORLD base $BASE link $LINK saves $SAVES rule $RULE exports $EXPS stage $STAGE" >> $K/LAUNCHED_JOB
echo "== SUBMITTED job $J ($K/logs/ft-$J.log)"

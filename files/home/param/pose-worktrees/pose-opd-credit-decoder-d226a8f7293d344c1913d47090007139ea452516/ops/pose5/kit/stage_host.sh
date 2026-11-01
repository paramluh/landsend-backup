#!/bin/bash
# pose5 SWITCH-TIME staging, ONE host (pose5_stage.sbatch runs it on each of the 16 hosts AFTER MoT printed "GPUS FREE for pose5"; never on
# slinky-9; nice/ionice idle). Every step idempotent; a failed check exits 2 (the job fails -> ppm811, queued afterok, never starts).
#  1 DROP17798=1 (main YES 06:33 PT): node-local /scratch/amartyavarma/ppm/ckpt/ppm-v1/dcp-17798 deleted only after a LIVE HF re-check (HF
#    InternalCan/ppm-v1-c dcp-17798/<hf dir>/: 20 files, every name + size == the local file; fail closed per host); one line per host to
#    pose5/STATUS.md. (dcp-17798 is the save before 17850; HF-verified 16/16 on 10-22/23.)
#  2 CLEANCACHE=1: regenerable caches only where the host is short (c/DISK_NEED.json delete_caches), never MoT's (incl. inductor-/triton-cache-ppm,
#    audit 2 R2-2), du printed, guard_rm, one STATUS.md line each. dcp-17798 still present after step 1 -> ABORT (audit 2 R2-8).
#  3 PIN: node-local CKPT/dcp-17850 (slinky-1/2/7/14) verified (stage_dcp17850.py: size + sha256 == HF) and hard-linked into
#    /scratch/amartyavarma/ppm/ckpt-keep-pose5/dcp-17850 (hfb_hold.sh method, 0 B): keep-1 prune can drop only the CKPT name, never the data.
#  4 RESUME LINK /scratch/amartyavarma/pose5-resume/dcp-17850 -> node-local CKPT copy (s1/2/7/14) | ppmvox/init-17850-slots/dcp-17850 (s15,
#    HF-sourced 10-25, re-verified) | Weka /data/home/amartyavarma/ppm/pose5/dcp17850-weka/dcp-17850/<hf dir> (the 11 D1 hosts; VERIFIED marker
#    + sizes). PPM_RESUME_FROM points at the link on every host (outside CKPT: prune / space_check / hfb never touch it).
#  5 B's hooks: slot-9 plans + npl3 window (npl5/bin/leg1_slot9_hook.sh, all hosts: stage on s15, check elsewhere); npl5 initial window
#    (B_NPL5_PREP, B's command) -- both B's, called unchanged.
#  6 slinky-15: slot-9 PPM pose store (slot9/stage_slot9_extras.sh).
#  7 GATE (fail closed): free >= peak saves 99.1 GB (KEEP=1 + chained hold) + the npl5 roll window still to come (B: ROLL_GB_LEFT) + floor
#    (slinky-13 20, others 30); trainer-side checks (plans incl. npl5, pose/speech store node, txtdup, resume link, no save > 17850 in CKPT).
# Prints "STAGEHOST <host> <slot> OK|ABORT|DRY ...". usage: stage_host.sh [--dry-run]
set -u
DRY=0; [ "${1:-}" = --dry-run ] && DRY=1
K=/data/home/amartyavarma/ppm/pose5/kit; P5=/data/home/amartyavarma/ppm/pose5
PY=/data/home/amartyavarma/venvs/duplex/bin/python
H=$(hostname -s)
[ "$H" = slinky-9 ] && { echo "STAGEHOST $H - ABORT never on slinky-9"; exit 2; }
case $H in slinky-7) S=slinky-10; HFD=slinky-7;; slinky-15) S=slinky-9; HFD=slinky-9;; *) S=$H; HFD=$H;; esac   # = PPM_HOST_ALIAS
[ -n "${STAGE_TEST_SLOT:-}" ] && [ $DRY = 1 ] && { S=$STAGE_TEST_SLOT; HFD=$([ $S = slinky-10 ] && echo slinky-7 || echo $S); }
FLOOR=$([ "$H" = slinky-13 ] && echo 20 || echo 30)
CK=/scratch/amartyavarma/ppm/ckpt/ppm-v1; D=$CK/dcp-17850; RL=/scratch/amartyavarma/pose5-resume/dcp-17850
PIN=/scratch/amartyavarma/ppm/ckpt-keep-pose5; WK=$P5/dcp17850-weka; V5=/scratch/amartyavarma/ppmvox/init-17850-slots/dcp-17850
ST=$P5/STATUS.md
say() { echo "STAGEHOST $H $S $*"; }
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M PT'; }
free_gb() { df -B1 --output=avail /scratch | tail -1 | awk '{printf "%.1f", $1/1e9}'; }
stat_line() { [ $DRY = 1 ] || echo "- $(pt) [pose5 switch, $H] $*" >> $ST; }
echo "  start: free $(free_gb) GB, floor $FLOOR GB, slot $S"

# ---- 1 drop node-local dcp-17798 (live HF re-check, fail closed)
X=$CK/dcp-17798
if [ -d $X ]; then
  CHK=$(nice -n 10 $PY -I - "$X" "$HFD" <<'EOF'
import json, os, sys
from huggingface_hub import HfApi
d, h = sys.argv[1], sys.argv[2]
hf = {e.path.rsplit("/", 1)[1]: int(e.size) for e in HfApi().list_repo_tree("InternalCan/ppm-v1-c", path_in_repo=f"dcp-17798/{h}", repo_type="model")
      if getattr(e, "size", None) is not None}
loc = {f: os.path.getsize(f"{d}/{f}") for f in os.listdir(d) if os.path.isfile(f"{d}/{f}")}
ok = len(hf) == 20 and hf == loc and os.path.isfile(f"{d}/COMPLETE")
print(json.dumps({"ok": ok, "hf_files": len(hf), "local_files": len(loc), "bytes": sum(loc.values()),
                  "diff": sorted(set(hf.items()) ^ set(loc.items()))[:4]}))
EOF
)
  echo "  dcp-17798 live HF re-check (InternalCan/ppm-v1-c dcp-17798/$HFD): $CHK"
  if [ "${DROP17798:-0}" != 1 ]; then echo "  dcp-17798 kept (DROP17798 not set)"
  elif echo "$CHK" | grep -q '"ok": true'; then
    if [ $DRY = 1 ]; then echo "  [dry] would delete $X ($(du -sh $X | cut -f1))"
    else F0=$(free_gb); nice -n 10 ionice -c3 rm -rf --one-file-system -- "${CK:?}/dcp-17798"; F1=$(free_gb)
         [ -e $X ] && { say "ABORT dcp-17798 delete failed"; exit 2; }
         stat_line "deleted node-local $X (HF InternalCan/ppm-v1-c dcp-17798/$HFD re-checked live: 20/20 names+sizes ==; main YES 06:33 PT) free $F0 -> $F1 GB"; fi
  else stat_line "KEPT $X: live HF re-check failed ($CHK)"; echo "  dcp-17798 KEPT (re-check failed, fail closed)"; fi
fi
# audit 2 R2-8: a dcp-17798 still in CKPT here would be deleted by keep-1 prune at the first pose save (not in PROTECT_STEPS) although main's
# OK was conditional on the live HF re-check: ABORT the stage (ppm811 never starts; main decides: re-check, pin, or drop)
if [ -d $X ]; then
  if [ $DRY = 1 ]; then echo "  [dry] $X present now: at the switch it is dropped after a PASSING live HF re-check, else the stage ABORTS"
  else echo "- $(pt) [pose5 ALERT, $H] dcp-17798 still in CKPT after step 1 (live HF re-check failed or DROP17798 unset): stage ABORTED (keep-1 prune would delete it at the first save): main decides" \
         | tee -a /data/home/amartyavarma/ppm/control/overnight_monitor_811f.md >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log
       say "ABORT dcp-17798 kept in CKPT (re-check: ${CHK:-not run}); keep-1 would prune it at the first save"; exit 2; fi
fi

# ---- 2 regenerable caches (main-approved category: triton* / inductor-cache-* except the run's / hfb xet-cache / aloe-pose-build temp pkls;
#      never data, ckpts, anything MoT): the host's list from c/DISK_NEED.json; du always printed; deleted (CLEANCACHE=1) only where the
#      host needs them ("delete_caches": a host under need + its margin). audit 2 R2-2 (user rule 06:16: NOTHING MoT on the 16 hosts is
#      deleted): inductor-cache-ppm / triton-cache-ppm are MoT's too (ppm/mot/code/task_ppm.sh:11-12) -> never deleted, on any host; and any
#      path a MoT script names (grep of ppm/mot/**/*.sh|sbatch|py|env) is skipped as well
DN=$P5/c/DISK_NEED.json
MOTP=$(grep -rhoE '/scratch/amartyavarma/[A-Za-z0-9_./-]+' /data/home/amartyavarma/ppm/mot --include='*.sh' --include='*.sbatch' --include='*.py' --include='*.env' 2>/dev/null | sed 's:/*$::' | sort -u)
read DELC CL < <(python3 -c "import json; h=json.load(open('$DN'))['hosts'].get('$H', {}); print(int(bool(h.get('delete_caches'))), ' '.join(c['path'] for c in h.get('caches', [])))" 2>/dev/null)
for p in $CL; do
  case $p in /scratch/amartyavarma/*mot*) echo "  cache $p SKIPPED (MoT)"; continue;; esac
  case $p in */inductor-cache-ppm|*/inductor-cache-ppm/*|*/triton-cache-ppm|*/triton-cache-ppm/*) echo "  cache $p SKIPPED (MoT's + the run's: ppm/mot/code/task_ppm.sh)"; continue;; esac
  if printf '%s\n' "$MOTP" | awk -v p="$p" '$0 != "" && (p == $0 || index(p, $0 "/") == 1 || index($0, p "/") == 1) {f = 1} END {exit !f}'; then
    echo "  cache $p SKIPPED (named by a MoT script)"; continue; fi
  [ -e $p ] || continue
  sz=$(du -sb --one-file-system $p 2>/dev/null | cut -f1); echo "  cache $p $(awk -v b=$sz 'BEGIN{printf "%.2f GB", b/1e9}')$([ "${DELC:-0}" = 1 ] && echo ' (to delete)')"
  { [ $DRY = 1 ] || [ "${CLEANCACHE:-0}" != 1 ] || [ "${DELC:-0}" != 1 ]; } && continue
  # (10-27 fix: `nice guard_rm` inside bash -c failed rc 127 -- nice cannot exec a shell function -- so stage 31091 deleted NO cache and
  #  aborted slinky-12; nice now wraps the whole bash)
  nice -n 10 bash -c 'source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_rm "${1:?}"' _ "${p:?}" \
    && stat_line "deleted regenerable cache $p ($sz B; main-approved category, host under need + 5 GB margin)" || stat_line "cache $p NOT deleted (guard refused)"
done

# ---- 3 pin node-local dcp-17850 (inside CKPT) + 4 the resume link. ON-DEMAND FALLBACK (main 10-27): a node-local / v5 copy that fails its
#      sha256 check is LEFT AS IT IS (logged); this host then downloads its own HF dir InternalCan/ppm-v1-c dcp-17850/<dir> straight into the
#      Weka path (c/weka/weka_dcp17850.py --fallback: /data >= 350 GB checked before every file, sha256 == HF LFS, VERIFIED marker) and links it.
fallback() {   # -> sets SRC to the verified Weka dir or aborts
  echo "  FALLBACK: $1 -> HF ppm-v1-c dcp-17850/$HFD into $WK (needs /data >= 350 GB)"
  [ $DRY = 1 ] && { SRC=$WK/dcp-17850/$HFD; return; }
  stat_line "dcp-17850 source $1 failed its sha256 check (left untouched); on-demand HF download of dcp-17850/$HFD into $WK"
  echo "- $(pt) [pose5 ALERT, $H] node-local dcp-17850 source $1 FAILED sha256 vs HF: left untouched (keep-1 prune may delete it later: main decides); run resumes from an HF re-download" \
    | tee -a /data/home/amartyavarma/ppm/control/overnight_monitor_811f.md >> /data/home/amartyavarma/setup-logs/watchdog/ALERTS.log
  HF_HUB_DISABLE_XET=1 HF_XET_HIGH_PERFORMANCE=0 nice -n 10 $PY -I $P5/c/weka/weka_dcp17850.py $HFD --fallback --threads 8 \
    || { say "ABORT on-demand HF fallback for dcp-17850/$HFD failed (see above)"; exit 2; }
  SRC=$WK/dcp-17850/$HFD; stat_line "on-demand HF fallback dcp-17850/$HFD VERIFIED -> $SRC"
}
[ $DRY = 1 ] || mkdir -p /scratch/amartyavarma/pose5-resume
if [ -f $D/COMPLETE ]; then
  if [ $DRY = 1 ]; then $PY -I $K/stage_dcp17850.py $S --dry-run | sed 's/^/  /'
  else nice -n 10 ionice -c3 $PY -I $K/stage_dcp17850.py $S && SRC=$D || fallback "node-local $D"; fi
  [ $DRY = 1 ] && SRC=$D
  if [ "$SRC" = "$D" ] && [ ! -f $PIN/dcp-17850/COMPLETE ]; then
    if [ $DRY = 1 ]; then echo "  [dry] pin: cp -al $D $PIN/dcp-17850 (0 B)"
    else mkdir -p $PIN; rm -rf "${PIN:?}/dcp-17850.tmp"; cp -al $D $PIN/dcp-17850.tmp && [ "$(ls $D | wc -l)" = "$(ls $PIN/dcp-17850.tmp | wc -l)" ] \
         && [ "$(stat -c %i $D/COMPLETE)" = "$(stat -c %i $PIN/dcp-17850.tmp/COMPLETE)" ] && mv -T $PIN/dcp-17850.tmp $PIN/dcp-17850 \
         || { say "ABORT pin of $D failed"; exit 2; }; stat_line "pinned node-local $D -> $PIN/dcp-17850 (hard links, 0 B; main YES 06:33 PT)"; fi
  elif [ "$SRC" = "$D" ]; then echo "  pin present: $PIN/dcp-17850"; fi
  if [ "$SRC" = "$D" ]; then SRC=$PIN/dcp-17850; echo "  resume source = the pin $SRC (same inodes, outside CKPT: keep-1 prune cannot dangle the link)"; fi   # audit 1 #5
elif [ "$S" = slinky-9 ] && [ -f $V5/COMPLETE ]; then
  if [ $DRY = 1 ]; then echo "  [dry] verify $V5 (20 files vs HF) -> link"
  else nice -n 10 $PY -I - "$V5" <<'EOF' && SRC=$V5 || fallback "ppmvox copy $V5"
import hashlib, os, sys
d = sys.argv[1]; C = "/data/home/amartyavarma/ppm/pose5/c/hf17850"
want = {l.split("\t")[0].rsplit("/", 1)[1]: (int(l.split("\t")[1]), l.split("\t")[2]) for l in open(f"{C}/hf_files.tsv").read().splitlines()[1:]
        if l.startswith("dcp-17850/slinky-9/")}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(16 << 20), b""): h.update(c)
    return h.hexdigest()
bad = [f for f, (s, x) in want.items() if not os.path.isfile(f"{d}/{f}") or os.path.getsize(f"{d}/{f}") != s
       or (x and sha(f"{d}/{f}") != x) or (not x and open(f"{d}/{f}", "rb").read() != open(f"{C}/dl/dcp-17850/slinky-9/{f}", "rb").read())]
print("  v5 slot-9 copy:", "OK 20/20 == HF" if not bad and len(want) == 20 else f"BAD {bad[:4]}"); sys.exit(1 if bad or len(want) != 20 else 0)
EOF
  fi
  [ $DRY = 1 ] && SRC=$V5
else
  SRC=$WK/dcp-17850/$HFD
  if [ ! -f $WK/VERIFIED-$HFD.json ]; then
    case $HFD in slinky-1|slinky-2|slinky-7|slinky-9|slinky-14) fallback "missing node-local copy (expected on this host)";;      # on-demand HF
      *) [ $DRY = 1 ] && echo "  [dry] Weka copy $SRC not VERIFIED yet" || { say "ABORT Weka copy $SRC not VERIFIED"; exit 2; };; esac
  else python3 - "$SRC" "$WK/VERIFIED-$HFD.json" <<'EOF' || { say "ABORT Weka copy sizes differ"; exit 2; }
import json, os, sys
d, v = sys.argv[1], json.load(open(sys.argv[2]))
bad = [f for f, m in v["files"].items() if not os.path.isfile(f"{d}/{f}") or os.path.getsize(f"{d}/{f}") != m["size"]]
print(f"  Weka copy {d}: {len(v['files']) - len(bad)}/20 sizes == (sha256 verified at copy {v['verified']})"); sys.exit(1 if bad or len(v["files"]) != 20 else 0)
EOF
  fi
fi
if [ $DRY = 1 ]; then echo "  [dry] ln -sfn $SRC $RL"
else ln -sfn $SRC $RL.tmp && mv -T $RL.tmp $RL && [ -f $RL/COMPLETE ] && [ "$(cat $RL/COMPLETE)" = 17850 ] || { say "ABORT resume link $RL"; exit 2; }; fi
echo "  resume link $RL -> $SRC"

# ---- 5 slinky-15: slot-9 PPM pose store (before the npl5 window, so the window budget sees it)
if [ "$S" = slinky-9 ]; then
  if [ $DRY = 1 ]; then bash $P5/slot9/stage_slot9_extras.sh --dry-run 2>&1 | sed 's/^/  /'
  else nice -n 10 ionice -c3 bash $P5/slot9/stage_slot9_extras.sh || { say "ABORT slot-9 pose store"; exit 2; }; fi
fi

# ---- 6 B's hooks (unchanged; B owns them) + the npl5 roll window size (AUTOSHRINK=1, main OK 10-27: a host whose free cannot hold B's default
#      window gets the largest --ahead >= 40 that fits, from B's WINDOW-<key>-a<ahead>-b<back>.json; NPL5_AHEAD is exported to B's prep and
#      written to kit/state/AHEAD-<host> for B's roll daemon; logged)
GR=$K/gate_rules.py     # main 10-27: the SAME rules as the pre-pause gate (c/disk/build_disk_need.py imports it); slinky-13: window on Weka, ahead 100
# B 10-27 (npl5/S13_WEKA_FOR_C.txt; main OK): a Weka-served host (gate_rules WEKA_HOSTS: slinky-13) gets its canonical npl5 jobs dir as a
# SYMLINK to <weka>/jobs, BEFORE B's hooks / prep (the plan's src_root and digests are unchanged; the trainer, fresh_pose check and B's
# fetcher open f"{src_root}/{file}" through it). A real dir/file there is NEVER replaced (ABORT); a symlink elsewhere -> ABORT. Its fetcher
# floor is on /data (statvfs of the resolved dest): MIN_FREE_GB = 350 (/data >= 350 GB) and AHEAD = 100, written HERE and never overwritten
# below. B's fetcher v2.1 refuses to start unless the link resolves exactly to npl5-weka/<host>/jobs (exit 4).
if WKP=$(python3 -I $GR weka $H); then
  WKJ="${WKP:?}/jobs"; CJ=/scratch/amartyavarma/npl-append/npl5/jobs
  if [ $DRY = 1 ]; then echo "  [dry] mkdir -p $WKJ; ln -s $WKJ $CJ (only if $CJ is absent or already that symlink); MIN_FREE_GB-$H = 350, AHEAD-$H = 100"
  else
    mkdir -p "${WKJ:?}" /scratch/amartyavarma/npl-append/npl5
    if [ -L "$CJ" ]; then
      [ "$(readlink -f "$CJ")" = "$(readlink -f "$WKJ")" ] || { say "ABORT $CJ is a symlink to $(readlink "$CJ"), not $WKJ"; exit 2; }
      echo "  $CJ -> $WKJ already (kept)"
    elif [ -e "$CJ" ]; then
      say "ABORT $CJ exists as a real dir/file on $H: not replacing it (move its files to $WKJ by hand, or remove it if empty)"; exit 2
    else
      ln -s "${WKJ:?}" "${CJ:?}" || { say "ABORT symlink $CJ"; exit 2; }
      stat_line "npl5 window on Weka: $CJ -> $WKJ (main OK 10-27; B's S13_WEKA_FOR_C)"
    fi
    echo 350 > $K/state/MIN_FREE_GB-$H     # the fetcher's floor is now on /data: keep /data >= 350 GB (main's floor)
    echo 100 > $K/state/AHEAD-$H           # nothing on this host's /scratch: the full ahead fits (the window is on /data)
  fi
fi
B9=${B_SLOT9_HOOK:-$P5/npl5/bin/leg1_slot9_hook.sh}
if [ $DRY = 1 ]; then echo "  [dry] bash $B9   (B: slot-9 plans + npl3 window; check-only on other hosts)"
else nice -n 10 bash $B9 > $K/logs/b9-$H.log 2>&1 || { say "ABORT B's slot-9/npl3 hook (see $K/logs/b9-$H.log)"; exit 2; }; fi
KEY=$([ "$S" = slinky-9 ] && echo slinky-9 || echo $H)
BACKMODE=oldest                                                         # B v2 10-27: BACKMODE newest dropped (saved nothing with keep=1)
# B v2: the fetcher's per-host disk floor = room for the trainer's next save (space_check 1.15 x 46.09 = 53 GB) + this host's floor
# (Weka-served hosts: 350, written above; not overwritten here)
if [ -z "$WKP" ]; then MINF=$(awk -v f=$FLOOR 'BEGIN{printf "%.0f", 53 + f}')
  [ $DRY = 1 ] && echo "  [dry] kit/state/MIN_FREE_GB-$H = $MINF" || echo $MINF > $K/state/MIN_FREE_GB-$H; fi
read AHEAD WGB < <(python3 -I $GR ahead $H "$(free_gb)" ${HFB_FLOOR:-30} ${AUTOSHRINK:-0})
[ -n "$WKP" ] && { AHEAD=100; WGB=0.0; }                                 # Weka-served: = the AHEAD file written above (window on /data)
[[ "$AHEAD" =~ ^[0-9]+$ ]] && [[ "$WGB" =~ ^[0-9.]+$ ]] || { say "ABORT gate_rules.py ahead gave '$AHEAD' '$WGB'"; exit 2; }   # (audit 2 NIT: never fail open)
echo "  npl5 roll: key $KEY ahead $AHEAD steps, /scratch window ${WGB} GB (gate_rules.py; b110)${WKP:+; window on Weka $WKP}; free $(free_gb) GB"
[ $DRY = 1 ] || [ -n "$WKP" ] || echo "$AHEAD" > $K/state/AHEAD-$H
[ $DRY = 1 ] || stat_line "npl5 roll --ahead $AHEAD steps (/scratch window $WGB GB${WKP:+; on Weka})"
if [ -n "${B_NPL5_PREP:-}" ]; then
  if [ $DRY = 1 ]; then echo "  [dry] NPL5_AHEAD=$AHEAD $B_NPL5_PREP   (B: npl5 plan dir + initial roll window)"
  else NPL5_AHEAD=$AHEAD nice -n 10 bash -c "$B_NPL5_PREP" > $K/logs/b5-$H.log 2>&1 || { say "ABORT B's npl5 prep (see $K/logs/b5-$H.log)"; exit 2; }; fi
else [ $DRY = 1 ] && echo "  [dry] B_NPL5_PREP not set yet (B's npl5 plan + initial window command)" || { say "ABORT B_NPL5_PREP not set"; exit 2; }; fi

# ---- 7 gate + trainer-side checks (fail closed)
HAVE5=$(du -sb --one-file-system /scratch/amartyavarma/npl-append/npl5 2>/dev/null | awk '{printf "%.1f", $1/1e9}')
PEND=$(awk -v w=$WGB -v h=${HAVE5:-0} 'BEGIN{r = w - h; printf "%.1f", (r > 0 ? r : 0)}')     # window not yet on disk (B's daemon fetches it)
read NEED LEFT GOK < <(python3 -I $GR eval $H "$(free_gb)" $PEND ${HFB_FLOOR:-30})
F=$(free_gb)
CHK=$(python3 - "$S" <<'EOF'
import glob, json, os, sys
S = sys.argv[1]; R = "/scratch/amartyavarma"
SW = {"slinky-4": "slinky-14", "slinky-7": "slinky-5", "slinky-14": "slinky-19"}   # = the cfg node_swap
out = []
for nm, root in (("pose", f"{R}/pose-s3-samdata/rows"), ("speech", f"{R}/s3-rows")):
    try:
        n = json.load(open(f"{root}/NODE.json"))["node"]; out.append(f"{nm}:{n}" + ("" if (n == S or SW.get(n) == S) else "!WRONG"))
    except Exception:
        out.append(f"{nm}:MISSING!WRONG")
hosts = set()
for r in ("ppm-fresh-pose", "ppm-fresh-pose-npl2", "ppm-fresh-pose-npl3", "ppm-fresh-pose-npl5"):
    try:
        hosts.add(json.load(open(f"{R}/{r}/PLAN.json"))["host"]); os.stat(f"{R}/{r}/PLAN.npz")
    except Exception:
        out.append(f"plan:{r}!WRONG")
out.append("planhost:" + ",".join(sorted(hosts)) + ("" if hosts and hosts <= {S, os.uname().nodename.split('.')[0]} else "!WRONG"))
out.append("txtdup:" + ("ok" if os.path.isfile(f"{R}/ppm/txtdup-full/MANIFEST.json") else "MISSING!WRONG"))
rl = f"{R}/pose5-resume/dcp-17850"
try:
    out.append("resume:" + ("ok" if os.path.isfile(f"{rl}/COMPLETE") and json.load(open(f"{rl}/train_state.json"))["step"] == 17850 else "MISSING!WRONG"))
except Exception:
    out.append("resume:MISSING!WRONG")
ck = f"{R}/ppm/ckpt/ppm-v1"
newer = [d for d in (os.listdir(ck) if os.path.isdir(ck) else []) if d.startswith("dcp-") and d[4:].isdigit() and int(d[4:]) > 17850]
out.append("newer_saves:" + (",".join(newer) + "!WRONG" if newer else "none"))
# audit 1 #16: keep-1 prunes every older COMPLETE save except 16861 at the first pose save; anything in CKPT other than 16861 / 17798 (dropped
# at this switch) / 17850 (pinned) must have a PASS HF verify record for THIS host, else the stage aborts and lists it (main decides)
phys = os.uname().nodename.split(".")[0]
unk = []
for d in (os.listdir(ck) if os.path.isdir(ck) else []):
    if not (d.startswith("dcp-") and d[4:].isdigit()) or int(d[4:]) in (16861, 17798, 17850) or int(d[4:]) > 17850:
        continue
    ok = False
    for f in glob.glob(f"/data/home/amartyavarma/setup-logs/hfb/verify/InternalCan__ppm-v1*__{d}__{phys}.json"):
        try:
            ok = ok or json.load(open(f)).get("PASS") is True
        except Exception:
            pass
    if not ok:
        unk.append(d)
out.append("older_saves_without_HF:" + (",".join(unk) + "!WRONG" if unk else "none"))
print(" ".join(out))
EOF
)
echo "  gate (gate_rules.py, same as pre-pause): free $F GB vs need $NEED GB (saves+floor, margin $([ $H = slinky-13 ] && echo 3.5 || echo 5), window still to fetch $PEND) -> left $LEFT $GOK; trainer-side: $CHK"
SHORT=$([ "$GOK" = OK ] || awk -v l=$LEFT 'BEGIN{printf "%.1f", -l}')
if [ $DRY = 1 ]; then say "DRY free $F need $NEED${SHORT:+ SHORT $SHORT}; $CHK"; exit 0; fi
[ -z "$SHORT" ] || { say "ABORT gate: free $F GB < need $NEED GB (short $SHORT GB)"; exit 2; }
echo "$CHK" | grep -q '!WRONG' && { say "ABORT trainer-side check: $CHK"; exit 2; }
say "OK free $F GB (need $NEED); $CHK"

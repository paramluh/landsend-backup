#!/bin/bash
# monosam go/ (10-26): shared definitions for mot_stop.sh / mot_resume.sh (sourced, never run). Read-only helpers only.
M=/data/home/amartyavarma/ppm/mot
GO=/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/go
HC=$GO/mot_hostcheck.sh                                   # read-only per-host probe
ENVF=$M/mot.env; SB=$M/mot.sbatch; LREC=$M/launch/LAST_LAUNCH.env
CTL_DEFAULT=$M/control                                    # mot.sbatch: PPM_CONTROL_DIR default; mot.env sets the same
CKPT=/scratch/amartyavarma/ppm-mot/ckpt/ppm-mot          # mot.env PPM_SAVE_ROOT + run_base ppm-mot (train_ilp.py CKPT)
STATE=$GO/MOT_STOP_STATE; STEPF=$GO/MOT_STOP_STEP; INPROG=$GO/MOT_PAUSE_IN_PROGRESS
GL=$M/GUARD_LOG.md
SAVE_GIB=33.673                                           # STATUS.md / launch.sh: one MoT save per host (tools/save_size.py)
NEED_SAVE_GIB=63.74                                       # safety_ppm.space_check: 33.673 x PPM_SAVE_MARGIN 1.002 + PPM_SAVE_FLOOR_GB 30
GUARD_NEED_GIB=66                                         # guard/tick.sh NEED_GIB (alert threshold)
MAXSTEP=17366
# stray Slurm / PPM / ILP / MOT vars of the calling shell must not leak into srun / sbatch (as launch.sh / swap_exec.sh / the watchdog)
UNSET=$(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC|PPM|ILP|MOT)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /' | tr '\n' ' ')

pt()   { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
ptm()  { TZ=America/Los_Angeles date '+%m-%d %H:%M'; }
say()  { echo "[$(pt)] $*"; }
phase(){ echo; echo "######## [$(pt)] PHASE $*"; }
ok()   { echo "   ok   $*"; }
warn() { echo "   WARN $*"; WARNS+=("$*"); }
fail() { echo "   FAIL $*"; FAIL+=("$*"); }
would(){ echo "   WOULD $*"; }
FAIL=(); WARNS=()
glog() { [ "${DRY:-1}" = 1 ] && return 0; echo "- $(ptm) PT: [monosam planned pause] $*" >> "$GL"; }

# "[10-26 05:11:36 PT] ..." -> epoch (PT)
line_epoch() { local d t; d=$(cut -c2-6 <<<"$1"); t=$(cut -c8-15 <<<"$1"); TZ=America/Los_Angeles date -d "$(date +%Y)-$d $t" +%s 2>/dev/null || echo 0; }
last_step_line() { grep -aE '\] step [0-9]+/[0-9]+ [0-9.]+s ' "$1" 2>/dev/null | tail -1; }
step_of() { sed -nE 's/.*\] step ([0-9]+)\/[0-9]+ .*/\1/p' <<<"$1"; }
sps_avg() { grep -aE '\] step [0-9]+/[0-9]+ [0-9.]+s ' "$1" | tail -${2:-28} | sed -nE 's/.*\] step [0-9]+\/[0-9]+ ([0-9.]+)s .*/\1/p' | awk '{s+=$1;n++} END{if(n) printf "%.1f", s/n; else print "NA"}'; }
# first metric after '|' on a step line = speech word_ce_norm; mean / sd over the last N step lines
wce_stats() { grep -aE '\] step [0-9]+/[0-9]+ [0-9.]+s ' "$1" | tail -${2:-50} | sed -nE 's/.*\| word_ce_norm=([0-9.]+) .*/\1/p' | awk '{s+=$1;q+=$1*$1;n++} END{if(n){m=s/n; v=q/n-m*m; if(v<0)v=0; printf "%.4f %.4f %d", m, sqrt(v), n} else print "NA NA 0"}'; }
# EVAL lines: "<step> <eval_speech/word_ce_norm>"
evals() { grep -aE '\] EVAL step [0-9]+: eval_speech/word_ce_norm' "$1" | sed -nE 's/.*EVAL step ([0-9]+): eval_speech\/word_ce_norm ([0-9.]+) .*/\1 \2/p'; }

# recipe fingerprint: env, sbatch, host file, every code file the job imports (tests / __pycache__ excluded)
recipe_sha() {
  local nodef=$1
  { sha256sum "$ENVF" "$SB" "$nodef"
    find "$M/code" -type f -not -path '*/__pycache__/*' -not -path '*/tests/*' -print0 | sort -z | xargs -0 sha256sum; } 2>/dev/null \
    | awk '{print $1}' | sha256sum | cut -c1-16
}
recipe_newest_mtime() {   # newest mtime (epoch) of the recipe files
  local nodef=$1
  { stat -c %Y "$ENVF" "$SB" "$nodef"; find "$M/code" -type f -not -path '*/__pycache__/*' -not -path '*/tests/*' -printf '%T@\n'; } 2>/dev/null | cut -d. -f1 | sort -n | tail -1
}
# resolved trainer cfg (the "... cfg {json}" line), run_tag masked -> sha16 (resumes differ only in run_tag: GUARD_LOG 10-24 08:33)
cfg_sha() { grep -a -m1 -E '\] ILP ppm-mot-j[0-9]+: .* cfg \{' "$1" 2>/dev/null | sed -E 's/^.* cfg \{/{/; s/"run_tag": "[^"]*"/"run_tag": "X"/' | sha256sum | cut -c1-16; }
# the cfg line can be cut by another rank's stderr (job 29514: a torch "[W1024 ...] Warning" landed inside it) -> 0 = clean single line
cfg_clean() { local l; l=$(grep -a -m1 -E '\] ILP ppm-mot-j[0-9]+: .* cfg \{' "$1" 2>/dev/null); [ -n "$l" ] && ! grep -qE '\[W[0-9]{4} |Warning' <<<"$l" && grep -qE '\}[[:space:]]*$' <<<"$l"; }
cfg_keys() { grep -a -m1 -E '\] ILP ppm-mot-j[0-9]+: .* cfg \{' "$1" 2>/dev/null | sed -E 's/^.* cfg \{/{/; s/"run_tag": "[^"]*"/"run_tag": "X"/' | sed 's/, "/\n"/g'; }
sub_ovr() { sed -n '1,5p' "$1" 2>/dev/null | grep -a -m1 '^== submission overrides:' | sed 's/^== submission overrides: //' | tr ' ' '\n' | grep . | sort | paste -sd' '; }
rank_order() { sed -n '1,8p' "$1" 2>/dev/null | grep -a -m1 '^== host alias' | sed -E 's/.*rank order //'; }

# read-only host scan. $1 = running job to overlap ("" = new tiny allocation on the hosts), $2 = hosts csv, $3 = step to check (optional)
hostscan() {
  local j=$1 hosts=$2 want=${3:-} n
  n=$(tr , '\n' <<<"$hosts" | grep -c .)
  if [ -n "$j" ]; then
    timeout 180 env $UNSET srun --overlap --jobid "$j" -N"$n" -w "$hosts" --ntasks-per-node=1 -c1 --mem=256M --cpu-bind=none -t 3 \
      nice -n 19 bash "$HC" $want 2>&1 | grep -a '^MOTHC' | sort -V
  else
    timeout 420 env $UNSET srun -p all -N"$n" -w "$hosts" --ntasks-per-node=1 -c1 --mem=256M --gres=none -t 3 --immediate=300 \
      -J mot-savecheck nice -n 19 bash "$HC" $want 2>&1 | grep -a '^MOTHC' | sort -V
  fi
}
# summary of a scan: "<hosts answered> <min free GiB> <min-free host> <newest common step> <hosts want=OK> <hosts with .tmp> <hosts with hfhold>"
scan_summary() {
  awk '{h=$2; f=$3; sub("free_GiB=","",f); nw=$4; sub("newest=","",nw); w=$5; t=$6; sub("tmp=","",t);
        n++; if (mn=="" || f+0<mn) {mn=f+0; mh=h}; if (nc=="" || nw+0<nc) nc=nw+0; if (w ~ /:OK$/) okc++; if (t+0>0) tc++;
        if ($0 ~ /hfhold=\[[^]]+\]/) hc++}
       END {printf "%d %s %s %s %d %d %d\n", n, (mn==""?"NA":mn), (mh==""?"NA":mh), (nc==""?"-1":nc), okc, tc, hc}' <<<"$1"
}

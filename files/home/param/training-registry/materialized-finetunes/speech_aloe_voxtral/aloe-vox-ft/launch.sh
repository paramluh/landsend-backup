#!/bin/bash
# aloe-vox-ft: EVERYTHING after the user's "go" (PPM main paused with STOP_AFTER_SAVE). One command, run from the login node (only squeue /
# sbatch / file reads here; all compute goes to the submitted jobs):
#     bash /data/home/amartyavarma/ppm/aloe-vox-ft/launch.sh
# Steps: (1) assert PPM stopped WITH a save (job gone from the queue, its log's "[control] STOP_AFTER_SAVE at step N: saved .../dcp-N -> exiting
# cleanly" line, control STOP_AFTER_SAVE.done-N); (2) resolve the init (QQA_INIT env overrides): local-N inference copy if it exists, else the
# newest Weka export w-M if M >= N, else convert dcp-N -> /data local-N (ckpt_conv.sbatch on the 16 PPM hosts, training afterok); print it;
# (3) assert the 8 prepared hosts are free + every store / FLAC ledger + STORES.json present; (4) sbatch the training (aloe_vox.sbatch),
# the GCS backup (afterok) and the live page refresher (ppm_aloe_vox.html on the klweb tunnel). DRY=1: print the resolution, submit nothing.
set -uo pipefail
V=/data/home/amartyavarma/ppm/aloe-vox-ft
HOSTS=slinky-0,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-12
PPM16=slinky-0,slinky-1,slinky-2,slinky-3,slinky-5,slinky-6,slinky-8,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-17,slinky-19
CTL=/data/home/amartyavarma/ppm/control/ppm-v1; CK=/scratch/amartyavarma/ppm/ckpt/ppm-v1
INF=/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1; WEKA=/data/home/amartyavarma/ppm/ckpt-weka/ppm-v1
PPM_JOB=${PPM_JOB:-$(cat $CTL/CURRENT_JOB 2>/dev/null)}
LOG=${PPM_LOG:-/data/home/amartyavarma/ppm/logs/ppm-$PPM_JOB.log}   # PPM_LOG: tests only
die() { echo "ABORT: $*"; exit 1; }
pt() { TZ=America/Los_Angeles date '+%F %T PT'; }
echo "== aloe-vox-ft launch $(pt)"
# ---- (1) PPM stopped with a save
[ -n "$PPM_JOB" ] && [ -f "$LOG" ] || die "PPM job / log unknown (PPM_JOB=$PPM_JOB)"
if squeue -h -j "$PPM_JOB" -o %T 2>/dev/null | grep -q .; then die "PPM job $PPM_JOB is still in the queue ($(squeue -h -j $PPM_JOB -o %T)): pause it with a save first (touch $CTL/STOP_AFTER_SAVE) and wait for it to exit"; fi
[ -z "$(squeue -h -u $USER -n ppm -o %i)" ] || [ "${DRY:-0}" = 1 ] || die "another job named ppm is queued/running: $(squeue -h -u $USER -n ppm -o '%i %T')"
SL=$(grep -E "\[control\] STOP_AFTER_SAVE at step [0-9]+: saved $CK/dcp-[0-9]+ -> exiting cleanly" "$LOG" | tail -1)
[ -n "$SL" ] || die "no 'STOP_AFTER_SAVE ... saved ... exiting cleanly' line in $LOG: PPM did not stop with a save"
N=$(echo "$SL" | sed -E 's/.*at step ([0-9]+):.*/\1/')
LASTSAVE=$(grep -oE "\[save\] $CK/dcp-[0-9]+ node-local" "$LOG" | tail -1 | grep -oE "dcp-[0-9]+" | cut -d- -f2)
[ "$LASTSAVE" = "$N" ] || die "the stop step $N is not the newest node-local save ($LASTSAVE)"
[ -f "$CTL/STOP_AFTER_SAVE.done-$N" ] || echo "WARN: $CTL/STOP_AFTER_SAVE.done-$N missing (stop line present: continuing)"
echo "PPM job $PPM_JOB stopped with a save at step $N: $SL"
# ---- (2) init
M=$(ls -d $WEKA/w-* 2>/dev/null | while read d; do [ -f $d/COMPLETE ] && basename $d | cut -d- -f2; done | sort -n | tail -1)
CONV_DEP=""
if [ -n "${QQA_INIT:-}" ]; then
  INIT=$QQA_INIT; WHY="QQA_INIT override"
elif [ -f $INF/local-$N/COMPLETE ]; then
  INIT=$INF/local-$N; WHY="existing inference copy of the stop save dcp-$N"
elif [ -n "$M" ] && [ "$M" -ge "$N" ]; then
  INIT=$WEKA/w-$M; WHY="newest Weka export w-$M (>= stop save $N)"
else
  INIT=$INF/local-$N; WHY="stop save dcp-$N converted to bf16 shards on /data by ckpt_conv.sbatch (newest Weka export w-${M:-none} is older)"
  FREE=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
  [ "$FREE" -ge 390 ] || die "/data free ${FREE} GB: the ~84 GB conversion would go below the 300 GB floor (set QQA_INIT to the Weka export $WEKA/w-$M instead)"
  if [ "${DRY:-0}" != 1 ]; then
    CJ=$(SRC=$CK/dcp-$N OUT=$INIT sbatch --parsable --export=ALL -w $PPM16 $V/ckpt_conv.sbatch) || die "conversion sbatch failed"
    CONV_DEP="--dependency=afterok:$CJ"; echo "conversion job $CJ: $CK/dcp-$N -> $INIT (16 PPM hosts, ~2 min)"
  fi
fi
echo "INIT = $INIT   ($WHY)"
# ---- (3) hosts + data
for h in ${HOSTS//,/ }; do
  J=$(squeue -h -w $h -o "%i %j" | grep -v aloevox-ckconv)
  if [ -n "$J" ]; then [ "${DRY:-0}" = 1 ] && echo "DRY: $h not free: $J" || die "$h is not free: $J"; fi
done
[ -f $V/plan/STORES.json ] || die "plan/STORES.json missing"
python3 - <<PYEOF || die "STORES.json does not list every prod host's store"
import json; s = json.load(open("$V/plan/STORES.json")); hosts = "$HOSTS".split(",")
bad = [k for k in range(8) if hosts[k] not in s[f"p{k}"]["hosts"]]
assert not bad, ("stores not verified on their prod host", bad)
PYEOF
# ---- (4) submit
[ "${DRY:-0}" = 1 ] && { echo "DRY=1: nothing submitted"; exit 0; }
TJ=$(QQA_INIT=$INIT sbatch --parsable --export=ALL -w $HOSTS --cpus-per-task=8 --mem=540G $CONV_DEP $V/aloe_vox.sbatch) || die "training sbatch failed"
GJ=$(sbatch --parsable --dependency=afterok:$TJ $V/gcs_local_exports.sbatch) || echo "WARN: GCS backup sbatch failed (resubmit: sbatch --dependency=afterok:$TJ $V/gcs_local_exports.sbatch)"
RJ=$(squeue -h -u $USER -n klweb-ppm-aloe-vox-refresh -o %i | head -1)
[ -n "$RJ" ] || RJ=$(sbatch --parsable /data/home/amartyavarma/setup-logs/klweb/ppm_aloe_vox_refresh.sbatch)
cat <<EOT
== submitted $(pt)
  training  job $TJ  (aloevoxft, 8 nodes $HOSTS, init $INIT${CONV_DEP:+, after conversion $CONV_DEP})  log $V/logs/aloevoxft-$TJ.log
  GCS backup job $GJ (afterok:$TJ)   live page refresher job $RJ
  live page: $(cat /data/home/amartyavarma/setup-logs/klweb/TUNNEL_URL)/ppm_aloe_vox.html
  stop with save: touch $V/control/STOP_AFTER_SAVE   (or scancel -b -s USR1 $TJ)
EOT

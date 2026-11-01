#!/bin/bash
# MONOFT one-command launcher (prepared 10-22 PT; user 10-22 ~21:55 PT: "fine-tune on the monoplex data on 8 nodes ... pause the main run and use 8
# of its nodes"). Run by the MAIN session AFTER the main run (ppm811) is paused with STOP_AFTER_SAVE (ppm/monoft/pause_main.sh prints S):
#
#   bash /data/home/amartyavarma/ppm/monoft/launch.sh <S> <8 FT hosts | default> [final save 0|1, default 0]
#
# The FT = aloe-v2pure-17641 (job 27622) VERBATIM except the init = the main run's save dcp-<S>. It:
#  1. checks: no ppm811 job queued (main paused), 8 hosts out of the 16 PPM hosts (none of slinky-4/10/15/18), no aloev2pm<S> job yet;
#  2. creates /data/home/amartyavarma/ppm/aloe-v2pure-main-<S> from ppm/monoft/template (@S@ rendered) + a byte-identical copy of 17641's code/;
#     CFG_DIFF.txt = run.env vs 17641's (only the allowed path / name keys may differ; code/ must be identical) else ABORT;
#  3. submits the init distribution job (tools/init_convert.sbatch: dcp-<S> -> node-local bf16 shards on the 8 FT hosts, no /data copy;
#     load_init-checked), then CHAINED: the FT (afterok), xfree (after: intermediate exports -> private HF, node copies freed), the final export
#     HF backup (afterok; private InternalCan/ppm-aloe-v2pure-main-<S>-final; never GCS) and the save + init-copy cleanup (afterok), all with
#     --kill-on-invalid-dep=yes (a failed init cancels the chain instead of leaving it pending);
#  4. registers the point on the samdata_ft scaling page (tools/samdata_register.py), writes BACKUP_JOB / LAUNCH.json, appends to STATUS.md.
# Returns in seconds; progress: <run dir>/logs/init_convert-<job>.log, then logs/aloev2pm<S>-<job>.log.
set -uo pipefail
S=${1:?usage: launch.sh <S> <8 FT hosts|default> [final save 0|1]}; HOSTS=${2:?hosts}; FS=${3:-0}
M=/data/home/amartyavarma/ppm/monoft; TPL=$M/template; REF=/data/home/amartyavarma/ppm/aloe-v2pure-17641
NODES16=$(cat /data/home/amartyavarma/ppm/code-811/nodes16_s7.txt)
RUN=aloe-v2pure-main-$S; Q=/data/home/amartyavarma/ppm/$RUN; LOG=$M/logs/launch.log
TEST=${MF_TEST:-0}   # MF_TEST=1: rehearsal: run dir under monoft/dryrun/launchtest, main-run check = warning, sbatch / scancel / page / STATUS only echoed
[ "$TEST" = 1 ] && { Q=$M/dryrun/launchtest/$RUN; LOG=$M/dryrun/launchtest/launch.log; mkdir -p $M/dryrun/launchtest; }
mkdir -p $M/logs
pt() { TZ=America/Los_Angeles date '+%m-%d %T PT'; }
say() { echo "[$(pt)] $*" | tee -a $LOG; }
die() { say "ABORT: $*"; exit 1; }
clean_sb() {
  if [ "$TEST" = 1 ]; then echo "[TEST] sbatch $* (env: $(env | grep -oE '^(MF|XF|FTB|HFJ)_[A-Z_]*=[^ ]*' | grep -v '^MF_ALL_HOSTS' | paste -sd' '))" >> $LOG; echo "T$RANDOM"; return; fi
  env $(env | grep -oE '^(SLURM|SBATCH|SRUN|SALLOC)_[A-Za-z0-9_]*=' | tr -d = | sed 's/^/-u /') sbatch "$@"
}
vsort() { tr , '\n' | grep . | sort -V | paste -sd,; }
# ---- 1. checks
[[ "$S" =~ ^[0-9]+$ ]] || die "S '$S' is not a step number"
[[ "$FS" =~ ^[01]$ ]] || die "final save must be 0 or 1"
[ "$HOSTS" = default ] && HOSTS=$(cat $M/FT_HOSTS.txt)
FT=$(echo $HOSTS | vsort)
[ "$(echo $FT | tr , '\n' | wc -l)" = 8 ] || die "need 8 FT hosts, got '$FT'"
[ "$(echo $FT | tr , '\n' | sort -u | wc -l)" = 8 ] || die "duplicate host in '$FT'"
for h in $(echo $FT | tr , ' '); do
  echo ",$NODES16," | grep -q ",$h," || die "$h is not one of the 16 PPM hosts"
  case $h in slinky-4|slinky-10|slinky-15|slinky-18) die "$h is forbidden (slinky-10 rail 7 / spares)";; esac
done
OTHER=$(for h in $(echo $NODES16 | tr , ' '); do echo ",$FT," | grep -q ",$h," || echo $h; done | paste -sd, | vsort)
MAINJ=$(cat /data/home/amartyavarma/ppm/control/ppm-v1/CURRENT_JOB_811 2>/dev/null)
if [ -n "$MAINJ" ] && squeue -h -j "$MAINJ" -o %i 2>/dev/null | grep -q .; then
  [ "$TEST" = 1 ] && say "[TEST] main run $MAINJ still running (would ABORT)" || die "the main run (job $MAINJ) is still queued/running: pause it first (monoft/pause_main.sh)"
fi
for h in $(echo $FT | tr , ' '); do
  g=$(squeue -h -t R,CF,PD -w $h -o "%i %j %b" | grep gpu | head -1); [ -z "$g" ] || say "WARNING: FT host $h has a GPU job ($g): the FT queues behind it"
done
[ -z "$(squeue -h -u amartyavarma -n aloev2pm$S -o %i)" ] || die "an aloev2pm$S job already exists"
[ -f $Q/LAUNCH.json ] && [ "${MF_RELAUNCH:-0}" != 1 ] && die "$Q/LAUNCH.json exists (already launched; MF_RELAUNCH=1 to re-launch into the same dir)"
ML=$(ls -t /data/home/amartyavarma/ppm/logs/ppm-*.log | head -1)
if grep -aq "STOP_AFTER_SAVE at step $S: saved" $ML; then say "main log $ML: stopped with a save at step $S"
elif grep -aq "dcp-$S node-local" $ML; then say "main log $ML: dcp-$S is an hourly save (the stop was at a later step?)"
else say "WARNING: no save of step $S in $ML (the init job checks dcp-$S COMPLETE on every host itself)"; fi
# the 8 other hosts: free (one -N16 job) or all inside ONE running job (its overlap step does their conversion)
SRCJ=$(for h in $(echo $OTHER | tr , ' '); do squeue -h -t R,CF -w $h -o "%i %b" | awk '$2 ~ /gpu/ {print $1}'; done | sort -u | { [ "$TEST" = 1 ] && grep -vx "${MAINJ:-x}" || cat; })
if [ -z "$SRCJ" ]; then MODE=all16
elif [ "$(echo "$SRCJ" | wc -l)" = 1 ] && [ "$(squeue -h -j $SRCJ -u amartyavarma -o %i)" = "$SRCJ" ] \
     && [ "$(scontrol show hostnames "$(squeue -h -j $SRCJ -o %N)" | sort -V | paste -sd,)" = "$OTHER" ]; then MODE=overlap
else die "the other hosts $OTHER are held by several / foreign jobs: $(echo $SRCJ | tr '\n' ' ')"; fi
say "launch $RUN: FT hosts $FT (R0 $(echo $FT | cut -d, -f1)), other hosts $OTHER, final save $FS, init job mode $MODE${SRCJ:+ (overlap $SRCJ)}"
# ---- 2. run dir
mkdir -p $Q/logs $Q/control $Q/tools $Q/initctl
rsync -a --delete --exclude __pycache__ $REF/code/ $Q/code/ || die "code copy failed"
for f in run.env run.sh run.sbatch hf_final_export.sbatch rm_saves_after_hf.sbatch tools/xfree_run.sbatch; do
  sed "s/@S@/$S/g" $TPL/$f.tpl > $Q/$f || die "render $f"
done
cp -p $TPL/tools/xfree_run.sh $TPL/tools/ref_cfg_26089.json $Q/tools/; chmod +x $Q/run.sh
rsync -a --exclude __pycache__ $M/tools/ $Q/tools/monoft/ || die "tools snapshot failed"   # the init job runs this frozen copy (never a live-edited script)
grep -l '@S@' $Q/run.env $Q/run.sh $Q/*.sbatch $Q/tools/*.sbatch && die "unrendered @S@"
grep -q '@' $Q/run.env && die "run.env has '@' (run.sh would abort)"
{
  echo "# $RUN run.env vs aloe-v2pure-17641/run.env (job 27622), non-comment lines; $(pt)"
  diff <(grep -v '^#' $REF/run.env) <(grep -v '^#' $Q/run.env)
  echo "# code/ vs aloe-v2pure-17641/code/ (excl. __pycache__): $(diff -rq -x __pycache__ $REF/code $Q/code > /dev/null && echo IDENTICAL || echo DIFFERENT)"
  echo "# recipe-relevant scripts: run.sh / run.sbatch / hf_final_export.sbatch / rm_saves_after_hf.sbatch differ from 17641's only in names / dirs / the init lines (template/*.tpl headers)"
} > $Q/CFG_DIFF.txt
BAD=$(diff <(grep -v '^#' $REF/run.env) <(grep -v '^#' $Q/run.env) | grep '^[<>]' | sed 's/^[<>] //' | cut -d= -f1 | sort -u \
      | grep -vxE 'QQA_RUN|QQA_LOG_ROOT|QQA_PLAN_DIR|QQA_CONTROL_DIR|QQA_EXPORT|ALOE_LOCAL_EXPORT_ROOT|ALOE_SAVE_ROOT|QQA_REF_CFG')
[ -z "$BAD" ] || die "run.env differs from 17641's in non-path keys: $BAD (see $Q/CFG_DIFF.txt)"
grep -q "IDENTICAL" $Q/CFG_DIFF.txt || die "code/ differs from 17641's"
say "run dir $Q ready (CFG_DIFF.txt: only path / name keys differ; code identical)"
# ---- 3. submit
export MF_S=$S MF_FT_HOSTS=$FT MF_ALL_HOSTS=$NODES16 MF_CTL=$Q/initctl MF_INIT=/scratch/amartyavarma/$RUN-init MF_STAGE=/scratch/amartyavarma/monoft-stage-$S \
  MF_PORT=29891 MF_FINAL_SAVE=$FS MF_TRAINER=$Q/code/train_aloe_pose.py MF_TOOLS=$Q/tools/monoft
if [ $MODE = all16 ]; then
  CJ=$(clean_sb --parsable -J mf-init$S -o $Q/logs/init_convert-%j.log -N16 -w $NODES16 $Q/tools/monoft/init_convert.sbatch)
else
  CJ=$(MF_SRC_JOB=$SRCJ MF_SRC_HOSTS=$OTHER clean_sb --parsable -J mf-init$S -o $Q/logs/init_convert-%j.log -N8 -w $FT $Q/tools/monoft/init_convert.sbatch)
fi
[ -n "$CJ" ] || die "init job sbatch failed"
DEP="--kill-on-invalid-dep=yes"
FJ=$(FTB_FINAL_SAVE=$FS clean_sb --parsable --dependency=afterok:$CJ $DEP -w $FT $Q/run.sbatch)
[ -n "$FJ" ] || { [ "$TEST" = 1 ] || scancel $CJ; die "FT sbatch failed (init job $CJ cancelled)"; }
XJ=$(XF_RUN=$RUN XF_REPO=InternalCan/ppm-$RUN XF_STEPS="33 67 100 133 167" XF_FINAL=200 XF_HOSTS=$FT XF_NTASKS=8,8,8,8,8,8,8,8 \
     XF_TLOG=$Q/logs/aloev2pm$S-$FJ.log XF_TJOB=$FJ clean_sb --parsable --dependency=after:$FJ $DEP -N8 -w $FT $Q/tools/xfree_run.sbatch)
HJ=$(clean_sb --parsable --dependency=afterok:$FJ $DEP -N8 -w $FT $Q/hf_final_export.sbatch)
RJ=$([ -n "$HJ" ] && HFJ=$HJ clean_sb --parsable --dependency=afterok:$HJ $DEP -N8 -w $FT $Q/rm_saves_after_hf.sbatch)
[ -n "$XJ" ] && [ -n "$HJ" ] && [ -n "$RJ" ] || { [ "$TEST" = 1 ] || scancel $CJ $FJ $XJ $HJ $RJ 2>/dev/null; die "chain sbatch failed (xfree '$XJ' hf '$HJ' rm '$RJ'); everything just submitted cancelled"; }
printf '{"train_job": "%s", "hosts": "%s", "final_save": "%s", "xfree_job": "%s", "backup_job": "%s", "rm_saves_job": "%s", "repo_final": "InternalCan/ppm-%s-final", "repo_intermediates": "InternalCan/ppm-%s", "time": "%s"}\n' \
  "$FJ" "$FT" "$FS" "$XJ" "$HJ" "$RJ" "$RUN" "$RUN" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/BACKUP_JOB
printf '{"S": %s, "run": "%s", "init_job": "%s", "init_mode": "%s", "src_overlap_job": "%s", "train_job": "%s", "xfree_job": "%s", "hf_final_job": "%s", "cleanup_job": "%s", "ft_hosts": "%s", "other_hosts": "%s", "final_save": %s, "init": "%s", "time": "%s"}\n' \
  "$S" "$RUN" "$CJ" "$MODE" "${SRCJ:-}" "$FJ" "$XJ" "$HJ" "$RJ" "$FT" "$OTHER" "$FS" "$MF_INIT" "$(TZ=America/Los_Angeles date '+%F %T PT')" > $Q/LAUNCH.json
say "LAUNCHED $RUN: init $CJ ($MODE) -> FT $FJ (afterok) on $FT; xfree $XJ, HF final $HJ, cleanup $RJ"
# ---- 4. scaling page registration (never blocks the launch)
if [ "$TEST" = 1 ]; then say "[TEST] would run: python3 $M/tools/samdata_register.py $S aloev2pm$S"; exit 0
elif [ -f $M/tools/samdata_register.py ]; then
  python3 $M/tools/samdata_register.py $S aloev2pm$S >> $M/logs/samdata_register.log 2>&1 && say "samdata_ft page: point p$S registered" \
    || say "WARNING: samdata_ft registration failed (see $M/logs/samdata_register.log); the run is unaffected"
fi
echo "- $(pt): LAUNCHED $RUN (S $S): init job $CJ ($MODE), FT $FT, final save $FS; chain xfree / HF final / cleanup submitted (ids in $Q/LAUNCH.json)" >> $M/STATUS.md
echo "$FJ"

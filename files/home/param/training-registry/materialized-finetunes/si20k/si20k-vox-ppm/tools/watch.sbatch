#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=2G --gres=none -t 2-00:00:00 -J si20kvox-watch -w slinky-15
# si20k-vox export watcher (copy / verify / launch only, CPU) = aloe-nextvox-6242/tools/watch.sbatch parametrised by run, with the /data budget:
#  * each intermediate node-local export-N (plan/PLAN.env QQA_EXPORT_STEPS): when the trainer logs it complete and /data has >= 300 GB + its size
#    free, every node copies its shard files into <run>/exports/export-N (srun --overlap into the FT job, 1 CPU, nice), sizes verified vs the
#    node-local files, COMPLETE, then the private-HF upload (tools/hf_up.sbatch ... delete-after-verify: the /data copy is removed once HF
#    verifies; the node-local shards stay). One intermediate in flight at a time.
#  * final: the consolidated DCP <run>/export-<steps> COMPLETE -> verified (64 shard files, none empty, train_cfg step, metadata incl. vadapter
#    (6 weights + target_rms) + vgain) -> IMMEDIATELY delete the run's drain file si20k-tts/control/DRAIN_FOR_FT.<tag>, README line,
#    <run>/FINAL_EXPORT_READY (= the path), then the HF upload of the final (kept on /data). The FT job never waits on uploads.
#  * FT job ends without the final export: FT_ENDED_NO_FINAL + log; the drain file stays (a relaunch needs the nodes).
#  * optional 5th arg FREE_INIT (PPM: /data/.../local-12829): once the trainer has logged step 1 (every rank loaded the init) and /data free is
#    below 300 + this run's final export + 30 (INTL final) + 10 GB, the /data init copy is deleted (it is a conversion of the HF-verified,
#    node-local dcp-12829; tools/ckpt_conv_12829_part.sbatch + _finalize re-create it in ~2 min). Logged in BACKUP_LOG.txt and README.
#  * optional 6th / 7th args PEER_RUN PEER_JOB (INTL watcher: the PPM run): once BOTH runs have logged step 1 (both staged the si20k stores
#    node-local) and /data free < 300 + both final exports + 10 GB, the /data copies of the si20k stores (s20n0..3, ~20 GB) are deleted
#    (STORES.json / BUILD.json kept; tools/assemble.sbatch rebuilds them from the prep cache in ~4 min if a relaunch needs them).
# usage: sbatch -o <run>/logs/watch-%j.out watch.sbatch <run dir> <FT job id> <drain tag> <final export GB> [FREE_INIT|-] [PEER_RUN PEER_JOB]
Q=$1; J=$2; TAGD=$3; GB=$4; FREE_INIT=${5:-}; [ "$FREE_INIT" = - ] && FREE_INIT=""; PEER_RUN=${6:-}; PEER_J=${7:-}
STORES_SRC=/data/home/amartyavarma/qwen-1ch-data/si20k-vox/stores
set -a; source $Q/plan/PLAN.env; set +a
RUN=$(basename $Q); EX=$Q/exports; FIN=$QQA_EXPORT; S=$QQA_STEPS; DR=/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.$TAGD
LOC=/scratch/amartyavarma/$RUN/exports; LOGF=$(ls $Q/logs/si20kvox-*-$J.log 2>/dev/null | head -1); BL=$EX/BACKUP_LOG.txt; HFS=/data/home/amartyavarma/ppm/si20k-vox-ppm/tools/hf_up.sbatch
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $EX
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
lg() { echo "[$(pt)] $*" | tee -a $BL; }
inft() { env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER srun --jobid=$J --overlap -N8 -n8 --ntasks-per-node=1 -c1 --mem=2G --gres=none --cpu-bind=none "$@"; }
lg "watcher job $SLURM_JOB_ID on $(hostname -s) for FT job $J ($RUN, steps $S, exports $QQA_EXPORT_STEPS, final $FIN, drain $DR)"
done_steps=" "; hfj=""
while true; do
  [ -n "$LOGF" ] || LOGF=$(ls $Q/logs/si20kvox-*-$J.log 2>/dev/null | head -1)
  if [ -n "$FREE_INIT" ] && [ -f "$FREE_INIT/COMPLETE" ] && grep -q "\] step 1/$S " $LOGF 2>/dev/null && grep -q "\[init\] $FREE_INIT:" $LOGF 2>/dev/null; then
    FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
    if [ "$FD" -lt $((300 + GB + 30 + 10)) ]; then
      rm -rf "$FREE_INIT"
      lg "DELETED the /data init copy $FREE_INIT after step 1 (every rank loaded it; /data free was $FD GB < 300 + $GB + 30 + 10 needed for both final exports); it is a conversion of the HF-verified node-local dcp-12829, re-creatable with tools/ckpt_conv_12829_part.sbatch + _finalize"
      echo "- $(pt): deleted the /data init copy $FREE_INIT after step 1 (/data free $FD GB; re-creatable from dcp-12829 in ~2 min)" >> $Q/README.md
    fi
    FREE_INIT=""
  fi
  if [ -n "$PEER_RUN" ] && [ -d "$STORES_SRC/s20n0" ] && grep -q "\] step 1/$S " $LOGF 2>/dev/null; then
    PLOG=$(ls $PEER_RUN/logs/si20kvox-*-$PEER_J.log 2>/dev/null | head -1)
    PS=$(grep -h '^QQA_STEPS=' $PEER_RUN/plan/PLAN.env | cut -d= -f2)
    if [ -n "$PLOG" ] && grep -q "\] step 1/$PS " $PLOG 2>/dev/null; then
      FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
      if [ "$FD" -lt $((300 + 80 + 32 + 10)) ]; then
        rm -rf $STORES_SRC/s20n0 $STORES_SRC/s20n1 $STORES_SRC/s20n2 $STORES_SRC/s20n3
        lg "DELETED the /data copies of the si20k stores ($STORES_SRC/s20n*) after both runs passed step 1 (/data free was $FD GB); STORES.json kept; tools/assemble.sbatch rebuilds them from the prep cache"
      fi
      PEER_RUN=""
    fi
  fi
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  for N in $(echo $QQA_EXPORT_STEPS | tr , ' '); do
    case "$done_steps" in *" $N "*) continue;; esac
    grep -q "\[export-local\] $LOC/export-$N on every node" $LOGF 2>/dev/null || continue
    if [ -n "$hfj" ] && squeue -h -j $hfj -o %T 2>/dev/null | grep -q .; then break; fi        # one intermediate in flight at a time
    FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
    [ "$FD" -ge $((300 + GB + 10)) ] || { lg "export-$N: waiting for /data space ($FD GB free < 300 + $GB + 10)"; break; }
    D=$EX/export-$N; mkdir -p $D
    t0=$(date +%s)
    out=$(inft bash -c "cd $LOC/export-$N && for f in rank*.pt rep.pt meta.json; do [ -f \$f ] || continue; nice -n 10 cp -p \$f $D/\$f.tmp && mv $D/\$f.tmp $D/\$f && echo \"\$f \$(stat -c %s \$f)\"; done" 2>&1)
    ok=1; nf=0
    while read -r f s; do
      case "$f" in rank*.pt|rep.pt|meta.json) ;; *) continue;; esac
      nf=$((nf + 1)); [ "$(stat -c %s $D/$f 2>/dev/null)" = "$s" ] || { ok=0; lg "export-$N: size mismatch $f"; }
    done <<< "$out"
    nr=$(ls $D | grep -c '^rank[0-9]*\.pt$')
    if [ $ok -eq 1 ] && [ $nr -eq 64 ] && [ -f $D/rep.pt ] && [ -f $D/meta.json ] && [ $nf -eq 66 ]; then
      echo $N > $D/COMPLETE
      lg "export-$N -> Weka $D: COPIED + VERIFIED ($nf files, $(du -sb $D | cut -f1) B, sizes == node-local; $(( $(date +%s) - t0 ))s); /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"
      hfj=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $D export-$N delete-after-verify); lg "export-$N: HF upload job $hfj -> private InternalCan/$RUN/export-$N/ (then the /data copy is deleted)"
      done_steps="$done_steps$N "
    else
      lg "export-$N: copy NOT verified (files $nf, rank files $nr, ok $ok); retry next round"; echo "$out" | tail -3 >> $BL
    fi
    break
  done
  if [ -f $FIN/COMPLETE ] && [ "$(cat $FIN/COMPLETE)" = "$S" ] && grep -q "\[export\] $FIN: consolidated" $LOGF 2>/dev/null; then
    nd=$(ls $FIN | grep -c '^__[0-9]*_0\.distcp$'); ne=$(find $FIN -maxdepth 1 -type f -size 0 | wc -l)
    nk=$($PY -I -c "import pickle,sys; m=pickle.load(open(sys.argv[1]+'/.metadata','rb')).state_dict_metadata; k=list(m); print(len(k), sum(x.startswith('vadapter.') for x in k), sum(x.startswith('vgain.') for x in k))" $FIN 2>&1)
    js=$($PY -I -c "import json,sys; j=json.load(open(sys.argv[1]+'/train_cfg.json')); print(j['step'])" $FIN 2>&1)
    if [ "$nd" -eq 64 ] && [ "$ne" -eq 0 ] && [ "$js" = "$S" ] && [ "$(echo $nk | cut -d' ' -f1)" -ge 600 ] && [ "$(echo $nk | cut -d' ' -f2)" -eq 7 ] && [ "$(echo $nk | cut -d' ' -f3)" -eq 1 ]; then
      rm -f $DR
      lg "FINAL export $FIN VERIFIED (64 shard files, no empty file, train_cfg step $S, metadata tensors / vadapter / vgain = $nk, $(du -sb $FIN | cut -f1) B); DELETED $DR (TTS / other jobs retake the nodes)"
      echo "$FIN" > $Q/FINAL_EXPORT_READY
      echo "- $(pt): FINAL EXPORT (consolidated bf16 DCP, step $S, verified): $FIN  -> FINAL_EXPORT_READY written; drain file deleted." >> $Q/README.md
      h=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $FIN export-$S-dcp); lg "export-$S-dcp: HF upload job $h -> private InternalCan/$RUN/export-$S-dcp/ (kept on /data)"
      lg "watcher done"; exit 0
    else
      lg "final export not verified yet (shards $nd, empty $ne, step $js, keys $nk); retry"
    fi
  fi
  if [ -z "$st" ]; then
    sleep 20
    if ! { [ -f $FIN/COMPLETE ] && grep -q "\[export\] $FIN: consolidated" $LOGF 2>/dev/null; }; then
      lg "FT job $J ended WITHOUT the final export (last log: $(grep -v Warning $LOGF | tail -1 | cut -c1-200)); drain file kept for a relaunch"
      echo "$(pt) FT job $J ended without the final export" > $Q/FT_ENDED_NO_FINAL; exit 1
    fi
  fi
  sleep 15
done

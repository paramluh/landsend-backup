#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=2G --gres=none -t 1-00:00:00 -J nextvox-watch
#SBATCH --exclude=slinky-4,slinky-7,slinky-15,slinky-17,slinky-18,slinky-9,slinky-10,slinky-11,slinky-12,slinky-13,slinky-14,slinky-16,slinky-19
#SBATCH -o /data/home/amartyavarma/ppm/aloe-nextvox-6242/logs/watch-%j.out
# aloe-nextvox-6242 export watcher (launch / copy / verify only, CPU). usage: sbatch watch.sbatch <FT job id>
#  * each intermediate node-local export-N (42 84 126 169 211): once the trainer logs it complete, every node copies its own shard files into
#    /data/.../exports/export-N (srun --overlap into the FT job, 1 CPU, nice), sizes verified against the node-local files, COMPLETE written,
#    then a private-HF upload job (tools/hf_up.sbatch, CPU, never on the training / forbidden nodes).
#  * final: the consolidated DCP /data/.../export-253 COMPLETE (step 253) -> verified (64 shard files + .metadata + train_cfg.json, metadata
#    tensor count, every file non-empty) -> IMMEDIATELY delete si20k-tts/control/DRAIN_FOR_FT.14bvox (TTS retakes the nodes), README line,
#    <run>/FINAL_EXPORT_READY (= the path), then the HF upload of export-253 (tag export-253-dcp). The FT job does not wait for any upload.
#  * the FT job ends without a final export (crash / stop): FT_ENDED_NO_FINAL marker + log; the drain file stays (a relaunch needs the nodes).
J=$1
Q=/data/home/amartyavarma/ppm/aloe-nextvox-6242; EX=$Q/exports; FIN=$Q/export-253; DR=/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.14bvox
LOC=/scratch/amartyavarma/aloe-nextvox-6242/exports; LOGF=$Q/logs/aloe-nextvox-$J.log; BL=$EX/BACKUP_LOG.txt
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $EX
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
lg() { echo "[$(pt)] $*" | tee -a $BL; }
inft() { env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER srun --jobid=$J --overlap -N8 -n8 --ntasks-per-node=1 -c1 --mem=2G --gres=none --cpu-bind=none "$@"; }   # clean env: srun from inside this job into the FT job (10-18 fix)
hfup() { sbatch --parsable $Q/tools/hf_up.sbatch "$1" "$2"; }
lg "watcher job $SLURM_JOB_ID on $(hostname -s) for FT job $J"
done_steps=" "
while true; do
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  for N in 42 84 126 169 211; do
    case "$done_steps" in *" $N "*) continue;; esac
    grep -q "\[export-local\] $LOC/export-$N on every node" $LOGF 2>/dev/null || continue
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
      lg "export-$N -> Weka $D: COPIED + VERIFIED ($nf files = 64 rank files + rep.pt + meta.json, $(du -sb $D | cut -f1) B, sizes == node-local; COMPLETE written; $(( $(date +%s) - t0 ))s); /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"
      h=$(hfup $D export-$N); lg "export-$N: HF upload job $h -> private InternalCan/aloe-nextvox-6242/export-$N/"
      done_steps="$done_steps$N "
    else
      lg "export-$N: copy NOT verified (files $nf, rank files $nr, ok $ok); retry next round"; echo "$out" | tail -3 >> $BL
    fi
  done
  if [ -f $FIN/COMPLETE ] && [ "$(cat $FIN/COMPLETE)" = 253 ] && grep -q "\[export\] $FIN: consolidated" $LOGF 2>/dev/null; then
    nd=$(ls $FIN | grep -c '^__[0-9]*_0\.distcp$'); ne=$(find $FIN -maxdepth 1 -type f -size 0 | wc -l)
    nk=$($PY -I -c "import pickle,sys; m=pickle.load(open(sys.argv[1]+'/.metadata','rb')).state_dict_metadata; k=list(m); print(len(k), sum(x.startswith('vadapter.') for x in k), sum(x.startswith('vgain.') for x in k), sum(x.startswith('cb0_head') for x in k))" $FIN 2>&1)
    js=$($PY -I -c "import json,sys; j=json.load(open(sys.argv[1]+'/train_cfg.json')); print(j['step'])" $FIN 2>&1)
    if [ "$nd" -eq 64 ] && [ "$ne" -eq 0 ] && [ "$js" = 253 ] && [ "$(echo $nk | cut -d' ' -f1)" -ge 600 ] && [ "$(echo $nk | cut -d' ' -f2)" -eq 7 ] && [ "$(echo $nk | cut -d' ' -f3)" -eq 1 ]; then
      rm -f $DR
      lg "FINAL export $FIN VERIFIED (64 shard files, no empty file, train_cfg step 253, metadata tensors / vadapter / vgain / cb0_head = $nk, $(du -sb $FIN | cut -f1) B); DELETED $DR (TTS retakes the nodes)"
      echo "$FIN" > $Q/FINAL_EXPORT_READY
      echo "- $(pt): FINAL EXPORT (consolidated bf16 DCP, step 253, verified): $FIN  -> FINAL_EXPORT_READY written; drain file deleted." >> $Q/README.md
      h=$(hfup $FIN export-253-dcp); lg "export-253-dcp: HF upload job $h -> private InternalCan/aloe-nextvox-6242/export-253-dcp/"
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

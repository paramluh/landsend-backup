#!/bin/bash
#SBATCH -p all -N1 -n1 -c 1 --mem=8G --gres=none -t 2-00:00:00 -J si20ktxt-watch --exclude=slinky-4,slinky-18,slinky-[0-3,5-6,8-14,16-17,19]
# si20k-txt export watcher (copy / verify / delete own node-local copies / launch only, CPU) = si20k-vox-ppm/tools/watch.sbatch adapted to the
# tight node-local /scratch of the text-out runs:
#  * each intermediate node-local export-N (QQA_EXPORT_STEPS): when the trainer logs it complete and /data has >= 330 GB + its size free (TTS
#    pauses at 320), every node copies its shard files into <run>/exports/export-N (srun --overlap into the FT job; after the FT job ended: a
#    fresh CPU-only srun on the same hosts), sizes verified vs the node-local files, COMPLETE, then the node-local export-N of THIS run is
#    deleted (it now exists verified on /data; /scratch stays >= 30 GB), then the private-HF upload (si20k-txt-ppm/tools/hf_up.sbatch ...
#    delete-after-verify: the /data copy is removed once HF verifies). Several uploads may be in flight; copies are gated by /data space.
#  * final: the consolidated DCP <run>/export-<steps> COMPLETE -> verified (64 shard files, none empty, train_cfg step + textout 1, metadata
#    tensor count >= MINKEYS, vadapter 6 + target_rms, vgain 1, NO audio tensor) -> delete the run's drain file si20k-tts/control/
#    DRAIN_FOR_FT.<tag>, README line, <run>/FINAL_EXPORT_READY, the node-local final export-<steps> copy deleted, HF upload of the final (kept
#    on /data). Exits once the final and every intermediate are handled.
#  * FREE_INIT (PPM: /data/.../local-12829): deleted once the trainer logged step 1 (every rank loaded it; coordinator 10-19: free it; a
#    conversion of the HF-verified node-local dcp-12829, re-created by tools/ckpt_conv_12829_part.sbatch + _finalize in ~2 min).
#  * FT job ends without the final export: FT_ENDED_NO_FINAL; the drain file stays (a relaunch needs the nodes).
#  * FP32 THETA FIX-UP (coordinator 10-19: the trainer saved vgain.theta in bf16, s off by ~1 %): every export copied to /data (intermediate)
#    and the final DCP get tools/theta_fixup.py (the exact fp32 theta of that step from the run's metrics.jsonl, recorded in meta.json /
#    train_cfg.json + THETA_FP32.json) BEFORE their HF upload. Restart-safe: an export already HF-verified (BACKUP_LOG) is skipped; one
#    already copied to /data (COMPLETE) goes straight to the fix-up + HF.
# usage: sbatch -o <run>/logs/watch-%j.out watch_txt.sbatch <run dir> <FT job id> <drain tag> <final export GB> <min metadata keys> [FREE_INIT|-]
Q=$1; J=$2; TAGD=$3; GB=$4; MINK=$5; FREE_INIT=${6:-}; [ "$FREE_INIT" = - ] && FREE_INIT=""
set -a; source $Q/plan/PLAN.env; set +a
RUN=$(basename $Q); EX=$Q/exports; FIN=$QQA_EXPORT; S=$QQA_STEPS; DR=/data/home/amartyavarma/data/si20k-tts/control/DRAIN_FOR_FT.$TAGD
LOC=/scratch/amartyavarma/$RUN/exports; BL=$EX/BACKUP_LOG.txt; HFS=/data/home/amartyavarma/ppm/si20k-txt-ppm/tools/hf_up.sbatch
PY=/data/home/amartyavarma/venvs/qwen-s2/bin/python
mkdir -p $EX
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
lg() { echo "[$(pt)] $*" | tee -a $BL; }
logf() { ls $Q/logs/si20ktxt-*-$J.log 2>/dev/null | head -1; }
hosts() { grep -m1 -o "job $J on [a-z0-9,-]*" $(logf) 2>/dev/null | awk '{print $4}'; }
onnodes() {                                                        # run a command once on every node of the run (FT job if alive, else a fresh CPU srun)
  if squeue -h -j $J -o %T 2>/dev/null | grep -q RUNNING; then
    env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER srun --jobid=$J --overlap -N8 -n8 --ntasks-per-node=1 -c1 --mem=2G --gres=none --cpu-bind=none "$@"
  else
    env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER srun -p all -w $(hosts) -N8 -n8 --ntasks-per-node=1 -c1 --mem=2G --gres=none -t 60 -J si20ktxt-wcopy "$@"
  fi
}
lg "watcher job $SLURM_JOB_ID on $(hostname -s) for FT job $J ($RUN, steps $S, exports $QQA_EXPORT_STEPS, final $FIN (>= $MINK tensors), drain $DR, free-init ${FREE_INIT:-none})"
done_steps=" "; fin_done=0
MET=$Q/runs/$RUN-j$J/metrics.jsonl; FIX=/data/home/amartyavarma/ppm/si20k-txt-ppm/tools/theta_fixup.py
fixup() {                                                          # <export dir> <step>: exact fp32 theta into the export (idempotent)
  r=$($PY -I $FIX $1 --metrics $MET --step $2 --run-dir $Q 2>&1 | grep -v Warn | tail -2 | tr '\n' ' ')
  echo "$r" | grep -q '"status": "\(fixed\|already fp32\)"' && { lg "$(basename $1): fp32 theta FIX-UP ok: $(echo $r | grep -o '"status": "[^"]*"\|"bf16_before": [^,]*\|"theta_fp32": [^,]*\|"s_fp32": [^,]*' | tr '\n' ' ')"; return 0; }
  lg "$(basename $1): fp32 theta FIX-UP FAILED: $(echo $r | cut -c1-300)"; return 1
}
for N in $(echo $QQA_EXPORT_STEPS | tr , ' '); do                  # restart-safe: HF-verified exports are done
  grep -q "export-$N HF upload rc 0" $BL 2>/dev/null && done_steps="$done_steps$N "
done
lg "restart state: done (HF-verified) exports: [$done_steps]"
while true; do
  LOGF=$(logf)
  if [ -n "$FREE_INIT" ] && [ -f "$FREE_INIT/COMPLETE" ] && grep -q "\] step 1/$S " $LOGF 2>/dev/null && grep -q "\[init\] $FREE_INIT:" $LOGF 2>/dev/null; then
    FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
    rm -rf "$FREE_INIT"
    lg "DELETED the /data init copy $FREE_INIT after step 1 (every rank loaded it; /data free was $FD GB); re-creatable from the HF-verified node-local dcp-12829 (tools/ckpt_conv_12829_part.sbatch + _finalize, ~2 min)"
    echo "- $(pt): deleted the /data init copy $FREE_INIT after step 1 (/data free $FD GB; re-creatable from dcp-12829 in ~2 min)" >> $Q/README.md
    FREE_INIT=""
  fi
  st=$(squeue -h -j $J -o %T 2>/dev/null)
  pend=0
  for N in $(echo $QQA_EXPORT_STEPS | tr , ' '); do
    case "$done_steps" in *" $N "*) continue;; esac
    if grep -q "\[export-local\] SKIPPED $LOC/export-$N" $LOGF 2>/dev/null; then lg "export-$N: node-local copy skipped by the trainer (/scratch short)"; done_steps="$done_steps$N "; continue; fi
    grep -q "\[export-local\] $LOC/export-$N on every node" $LOGF 2>/dev/null || { pend=1; continue; }
    if [ -f $EX/export-$N/COMPLETE ] && [ ! -d "$(ls -d $LOC/export-$N 2>/dev/null)" ]; then      # copied + verified before a watcher restart
      if fixup $EX/export-$N $N; then
        hfj=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $EX/export-$N export-$N delete-after-verify); lg "export-$N: HF upload job $hfj (after the fix-up) -> private InternalCan/$RUN/export-$N/"
        done_steps="$done_steps$N "
      else pend=1; fi
      break
    fi
    FD=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9)
    sleep 60; FD2=$(df -BG --output=avail /data | tail -1 | tr -dc 0-9); [ "$FD2" -lt "$FD" ] && FD=$FD2   # 10-19 fix: Weka df lags 30-60 s after a big write (trainer final DCP) -> re-read after 60 s, use the lower value (stale df let xfer-none-ss export-66 copy push /data to 266 GB)
    [ "$FD" -ge $((305 + GB)) ] || { lg "export-$N: waiting for /data space ($FD GB free < 305 + $GB)"; pend=1; break; }   # 10-19 fix: the 330 + GB gate kept 4 PPM exports node-local -> the final export_local filled /scratch (si20k-vox-ppm11488)
    D=$EX/export-$N; mkdir -p $D
    t0=$(date +%s)
    out=$(onnodes bash -c "cd $LOC/export-$N && for f in rank*.pt rep.pt meta.json; do [ -f \$f ] || continue; nice -n 10 cp -p \$f $D/\$f.tmp && mv $D/\$f.tmp $D/\$f && echo \"\$f \$(stat -c %s \$f)\"; done" 2>&1)
    ok=1; nf=0
    while read -r f s; do
      case "$f" in rank*.pt|rep.pt|meta.json) ;; *) continue;; esac
      nf=$((nf + 1)); [ "$(stat -c %s $D/$f 2>/dev/null)" = "$s" ] || { ok=0; lg "export-$N: size mismatch $f"; }
    done <<< "$out"
    nr=$(ls $D | grep -c '^rank[0-9]*\.pt$')
    if [ $ok -eq 1 ] && [ $nr -eq 64 ] && [ -f $D/rep.pt ] && [ -f $D/meta.json ] && [ $nf -eq 66 ]; then
      echo $N > $D/COMPLETE
      lg "export-$N -> Weka $D: COPIED + VERIFIED ($nf files, $(du -sb $D | cut -f1) B, sizes == node-local; $(( $(date +%s) - t0 ))s); /data free $(df -BG --output=avail /data | tail -1 | tr -dc 0-9) GB"
      rmo=$(onnodes bash -c "source /data/home/amartyavarma/ppm/scratch_guard.sh; guard_rm $LOC/export-$N && echo \"\$(hostname -s) deleted $LOC/export-$N, /scratch free \$(df -BG --output=avail /scratch | tail -1 | tr -dc 0-9) GB\"" 2>&1 | grep deleted | sort -V | tr '\n' ';')
      lg "export-$N: node-local copies DELETED (verified on /data): $rmo"
      if fixup $D $N; then
        hfj=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $D export-$N delete-after-verify); lg "export-$N: HF upload job $hfj -> private InternalCan/$RUN/export-$N/ (then the /data copy is deleted)"
        done_steps="$done_steps$N "
      else pend=1; fi
    else
      lg "export-$N: copy NOT verified (files $nf, rank files $nr, ok $ok); retry next round"; echo "$out" | tail -3 >> $BL; pend=1
    fi
    break
  done
  if [ $fin_done -eq 0 ] && [ -f $FIN/COMPLETE ] && [ "$(cat $FIN/COMPLETE)" = "$S" ] && grep -q "\[export\] $FIN: consolidated" $LOGF 2>/dev/null; then
    nd=$(ls $FIN | grep -c '^__[0-9]*_0\.distcp$'); ne=$(find $FIN -maxdepth 1 -type f -size 0 | wc -l)
    nk=$($PY -I -c "import pickle,sys; m=pickle.load(open(sys.argv[1]+'/.metadata','rb')).state_dict_metadata; k=list(m); print(len(k), sum(x.startswith('vadapter.') for x in k), sum(x.startswith('vgain.') for x in k), sum(x.startswith(('inp.aud.', 'tower_s.', 'dec_s.', 'cb0_head.')) for x in k))" $FIN 2>&1)
    js=$($PY -I -c "import json,sys; j=json.load(open(sys.argv[1]+'/train_cfg.json')); print(j['step'], j.get('textout'))" $FIN 2>&1)
    if [ "$nd" -eq 64 ] && [ "$ne" -eq 0 ] && [ "$js" = "$S 1" ] && [ "$(echo $nk | cut -d' ' -f1)" -ge $MINK ] && [ "$(echo $nk | cut -d' ' -f2)" -eq 7 ] && [ "$(echo $nk | cut -d' ' -f3)" -eq 1 ] && [ "$(echo $nk | cut -d' ' -f4)" -eq 0 ]; then
      fixup $FIN $S || { sleep 15; continue; }
      rm -f $DR
      lg "FINAL export $FIN VERIFIED (64 shard files, no empty file, train_cfg step / textout = $js, metadata tensors / vadapter / vgain / audio = $nk, $(du -sb $FIN | cut -f1) B); DELETED $DR (TTS / other jobs retake the nodes)"
      echo "$FIN" > $Q/FINAL_EXPORT_READY
      echo "- $(pt): FINAL EXPORT (consolidated bf16 DCP, step $S, text-out, verified): $FIN  -> FINAL_EXPORT_READY written; drain file deleted." >> $Q/README.md
      h=$(sbatch --parsable -o $Q/logs/hf-%j.out $HFS $Q $FIN export-$S-dcp); lg "export-$S-dcp: HF upload job $h -> private InternalCan/$RUN/export-$S-dcp/ (kept on /data)"
      rmo=$(onnodes bash -c "source /data/home/amartyavarma/ppm/scratch_guard.sh; [ -d $LOC/export-$S ] && guard_rm $LOC/export-$S && echo \"\$(hostname -s) deleted $LOC/export-$S\"" 2>&1 | grep -c deleted)
      lg "node-local final export-$S copies deleted on $rmo nodes (the consolidated DCP on /data is verified; HF upload running)"
      fin_done=1
    else
      lg "final export not verified yet (shards $nd, empty $ne, step/textout $js, keys $nk); retry"
    fi
  fi
  if [ $fin_done -eq 1 ] && [ $pend -eq 0 ]; then lg "watcher done (final + every intermediate handled)"; exit 0; fi
  if [ -z "$st" ] && [ $fin_done -eq 0 ]; then
    sleep 20
    if ! { [ -f $FIN/COMPLETE ] && grep -q "\[export\] $FIN: consolidated" $LOGF 2>/dev/null; }; then
      lg "FT job $J ended WITHOUT the final export (last log: $(grep -v Warning $LOGF | tail -1 | cut -c1-200)); drain file kept for a relaunch"
      echo "$(pt) FT job $J ended without the final export" > $Q/FT_ENDED_NO_FINAL; exit 1
    fi
  fi
  sleep 15
done

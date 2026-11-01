#!/bin/bash
# evalvox chain, RUN-SCOPED instance (PPMVOX v2+, 10-24) = chain.sh with a run prefix: markers in $EVALVOX_DONE_DIR named <tag>-<host> (tag w-N,
# $EVALVOX_NHOSTS of them = HF upload + verify done on every host), HF files at $EVALVOX_HF_REPO <tag>/<host>/ (v2: InternalCan/ppmvox-v2-16861).
# Key K = $EVALVOX_RUN-<tag> (e.g. ppmvox-v2-w-16911), staged at /scratch/amartyavarma/evalvox/$EVALVOX_RUN/<tag>, for state / env / staging dir / run name. Then as chain.sh: stage on slinky-18 / slinky-4,
# eval-only job afterok, word CE -> RESULTS.jsonl (with "run"), staged copies deleted after the eval (main's go; LISTEN_WANTED hold honoured).
E=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/evalvox; ST=$E/state; L=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs
PD=${EVALVOX_DONE_DIR:?}; RUNL=${EVALVOX_RUN:?}; REPO=${EVALVOX_HF_REPO:?}; NH=${EVALVOX_NHOSTS:-16}
mkdir -p $ST
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]+'); do unset $v; done
echo "[$(pt)] chain2 for $RUNL: markers $PD, HF repo $REPO, $NH hosts"
while true; do
  for t in $(ls $PD 2>/dev/null | grep -oE '^w-[0-9]+' | sort -u -t- -k2 -n); do
    FT=$RUNL/$t; K=$RUNL-$t
    [ -f $ST/$K.submitted ] && continue
    [ $(ls $PD | grep -c "^$t-") -ge $NH ] || continue
    python3 -I $E/mkenv.py $FT > /dev/null || { echo "[$(pt)] $FT mkenv FAILED"; continue; }
    j1=$(sbatch --parsable --export=ALL,EVALVOX_HF_REPO=$REPO -w slinky-18 $E/stage.sbatch $t 0 /scratch/amartyavarma/evalvox/$FT); j2=$(sbatch --parsable --export=ALL,EVALVOX_HF_REPO=$REPO -w slinky-4 $E/stage.sbatch $t 1 /scratch/amartyavarma/evalvox/$FT)
    j3=$(env QQA_ENV=$E/env-$K.env COTRAIN_RESUME= sbatch --parsable -d afterok:$j1:$j2 -N2 -w slinky-4,slinky-18 --ntasks-per-node=8 --cpus-per-task=4 --mem=520G $E/run_eval.sbatch)
    printf '{"run": "%s", "tag": "%s", "key": "%s", "stage": ["%s", "%s"], "eval_job": "%s", "time": "%s"}\n' $RUNL $t $K $j1 $j2 $j3 "$(pt)" > $ST/$K.submitted
    echo "[$(pt)] $FT: HF verified on $NH hosts -> stage $j1 $j2, eval job $j3"
  done
  for f in $(ls $ST/$RUNL-*.submitted 2>/dev/null); do
    K=$(basename $f .submitted); [ -f $ST/$K.done ] && continue; t=${K#$RUNL-}
    j=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_job"])' $f)
    squeue -h -j $j 2>/dev/null | grep -q . && continue
    s=$(sacct -nX -j $j -o State | head -1 | tr -d ' ')
    m=$(ls -d /data/home/amartyavarma/ppm/aloe-ft-32-samsep/runs/evalvox-ppmvox-$K-j$j 2>/dev/null)/metrics.jsonl
    ce=$(python3 -I -c 'import json,sys
v=None
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    if "eval_speech/word_ce_norm" in d: v=d["eval_speech/word_ce_norm"]
print("" if v is None else f"{v:.4f}")' $m 2>/dev/null)
    printf '{"run": "%s", "tag": "%s", "eval_job": "%s", "state": "%s", "word_ce_norm": "%s", "time": "%s"}\n' $RUNL $t $j "$s" "${ce:-NA}" "$(pt)" | tee -a $E/RESULTS.jsonl > $ST/$K.done
    echo "[$(pt)] $RUNL $t eval job $j $s: word_ce_norm ${ce:-NA}"
  done
  for f in $(ls $ST/$RUNL-*.done 2>/dev/null); do
    K=$(basename $f .done); [ -f $ST/$K.cleaned ] && continue; t=${K#$RUNL-}
    [ -f $E/DELETE_STAGED_OK ] || continue
    if [ -f $E/LISTEN_WANTED ] && [ ! -f $ST/$K.listen_done ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -lt 10800 ]; then continue; fi
    for h in slinky-18 slinky-4; do sbatch -w $h -c1 --mem=512M -t 10 -J evalvox-clean -o $L/evalvox-clean-%j.log --wrap "bash $E/clean_staged.sh /scratch/amartyavarma/evalvox/$RUNL/$t 'PPMVOX $RUNL export $t from private HF $REPO'" > /dev/null; done   # 10-24 fix: --wrap runs /bin/sh (no source): the deletion lives in clean_staged.sh
    touch $ST/$K.cleaned; echo "[$(pt)] $RUNL $t staged copies: cleanup submitted"
  done
  sleep 60
done

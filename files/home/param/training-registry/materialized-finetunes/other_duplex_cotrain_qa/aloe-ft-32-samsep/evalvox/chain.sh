#!/bin/bash
# evalvox chain (10-24, main: "eval every PPMVOX export on Aloe as it lands"): CPU-only loop (run as a 1-CPU Slurm job, survives login restarts).
# Every 60 s: each PPMVOX export w-N whose HF upload + verify is done on ALL its hosts (ppmvox companions/done/w-N-<host>, 16 markers) and not yet
# handled -> env (mkenv.py) + staging from HF on slinky-18 / slinky-4 (stage.sbatch; waits for /scratch room) + the eval-only job afterok
# (run_eval.sbatch, slinky-18 x8 + slinky-4 GPUs 0-5; pends while the GPUs are busy = idle fallback). When an eval job ends: its normalised
# word CE -> evalvox/RESULTS.jsonl. Staged copies are deleted after the eval ONLY if evalvox/DELETE_STAGED_OK exists (main's go).
E=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/evalvox; PD=/data/home/amartyavarma/ppm/ppmvox/companions/done; ST=$E/state; L=/data/home/amartyavarma/ppm/aloe-ft-32-samsep/logs
mkdir -p $ST; NH=${EVALVOX_NHOSTS:-16}
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
for v in $(env | grep -oE '^(SLURM|SBATCH|SRUN)_[A-Za-z0-9_]+'); do unset $v; done
while true; do
  for t in $(ls $PD 2>/dev/null | grep -oE '^w-[0-9]+' | sort -u -t- -k2 -n); do
    [ -f $ST/$t.submitted ] && continue
    [ $(ls $PD | grep -c "^$t-") -ge $NH ] || continue
    python3 -I $E/mkenv.py $t > /dev/null || { echo "[$(pt)] $t mkenv FAILED"; continue; }
    j1=$(sbatch --parsable -w slinky-18 $E/stage.sbatch $t 0); j2=$(sbatch --parsable -w slinky-4 $E/stage.sbatch $t 1)
    j3=$(env QQA_ENV=$E/env-$t.env COTRAIN_RESUME= sbatch --parsable -d afterok:$j1:$j2 -N2 -w slinky-4,slinky-18 --ntasks-per-node=8 --cpus-per-task=4 --mem=520G $E/run_eval.sbatch)
    echo "{\"tag\": \"$t\", \"stage\": [\"$j1\", \"$j2\"], \"eval_job\": \"$j3\", \"time\": \"$(pt)\"}" > $ST/$t.submitted
    echo "[$(pt)] $t: HF verified on $NH hosts -> stage $j1 $j2, eval job $j3"
  done
  for f in $(ls $ST/*.submitted 2>/dev/null); do
    t=$(basename $f .submitted); [ -f $ST/$t.done ] && continue
    j=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["eval_job"])' $f)
    squeue -h -j $j 2>/dev/null | grep -q . && continue
    s=$(sacct -nX -j $j -o State | head -1 | tr -d ' ')
    m=$(ls -d /data/home/amartyavarma/ppm/aloe-ft-32-samsep/runs/evalvox-ppmvox-$t-j$j 2>/dev/null)/metrics.jsonl
    ce=$(python3 -I -c 'import json,sys
v=None
for l in open(sys.argv[1]):
    try: d=json.loads(l)
    except Exception: continue
    if "eval_speech/word_ce_norm" in d: v=d["eval_speech/word_ce_norm"]
print("" if v is None else f"{v:.4f}")' $m 2>/dev/null)
    echo "{\"tag\": \"$t\", \"eval_job\": \"$j\", \"state\": \"$s\", \"word_ce_norm\": \"${ce:-NA}\", \"time\": \"$(pt)\"}" | tee -a $E/RESULTS.jsonl > $ST/$t.done
    echo "[$(pt)] $t eval job $j $s: word_ce_norm ${ce:-NA}"
  done
  # cleanup (main's go: evalvox/DELETE_STAGED_OK): a staged copy is deleted after its Aloe eval; if PPMVOX's listening eval registered
  # (evalvox/LISTEN_WANTED) it is kept until state/<tag>.listen_done appears (written by that job) or 3 h after the Aloe eval
  for f in $(ls $ST/*.done 2>/dev/null); do
    t=$(basename $f .done); [ -f $ST/$t.cleaned ] && continue
    [ -f $E/DELETE_STAGED_OK ] || continue
    if [ -f $E/LISTEN_WANTED ] && [ ! -f $ST/$t.listen_done ] && [ $(( $(date +%s) - $(stat -c %Y $f) )) -lt 10800 ]; then continue; fi
    for h in slinky-18 slinky-4; do sbatch -w $h -c1 --mem=512M -t 10 -J evalvox-clean -o $L/evalvox-clean-%j.log --wrap "source /data/home/amartyavarma/ppm/scratch_guard.sh; d=/scratch/amartyavarma/evalvox/$t; [ -f \$d/EVALVOX_STAGE.json ] && sz=\$(du -sh \$d | cut -f1) && guard_rm \$d && echo \"[\$(TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT')] evalvox (ALOEFT32, main go): deleted \$(hostname -s):\$d (\$sz, staged copy of PPMVOX export $t from private HF InternalCan/ppmvox-16861, after its Aloe eval)\" >> /data/home/amartyavarma/setup-logs/hfb/DELETED-1021-hfb-hold.log" > /dev/null; done
    touch $ST/$t.cleaned; echo "[$(pt)] $t staged copies: cleanup submitted"
  done
  sleep 60
done

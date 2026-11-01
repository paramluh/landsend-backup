#!/bin/bash
#SBATCH -J npl5-plan -p all -x slinky-[0-3,5-9,11-17,19] -N1 -n1 -c 6 --mem=44G --gres=none -t 1:30:00 --time-min=0:45:00 --deadline=2026-10-27T17:52:00
#SBATCH -o /data/home/amartyavarma/ppm/pose5/npl5/logs/plan-%j.out
# npl5 off-host chain, auto-submitted by the last staging lane (spares only: slinky-4/10/18 by -x; never the 16 hosts, never slinky-9):
#   1 plan build (bin/plan_offhost.py)   2 off-host dry run, append at 17850, final env (bin/dryrun_offhost.sh)
#   3 the rolling-fetcher WINDOW files for C's gate: every key x ahead {100,80,60,40}: back 110 -> roll/, back 53 (newest-save mode) -> roll/alt/
#   4 submits the launch-day rehearsal (slot 9 on slinky-4) and then the 16-slot end-to-end dry run (slinky-4, after the rehearsal)
W=/data/home/amartyavarma/ppm/pose5/npl5
pt() { TZ=America/Los_Angeles date '+%m-%d %H:%M:%S PT'; }
t0=$(date +%s)
nice -n19 ionice -c3 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I $W/bin/plan_offhost.py --pool ${POOL:-5} || { echo "== $(pt) PLAN FAILED"; exit 1; }
printf "plan\t-\t%ss\t%s\t%s\n" "$(( $(date +%s) - t0 ))" "$(hostname -s)" "$(pt)" >> $W/TIMES.tsv
echo "$(pt) plans built" > $W/state/plan.ok; echo "== $(pt) plans built"
bash $W/bin/dryrun_offhost.sh || exit 1
END=$(python3 -c "import json;print(json.load(open('$W/dryrun/RESULT.json'))['max_steps'])")
t0=$(date +%s); mkdir -p $W/roll/alt $W/roll/plan-logs
for K in slinky-0 slinky-1 slinky-2 slinky-3 slinky-5 slinky-6 slinky-7 slinky-8 slinky-9 slinky-11 slinky-12 slinky-13 slinky-14 slinky-16 slinky-17 slinky-19; do
  for b in 110 53; do echo "$K $b"; done; done | \
  xargs -P 6 -n 2 bash -c 'K=$0; b=$1; P=$([ $K = slinky-9 ] && echo slinky-15 || echo $K); D=$([ $b = 110 ] && echo '$W'/roll || echo '$W'/roll/alt)
    NPL5_ROLL_DIR='$W'/roll/plan-logs nice -n19 /data/home/amartyavarma/venvs/qwen-s2/bin/python -I '$W'/bin/npl5_roll.py plan --end '$END' --phys $P --key $K \
      --root '$W'/plans/$K --audit13 --back $b --alt-dir $D > '$W'/roll/plan-logs/plan-$K-b$b.out 2>&1 || echo "WINDOW FAILED $K $b"'
echo "== $(pt) WINDOW files: $(ls $W/roll/WINDOW-*.json | wc -l) in roll/, $(ls $W/roll/alt/WINDOW-*.json | wc -l) in roll/alt/ ($(( $(date +%s) - t0 )) s)"
printf "windows\t-\t%ss\t%s\t%s\n" "$(( $(date +%s) - t0 ))" "$(hostname -s)" "$(pt)" >> $W/TIMES.tsv
if [ -f $W/state/dryrun.ok ] && [ "${NO_REHEARSE:-0}" != 1 ]; then
  for v in $(env | grep -oE '^(SLURM|SBATCH)_[A-Za-z0-9_]*'); do unset $v; done
  R=$(sbatch --parsable $W/bin/rehearse.sbatch); E=$(sbatch --parsable --dependency=afterany:$R $W/bin/e2e.sbatch)
  echo "== $(pt) launch-day rehearsal job $R, then 16-slot e2e job $E (slinky-4)"
fi
